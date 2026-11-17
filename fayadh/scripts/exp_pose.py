"""Eksperimen fitur pose (scratch). Kode fitur yang dipakai sama dengan main.ipynb."""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # folder fayadh/
DATA_DIR = os.path.join(ROOT, "..", "data")
CACHE_DIR = os.path.join(ROOT, "cache", "main")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
from lightgbm import LGBMClassifier

S = sys.argv[1] if len(sys.argv) > 1 else "30"
KP = np.load(f"{CACHE_DIR}/pose_train_s{S}.npy")          # (N, 10, 17, 3)
BOX = np.load(f"{CACHE_DIR}/box_train_s{S}.npy")          # (N, 4) xywh
meta = pd.read_csv(f"{CACHE_DIR}/pose_train_s{S}_meta.csv")
BEHAVIORS = sorted(meta["behavior"].unique())
CHEAT_B = set(meta.loc[meta.label == 1, "behavior"])
meta["beh_id"] = meta["behavior"].map({b: i for i, b in enumerate(BEHAVIORS)})

NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12


def pose_features(kp, box):
    """kp: (N, T, 17, 3) dalam piksel; box: (N, 4) xywh. Return DataFrame fitur per window."""
    N, T = kp.shape[:2]
    bx, by, bw, bh = [box[:, i][:, None, None] for i in range(4)]
    x = (kp[..., 0] - bx) / bh                      # (N, T, 17) dinormalisasi tinggi box
    y = (kp[..., 1] - by) / bh
    s = kp[..., 2]
    msx, msy = (x[..., LSH] + x[..., RSH]) / 2, (y[..., LSH] + y[..., RSH]) / 2
    sh_w = np.abs(x[..., LSH] - x[..., RSH]) + 1e-3
    ear_mx = (x[..., LEAR] + x[..., REAR]) / 2
    eye_mx = (x[..., LEYE] + x[..., REYE]) / 2
    series = {
        # kepala
        "nose_dx": x[..., NOSE] - msx, "nose_dy": y[..., NOSE] - msy,
        "head_yaw": (x[..., NOSE] - ear_mx) / sh_w, "eye_yaw": (x[..., NOSE] - eye_mx) / sh_w,
        "eye_ear_dy": (y[..., LEYE] + y[..., REYE]) / 2 - (y[..., LEAR] + y[..., REAR]) / 2,
        "ear_vis_diff": s[..., LEAR] - s[..., REAR], "nose_score": s[..., NOSE],
        # tangan
        "lwr_dx": x[..., LWR] - msx, "lwr_dy": y[..., LWR] - msy, "rwr_dx": x[..., RWR] - msx, "rwr_dy": y[..., RWR] - msy,
        "lwr_nose": np.hypot(x[..., LWR] - x[..., NOSE], y[..., LWR] - y[..., NOSE]),
        "rwr_nose": np.hypot(x[..., RWR] - x[..., NOSE], y[..., RWR] - y[..., NOSE]),
        "wr_dist": np.hypot(x[..., LWR] - x[..., RWR], y[..., LWR] - y[..., RWR]),
        "lel_dy": y[..., LEL] - msy, "rel_dy": y[..., REL] - msy,
        "lwr_score": s[..., LWR], "rwr_score": s[..., RWR],
        # badan
        "sh_w": sh_w, "sh_tilt": y[..., LSH] - y[..., RSH], "ms_y": msy, "ms_x": msx,
        "hip_y": (y[..., LHIP] + y[..., RHIP]) / 2 - msy,
        "kp_score": s.mean(-1),
    }
    feats = {}
    for k, v in series.items():
        feats[f"{k}_mean"] = v.mean(1); feats[f"{k}_std"] = v.std(1)
        feats[f"{k}_min"] = v.min(1); feats[f"{k}_max"] = v.max(1)
        feats[f"{k}_dabs"] = np.abs(np.diff(v, axis=1)).mean(1)
    # kecepatan keypoint (gerakan) per bagian tubuh
    vel = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))          # (N, T-1, 17)
    for name, idx in (("head", [0, 1, 2, 3, 4]), ("arm_l", [7, 9]), ("arm_r", [8, 10]), ("torso", [5, 6, 11, 12])):
        v = vel[..., idx].mean(-1)
        feats[f"vel_{name}_mean"] = v.mean(1); feats[f"vel_{name}_max"] = v.max(1)
    feats["box_ar"] = box[:, 2] / box[:, 3]; feats["box_h"] = box[:, 3] / 360; feats["box_cy"] = (box[:, 1] + box[:, 3] / 2) / 360
    return pd.DataFrame(feats)


def f1m(y, p):
    return f1_score(y, p, average="macro")


def group_postprocess(vdf, Pv, mode):
    out = np.zeros(len(vdf), int)
    for _, g in vdf.groupby(["subject", "angle"]):
        idx = vdf.index.get_indexer(g.index)
        if mode == "bal":
            p = Pv[idx][:, CHEAT_IDX].sum(1)
            out[idx[np.argsort(-p)[: len(idx) // 2]]] = 1
        else:
            r, c = linear_sum_assignment(-np.log(Pv[idx] + 1e-9))
            out[idx[r]] = [int(BEHAVIORS[k] in CHEAT_B) for k in c]
    return out


CHEAT_IDX = [i for i, b in enumerate(BEHAVIORS) if b in CHEAT_B]
LGBM = dict(objective="multiclass", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=42, verbose=-1)


def run(X, name, params=LGBM, add_ctx=False):
    X = X.copy()
    if add_ctx:
        # konteks video: rata-rata fitur semua window di video (di test = rantai), dan deviasi window dari rata-rata itu
        vm = X.groupby(meta["video"]).transform("mean")
        X = pd.concat([X, vm.add_prefix("vid_"), (X - vm).add_prefix("dev_")], axis=1)
    y = meta["beh_id"].values
    P = np.zeros((len(X), len(BEHAVIORS)))
    vids = meta[["video", "subject", "label"]].drop_duplicates("video").reset_index(drop=True)
    for tr_v, va_v in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(vids, vids["label"], vids["subject"]):
        tr = meta["video"].isin(vids["video"][tr_v]).values
        va = ~tr
        P[va] = LGBMClassifier(**params).fit(X[tr], y[tr]).predict_proba(X[va])
    p_clip = P[:, CHEAT_IDX].sum(1)
    res = {"name": name, "clip": f1m(meta["label"], p_clip > 0.5)}
    logP = pd.DataFrame(np.log(P + 1e-9)).groupby(meta["video"].values).mean()
    Pv = np.exp(logP.values); Pv /= Pv.sum(1, keepdims=True)
    vdf = meta.drop_duplicates("video").set_index("video").loc[logP.index].reset_index()
    res["video"] = f1m(vdf["label"], Pv[:, CHEAT_IDX].sum(1) > 0.5)
    res["video_bal"] = f1m(vdf["label"], group_postprocess(vdf, Pv, "bal"))
    res["video_hung"] = f1m(vdf["label"], group_postprocess(vdf, Pv, "hung"))
    res["beh_acc_clip"] = (P.argmax(1) == y).mean()
    res["beh_acc_video"] = (Pv.argmax(1) == vdf["beh_id"]).mean()
    print({k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in res.items()}, flush=True)
    return P


if __name__ == "__main__":
    X = pose_features(KP, BOX)
    print("windows", len(X), "features", X.shape[1])
    run(X, "pose")
    run(X, "pose+ctx", add_ctx=True)
