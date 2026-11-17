"""Eksperimen cepat di atas embedding DINOv2 yang sudah di-cache (scratch, hasil dicatat di logs.md)."""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # folder fayadh/
DATA_DIR = os.path.join(ROOT, "..", "data")
CACHE_DIR = os.path.join(ROOT, "cache", "dino")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment

TAG = sys.argv[1] if len(sys.argv) > 1 else "dinov2-small"
E = np.load(f"{CACHE_DIR}/dino_{TAG}_train.npy").astype(np.float32)      # (N_frames, 14, D)
meta = pd.read_csv(f"{CACHE_DIR}/dino_{TAG}_train_meta.csv")
BEHAVIORS = sorted(meta["behavior"].unique())
CHEAT_B = set(meta.loc[meta.label == 1, "behavior"])
meta["beh_id"] = meta["behavior"].map({b: i for i, b in enumerate(BEHAVIORS)})
offs = np.r_[0, np.cumsum(meta["n_emb"])]
W = 5  # 5 frame embedding = 1 detik


def clip_feat(e, mode):
    # e: (W, 14, D)
    parts = [e[:, :2].mean(0).ravel()]
    if "grid" in mode:
        parts.append(e[:, 2:].mean(0).ravel())
    if "std" in mode:
        parts.append(e[:, :2].std(0).ravel())
    if "diff" in mode:
        parts.append(np.abs(np.diff(e[:, 2:], axis=0)).mean(0).ravel())
    return np.concatenate(parts)


def build(mode, stride):
    X, rows = [], []
    for v, r in meta.iterrows():
        ev = E[offs[v] : offs[v + 1]]
        for s in range(0, len(ev) - W + 1, stride):
            X.append(clip_feat(ev[s : s + W], mode))
            rows.append((v, s))
    return np.stack(X), pd.DataFrame(rows, columns=["vid", "s"])


def f1m(y, p):
    return f1_score(y, p, average="macro")


def balanced_assign(vdf, col):
    # per (subject, angle): 4 video cheat tertinggi -> cheat
    out = np.zeros(len(vdf), int)
    for _, g in vdf.groupby(["subject", "angle"]):
        top = g[col].sort_values(ascending=False).index[:4]
        out[vdf.index.get_indexer(top)] = 1
    return out


def hungarian(vdf, probs):
    # per (subject, angle): tiap perilaku tepat satu video
    out = np.zeros(len(vdf), int)
    for _, g in vdf.groupby(["subject", "angle"]):
        idx = vdf.index.get_indexer(g.index)
        r, c = linear_sum_assignment(-np.log(probs[idx] + 1e-9))
        out[idx[r]] = [int(BEHAVIORS[k] in CHEAT_B) for k in c]
    return out


def run(mode, C=0.1, pca=256, multiclass=True):
    Xtr, idx = build(mode, stride=1)
    Xev, idx_ev = build(mode, stride=W)       # klip tidak overlap seperti test
    v_tr, v_ev = idx["vid"].values, idx_ev["vid"].values
    y_tr = meta["beh_id"].values[v_tr] if multiclass else meta["label"].values[v_tr]
    groups = meta["subject"].values
    n_cls = len(BEHAVIORS) if multiclass else 2
    P = np.zeros((len(Xev), n_cls))
    for tr_s, va_s in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(meta, meta["label"], groups):
        tr_m, va_m = np.isin(v_tr, tr_s), np.isin(v_ev, va_s)
        steps = [StandardScaler()] + ([PCA(pca, random_state=42)] if pca else []) + [LogisticRegression(C=C, max_iter=2000)]
        m = make_pipeline(*steps).fit(Xtr[tr_m], y_tr[tr_m])
        P[va_m] = m.predict_proba(Xev[va_m])
    cheat_cols = [i for i, b in enumerate(BEHAVIORS) if b in CHEAT_B] if multiclass else [1]
    p_clip = P[:, cheat_cols].sum(1)
    y_clip = meta["label"].values[v_ev]
    res = dict(mode=mode, C=C, pca=pca, mc=multiclass, clip=f1m(y_clip, p_clip > 0.5))
    # video: rata-rata log-prob per video
    vdf = meta.copy()
    logP = pd.DataFrame(np.log(P + 1e-9)).groupby(v_ev).mean()
    Pv = np.exp(logP.values); Pv /= Pv.sum(1, keepdims=True)
    vdf["p"] = Pv[:, cheat_cols].sum(1)
    res["video"] = f1m(vdf["label"], vdf["p"] > 0.5)
    res["video_bal"] = f1m(vdf["label"], balanced_assign(vdf, "p"))
    if multiclass:
        res["video_hung"] = f1m(vdf["label"], hungarian(vdf, Pv))
        res["beh_acc_video"] = (Pv.argmax(1) == vdf["beh_id"]).mean()
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    return res


if __name__ == "__main__":
    run("base", C=0.1, pca=256, multiclass=False)
    run("base", C=0.1, pca=256)
    run("base+grid", C=0.1, pca=256)
    run("base+grid+std+diff", C=0.1, pca=256)
    run("base+grid", C=0.02, pca=512)
