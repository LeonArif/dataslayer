"""Hand channels from the RTMW whole-body keypoints (cache/rtmw/kp_{train,test}.npy, (n, 30, 133, 3) = x, y, score; hands = 91-111 left, 112-132 right, COCO-WholeBody).
Per frame and hand: 20 finger points relative to the wrist, divided by 0.08 x person box height (40 values), the mean keypoint score of the hand (1), the hand centre relative to the nose in box heights (2) = 43 per hand, 86 per frame.
Output cache/hands/flow_series_{train,test}.npy (same name as the flow series so e3_generic / e58_generic read it with FLOWDIR). usage (from leon/): python scripts/make_hand_series.py"""
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "cache/hands"; OUT.mkdir(exist_ok=True)
import pandas as pd
clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")
meta = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_train_s30_meta.csv"); boxall = np.load(ROOT / "cache/pose_vitpose_all/box_train_s30.npy")
meta["key"] = [("cheat/" if l == 1 else "normal/") + v.replace(".MOV", "") for v, l in zip(meta.video, meta.label)]; look = {(k, s): i for i, (k, s) in enumerate(zip(meta.key, meta.start))}
boxtr = np.stack([boxall[look[(v, int(sec) * 30)]] for v, sec in zip(clips.video, clips.second)]); boxte = np.load(ROOT / "cache/pose_vitpose_all/box_test.npy")
def feats(kp, box):
    kp = kp.astype(np.float32); h = box[:, 3][:, None, None, None] + 1e-6; out = []                  # box xywh -> height
    nose = kp[:, :, 0, :2]
    for a, b in ((91, 112), (112, 133)):
        pts = kp[:, :, a:b, :2]; sc = kp[:, :, a:b, 2]; rel = (pts[:, :, 1:] - pts[:, :, :1]) / (0.08 * h); cen = (pts.mean(2) - nose) / h[:, :, 0]
        out += [rel.reshape(len(kp), 30, 40), sc.mean(2)[..., None], cen]
    return np.nan_to_num(np.concatenate(out, 2)).astype(np.float32)
for name, box in (("train", boxtr), ("test", boxte)):
    kp = np.load(ROOT / "cache/rtmw" / f"kp_{name}.npy"); f = feats(kp, box); assert f.shape[:2] == (len(kp), 30); np.save(OUT / f"flow_series_{name}.npy", f); print(name, f.shape, "finite", np.isfinite(f).all())
