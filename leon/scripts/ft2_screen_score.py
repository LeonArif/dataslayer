# Score the held-out fold of a fine-tune run without the logistic regression: the head's own cheat probability (sum of the 4 cheat action classes) averaged over
# the clip's 2 frames (frames 0 and 15), compared with the clip label. Used to screen augmentations on one fold (cheap proxy; the final choice is re-scored with member B).
# usage (from leon/): python scripts/ft2_screen_score.py RUN [RUN ...]   (RUN = folder under cache/ft2, files k12_f0_val.npy; env K, FOLD)
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
K, FOLD = int(os.environ.get("K", 12)), int(os.environ.get("FOLD", 0))
meta = pd.read_csv(ROOT / "cache/ft_frames_testfmt/train_meta.csv"); meta["second"] = meta["frame"] // 30
clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")[["video", "second", "label", "fold"]]
rows = []
for run in sys.argv[1:]:
    f = ROOT / "cache/ft2" / run / f"k{K}_f{FOLD}_val.npy"
    if not f.exists():
        continue
    v = np.load(f); idx = v[:, 0].astype(int); p = np.exp(v[:, 1:])
    d = meta.iloc[idx].assign(pc=p[:, :4].sum(1), pred=p.argmax(1))
    c = d.groupby(["video", "second"], as_index=False)["pc"].mean().merge(clips, on=["video", "second"], how="inner")
    thr = max(np.arange(0.2, 0.81, 0.025), key=lambda t: f1_score(c.label, c.pc > t, average="macro"))
    rows.append(dict(run=run, clips=len(c), auc=round(roc_auc_score(c.label, c.pc), 4), f1_at_0_5=round(f1_score(c.label, c.pc > 0.5, average="macro"), 4),
                     f1_best_thr=round(f1_score(c.label, c.pc > thr, average="macro"), 4), thr=round(thr, 3)))
print(pd.DataFrame(rows).to_string(index=False))
