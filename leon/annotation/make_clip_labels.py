# Turn the per-second labels of the cheat FRONT videos (annotation/annotations.csv) into clip labels for every clip of cache/members_2026-11-20/clips.csv.
# Front cheat video: the label of that second (1 cheat, 0 normal, -1 unsure). Side cheat video (FAQ #50.2: front annotation shifted onto the side video, camera lag 0.5 s as in ft2_train.py):
# side second s covers front time [s-0.5, s+0.5), i.e. half of front second s-1 and half of front second s: both 1 -> 1, both 0 -> 0, anything else -> -1 (unsure, left out of training and scoring).
# Normal videos: 0. Output annotation/clip_labels_ai.csv (video, second, label_new) in the row order of clips.csv.
# usage (from leon/): python annotation/make_clip_labels.py
import numpy as np, pandas as pd
c = pd.read_csv("../cache/members_2026-11-20/clips.csv"); a = pd.read_csv("annotation/annotations.csv")
lab = {(v, s): l for v, s, l in zip(a.video, a.second, a.label)}
out = []
for v, s, y in zip(c.video, c.second, c.label):
    if v.startswith("normal/"): out.append(0); continue
    if "/front_" in v: out.append(lab.get((v, s), -1)); continue
    f = v.replace("/side_", "/front_"); p, q = lab.get((f, s - 1), 0 if s - 1 < 3 else -1), lab.get((f, s), -1)
    out.append(1 if (p == 1 and q == 1) else 0 if (p == 0 and q == 0) else -1)
c["label_new"] = out
c[["video", "second", "label_new"]].to_csv("annotation/clip_labels_ai.csv", index=False)
print("clips", len(c), "| new labels: cheat", int((c.label_new == 1).sum()), "normal", int((c.label_new == 0).sum()), "unsure", int((c.label_new == -1).sum()), "| old cheat", int((c.label == 1).sum()))
print("old vs new on clips with a new label:", pd.crosstab(c.label[c.label_new >= 0], c.label_new[c.label_new >= 0]).to_string())
print("unsure by view/type:", c[c.label_new == -1].assign(t=lambda d: d.video.str.contains('/side_').map({True: 'side', False: 'front'})).t.value_counts().to_dict())
