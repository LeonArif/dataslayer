# Human per-second labels: builds the thumbnails and the viewer page for annotating the cheat front videos of the TRAIN set by hand.
# Rules the viewer follows (FAQ #44.3, #50.3, #63.1, #63.2, #72, #82): a 1-second clip (30 frames) is cheat only when the cheating behaviour dominates it, otherwise normal;
# start and end movements count when they already point toward cheating; under desk counts from the moment the head bends down; operating the smartwatch is cheat,
# only looking at its screen is normal; reading the notes is cheat when it dominates, a rare glance while writing is normal; repeated peeking = repeated glances toward one direction
# (another participant's paper), looking around = many directions. Side videos are NOT annotated: FAQ #50.2 lets the front annotation be shifted onto the side video.
# The viewer starts BLANK (no suggested labels: AI-suggested labels are not allowed, FAQ #45). The worklist order only says which videos had the most held-out errors (FAQ #51).
# Output: leon/annotation/thumbs/<video with __>/<second>_<k>.jpg (4 frames per second), leon/annotation/data.js, leon/annotation/viewer.html (open it in Chrome; it saves to the browser and exports a CSV).
# usage (from leon/): python scripts/annotation_prepare.py
import json
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; ANN = ROOT / "leon/annotation"; THUMBS = ANN / "thumbs"; THUMBS.mkdir(parents=True, exist_ok=True)
work = pd.read_csv(ANN / "worklist.csv")
OFFS = [0, 8, 15, 23]                                    # frames inside each second shown as 4 thumbnails
cv2.setNumThreads(2)
videos = []
for r in work.itertuples():
    d = THUMBS / r.video.replace("/", "__"); d.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / "data/train" / (r.video + ".MOV"))); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); secs = n // 30
    need = {s * 30 + o: (s, k) for s in range(secs) for k, o in enumerate(OFFS)}
    i = 0
    while cap.grab():
        if i in need:
            ok, img = cap.retrieve()
            if ok:
                s, k = need[i]; cv2.imwrite(str(d / f"{s}_{k}.jpg"), cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 75])
        i += 1
    cap.release()
    videos.append(dict(order=int(r.order), video=r.video, action=r.action, subject=int(r.subject), seconds=secs, dir="thumbs/" + r.video.replace("/", "__"),
                       errors=None if pd.isna(r.held_out_errors) or r.held_out_errors == "" else int(r.held_out_errors)))
    print(f"{r.order}/{len(work)} {r.video} {secs} s", flush=True)
(ANN / "data.js").write_text("const VIDEOS = " + json.dumps(videos) + ";\n", encoding="utf-8")
print("done", len(videos), "videos")
