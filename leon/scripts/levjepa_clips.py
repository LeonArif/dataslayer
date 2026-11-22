# Frame cache for fine-tuning LeVJEPA: every labelled 1-second train clip (cache/members_2026-11-20/clips.csv, 6,166 clips) as its 30 frames,
# in the test delivery format (640x360 INTER_AREA + JPEG q90), then squashed to 224x224 (INTER_AREA) as uint8 RGB, exactly the input of extract_levjepa.py.
# Output: cache/levjepa_clips/train_clips.npy (6166, 30, 224, 224, 3) uint8 (27.8 GB), row i = row i of clips.csv. Resumable per video (done list in done.txt).
# usage (from leon/): python scripts/levjepa_clips.py
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data"
OUT = ROOT / "cache/levjepa_clips"; OUT.mkdir(parents=True, exist_ok=True)
cv2.setNumThreads(2)
clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")
path = OUT / "train_clips.npy"
mm = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.uint8, shape=(len(clips), 30, 224, 224, 3))
done_f = OUT / "done.txt"
done = set(done_f.read_text().split("\n")) if done_f.exists() else set()


def prep(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
    return cv2.cvtColor(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def load(item):
    video, rows = item
    cap, frames = cv2.VideoCapture(str(DATA / "train" / (video + ".MOV"))), []
    while True:
        ok, img = cap.read()
        if not ok: break
        frames.append(img)
    cap.release()
    out = {}
    for r, sec in rows:
        assert (sec + 1) * 30 <= len(frames), (video, sec, len(frames))
        out[r] = np.stack([prep(frames[sec * 30 + i]) for i in range(30)])
    return video, out


groups = [(v, list(zip(g.index, g["second"].astype(int)))) for v, g in clips.groupby("video")]
todo = [g for g in groups if g[0] not in done]
t0 = time.time()
with ThreadPoolExecutor(2) as pool:
    for k, (video, out) in enumerate(pool.map(load, todo)):
        for r, arr in out.items(): mm[r] = arr
        mm.flush()
        with open(done_f, "a") as f: f.write(video + "\n")
        if k % 10 == 0: print(f"video {k + 1}/{len(todo)} | {time.time() - t0:.0f}s", flush=True)
print("done", len(clips), "clips", flush=True)
