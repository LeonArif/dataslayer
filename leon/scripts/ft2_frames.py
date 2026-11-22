# Fine-tuning data: every 15th frame (2 per second) of every train video, passed through the TEST delivery format
# (resize to 640x360, JPEG quality 90) and then resized to 400x224, stored as uint8 RGB. Same frame selection as ft_cache.py.
# Output: cache/ft_frames_testfmt/train_frames.npy (n, 224, 400, 3) and train_meta.csv (row, video, frame, sec).
# usage (from leon/): python scripts/ft2_frames.py
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import os
DATA, OUT = ROOT / "data", ROOT / os.environ.get("FRAMES_DIR", "cache/ft_frames_testfmt")      # env FRAMES_DIR and IMG_H / IMG_W for a different input size (default 224x400)
OUT.mkdir(parents=True, exist_ok=True)
STEP, IMG_HW = 15, (int(os.environ.get("IMG_H", 224)), int(os.environ.get("IMG_W", 400)))
cv2.setNumThreads(2)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)


def to_input(img):
    return cv2.cvtColor(cv2.resize(img, (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def load(path):
    cap, frames, idx, i = cv2.VideoCapture(path), [], [], 0
    while cap.grab():
        if i % STEP == 0:
            ok, img = cap.retrieve()
            if ok:
                frames.append(to_input(recode(img))); idx.append(i)
        i += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return np.stack(frames), idx, fps


rows = []
for p in sorted(DATA.glob("train/*/*/*.MOV")) + sorted(DATA.glob("train/*/*/*.mov")):
    rows.append(dict(path=str(p), video=f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"))
train = pd.DataFrame(rows).drop_duplicates("video").reset_index(drop=True)
cap_total = 0
for p in train["path"]:
    cap = cv2.VideoCapture(p)
    cap_total += len(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), STEP)) + 2
    cap.release()
mm = np.lib.format.open_memmap(OUT / "train_frames.npy", mode="w+", dtype=np.uint8, shape=(cap_total, *IMG_HW, 3))
meta, pos, t0 = [], 0, time.time()
with ThreadPoolExecutor(2) as pool:
    for k, (v, (frames, idx, fps)) in enumerate(zip(train["video"], pool.map(load, train["path"]))):
        mm[pos:pos + len(frames)] = frames
        meta += [dict(row=pos + j, video=v, frame=f, sec=f / fps) for j, f in enumerate(idx)]
        pos += len(frames)
        if k % 20 == 0:
            print(f"train {k}/{len(train)} frames {pos} | {time.time() - t0:.0f}s", flush=True)
mm.flush()
meta = pd.DataFrame(meta)
meta.to_csv(OUT / "train_meta.csv", index=False)
old = pd.read_csv(ROOT / "cache/ft_hplus_block20/train_meta.csv")
print("rows used", pos, "of", cap_total, "| same video/frame rows as the block-20 cache:", len(old) == len(meta) and (old[["video", "frame"]].values == meta[["video", "frame"]].values).all(), flush=True)
