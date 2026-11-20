# EVERY frame (30 per second) of every train video, passed through the TEST delivery format (resize to 640x360, JPEG quality 90), then resized to 400x224,
# stored as uint8 RGB. Used by ft2_train.py with ALLFRAMES=1 so each epoch can draw a different random frame of every half second.
# Output: cache/ft_frames_testfmt/train_frames_all.npy (n, 224, 400, 3) and train_meta_all.csv (row, video, frame, sec). Resumable per video.
# usage (from leon/): python scripts/ft2_frames_all.py
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data", ROOT / "cache/ft_frames_testfmt"
OUT.mkdir(parents=True, exist_ok=True)
IMG_HW = (224, 400)
cv2.setNumThreads(2)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)


def to_input(img):
    return cv2.cvtColor(cv2.resize(img, (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def load(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(to_input(recode(img)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return np.stack(frames), fps


rows = []
for p in sorted(DATA.glob("train/*/*/*.MOV")) + sorted(DATA.glob("train/*/*/*.mov")):
    rows.append(dict(path=str(p), video=f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"))
train = pd.DataFrame(rows).drop_duplicates("video").reset_index(drop=True)
caps = []
for p in train["path"]:
    cap = cv2.VideoCapture(p); caps.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 2); cap.release()
train["offset"] = np.r_[0, np.cumsum(caps)[:-1]]; train["cap"] = caps
total = int(np.sum(caps))
npy, part = OUT / "train_frames_all.npy", OUT / "train_meta_all_part.csv"
mm = np.lib.format.open_memmap(npy, mode="r+" if npy.exists() else "w+", dtype=np.uint8, shape=(total, *IMG_HW, 3))
done = pd.read_csv(part) if part.exists() else pd.DataFrame(columns=["row", "video", "frame", "sec"])
todo = train[~train["video"].isin(set(done["video"]))].reset_index(drop=True)
print(f"videos {len(train)} | total rows {total} ({total * 224 * 400 * 3 / 1e9:.1f} GB) | already done {len(train) - len(todo)}", flush=True)
t0 = time.time()
with ThreadPoolExecutor(2) as pool:
    for k, (r, (frames, fps)) in enumerate(zip(todo.itertuples(), pool.map(load, todo["path"]))):
        n = min(len(frames), r.cap)
        mm[r.offset:r.offset + n] = frames[:n]
        pd.DataFrame(dict(row=r.offset + np.arange(n), video=r.video, frame=np.arange(n), sec=np.arange(n) / fps)).to_csv(part, mode="a", header=not part.exists(), index=False)
        if k % 10 == 0:
            print(f"video {k}/{len(todo)} | {time.time() - t0:.0f}s", flush=True)
mm.flush()
meta = pd.read_csv(part).sort_values("row").reset_index(drop=True)
meta.to_csv(OUT / "train_meta_all.csv", index=False)
old = pd.read_csv(OUT / "train_meta.csv")
print("done rows", len(meta), "| old every-15th rows all present:", len(old.merge(meta, on=["video", "frame"])) == len(old), flush=True)
