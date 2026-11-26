"""ViTPose+ large (usyd-community/vitpose-plus-large, COCO expert) keypoints on ALL 30 frames of every window, same layout as cache/pose_vitpose_all.
The person boxes are the cached ones of pose_vitpose_all (RT-DETR on the middle frame), so only the pose model changes. Train frames go through the test delivery format (640x360 INTER_AREA + JPEG q90), test jpgs are read as delivered.
Output cache/pose_vitpose_plus_large/: pose_train_s30.npy (6166, 30, 17, 3), pose_test.npy (1515, 30, 17, 3), copies of the box and meta files. Train resumable per video (train_parts/*.npy).
env: LIMIT = n train videos and 20 test windows (speed test, nothing final saved). usage (from leon/): python scripts/extract_pose_plus.py [train|test|both]"""
import os, sys, time, shutil
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

ROOT = Path(__file__).resolve().parents[2]; SRC = ROOT / "cache/pose_vitpose_all"; OUT = ROOT / "cache/pose_vitpose_plus_large"; LIMIT = int(os.environ.get("LIMIT", 0))
(OUT / "train_parts").mkdir(parents=True, exist_ok=True)
NAME = "usyd-community/vitpose-plus-large"
pp = AutoProcessor.from_pretrained(NAME); pm = VitPoseForPoseEstimation.from_pretrained(NAME).half().cuda().eval()
cv2.setNumThreads(2)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def pose(imgs, box):                                        # 30 RGB frames, one box (xywh) -> (30, 17, 3)
    out = []
    for i in range(0, len(imgs), 30):
        bi = imgs[i:i + 30]; bb = [[box]] * len(bi)
        inp = pp(bi, boxes=bb, return_tensors="pt").to("cuda"); inp["pixel_values"] = inp["pixel_values"].half()
        o = pm(**inp, dataset_index=torch.zeros(len(bi), dtype=torch.long, device="cuda")); o.heatmaps = o.heatmaps.float()
        res = pp.post_process_pose_estimation(o, boxes=bb)
        out += [np.concatenate([r[0]["keypoints"].float().cpu().numpy(), r[0]["scores"].float().cpu().numpy()[:, None]], 1) for r in res]
    return np.stack(out)[:, :17].astype(np.float32)


def read_video(path):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, im = cap.read()
        if not ok: break
        fr.append(recode(im))
    cap.release()
    return fr


what = sys.argv[1] if len(sys.argv) > 1 else "both"
if what in ("train", "both"):
    meta = pd.read_csv(SRC / "pose_train_s30_meta.csv"); box = np.load(SRC / "box_train_s30.npy")
    meta["path"] = [str(ROOT / "data/train" / ("cheat" if l == 1 else "normal") / v) for v, l in zip(meta.video, meta.label)]
    vids = list(dict.fromkeys(meta.path)); part = lambda p: OUT / "train_parts" / (p.replace("\\", "/").split("/data/train/")[1].replace("/", "__") + ".npy")
    if LIMIT: vids = vids[:LIMIT]
    todo = [p for p in vids if not part(p).exists()]; t0 = time.time(); nwin = 0
    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(read_video, todo[0]) if todo else None
        for k, p in enumerate(todo):
            fr = nxt.result(); nxt = ex.submit(read_video, todo[k + 1]) if k + 1 < len(todo) else None
            rows = np.where(meta.path.values == p)[0]
            kp = np.stack([pose(fr[int(meta.start[i]):int(meta.start[i]) + 30], box[i]) for i in rows]); np.save(part(p), kp); nwin += len(rows)
            if k % 10 == 0: print(f"train {k + 1}/{len(todo)} | {nwin * 30 / (time.time() - t0):.1f} frames/s | {time.time() - t0:.0f}s", flush=True)
    if not LIMIT:
        KP = np.concatenate([np.load(part(p)) for p in vids]); order = np.concatenate([np.where(meta.path.values == p)[0] for p in vids])
        assert (order == np.arange(len(meta))).all() and len(KP) == len(meta), "meta rows are not grouped by video in the same order"
        np.save(OUT / "pose_train_s30.npy", KP); shutil.copy(SRC / "pose_train_s30_meta.csv", OUT / "pose_train_s30_meta.csv"); shutil.copy(SRC / "box_train_s30.npy", OUT / "box_train_s30.npy"); print("train saved", KP.shape, flush=True)

if what in ("test", "both") and not LIMIT:
    tm = pd.read_csv(SRC / "pose_test_meta.csv"); tbox = np.load(SRC / "box_test.npy"); ids = list(tm["id"]); t0 = time.time()
    def read_seq(s):
        files = sorted((ROOT / "data/test" / s).glob("*.jpg")); assert len(files) == 30
        return [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
    KP = []
    with ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(read_seq, s) for s in ids[:4]]
        for k, s in enumerate(ids):
            fr = futs.pop(0).result()
            if k + 4 < len(ids): futs.append(ex.submit(read_seq, ids[k + 4]))
            KP.append(pose(fr, tbox[k]))
            if k % 100 == 0: print(f"test {k + 1}/{len(ids)} | {time.time() - t0:.0f}s", flush=True)
    np.save(OUT / "pose_test.npy", np.stack(KP)); shutil.copy(SRC / "pose_test_meta.csv", OUT / "pose_test_meta.csv"); shutil.copy(SRC / "box_test.npy", OUT / "box_test.npy"); print("test saved", flush=True)
print("done", flush=True)
