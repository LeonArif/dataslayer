"""Copy of fayadh/scripts/extract_pose.py writing to cache/pose_vitpose (stride 30 = one window per test-like 1-second clip). Ekstraksi pose (RT-DETR person box + ViTPose 17 keypoint COCO) per window 1 detik. Hasil di cache/main/.

Train: video dipotong window 30 frame (stride WIN_STRIDE). Test: 1 window per sequence.
Per window: box orang dideteksi di frame tengah, lalu pose dihitung pada frame 0, 3, ..., 27 (10 frame).
"""
import os, glob, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "cache", "pose_vitpose")
DET_NAME = "PekingU/rtdetr_r50vd_coco_o365"
POSE_NAME = "usyd-community/vitpose-base-simple"
TEST_W, TEST_H = 640, 360
JPEG_QUALITY = 90
CLIP_LEN = 30
POSE_STRIDE = 3
WIN_STRIDE = int(os.environ.get("WIN_STRIDE", 30))
POSE_BATCH = 40

dp = AutoProcessor.from_pretrained(DET_NAME)
dm = RTDetrForObjectDetection.from_pretrained(DET_NAME).cuda().eval()
pp = AutoProcessor.from_pretrained(POSE_NAME)
pm = VitPoseForPoseEstimation.from_pretrained(POSE_NAME).cuda().eval()


def to_rgb(frame_bgr, from_video):
    if from_video:
        frame_bgr = cv2.resize(frame_bgr, (TEST_W, TEST_H), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def needed(i):
    # frame yang dipakai: frame pose (tiap POSE_STRIDE) dan frame tengah window untuk deteksi
    return i % POSE_STRIDE == 0 or i % CLIP_LEN == CLIP_LEN // 2


def read_video(path):
    """Dict {index: frame RGB} hanya untuk frame yang dipakai, plus jumlah frame."""
    cap, out, i = cv2.VideoCapture(path), {}, 0
    while cap.grab():
        if needed(i):
            _, f = cap.retrieve()
            out[i] = to_rgb(f, True)
        i += 1
    cap.release()
    return out, i


def read_seq(folder):
    files = sorted(glob.glob(f"{folder}/*.jpg"))
    return {i: to_rgb(cv2.imread(p), False) for i, p in enumerate(files) if needed(i)}


def prefetch(fn, items, n_workers=4, depth=8):
    """Seperti map paralel, tetapi maksimal `depth` hasil yang menunggu di memori."""
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    items = list(items)
    with ThreadPoolExecutor(n_workers) as ex:
        futs = deque(ex.submit(fn, x) for x in items[:depth])
        nxt = depth
        while futs:
            r = futs.popleft().result()
            if nxt < len(items):
                futs.append(ex.submit(fn, items[nxt])); nxt += 1
            yield r


@torch.no_grad()
def detect(imgs):
    """Box orang terbesar x skor per gambar, format xywh. Fallback: seluruh frame."""
    boxes = []
    for i in range(0, len(imgs), 16):
        batch = imgs[i : i + 16]
        inp = dp(images=batch, return_tensors="pt").to("cuda")
        res = dp.post_process_object_detection(dm(**inp), target_sizes=[(TEST_H, TEST_W)] * len(batch), threshold=0.3)
        for r in res:
            m = r["labels"] == 0
            if m.sum() == 0:
                boxes.append([0, 0, TEST_W, TEST_H]); continue
            b, s = r["boxes"][m].cpu().numpy(), r["scores"][m].cpu().numpy()
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            k = int(np.argmax(area * s))
            x0, y0, x1, y1 = b[k]
            boxes.append([x0, y0, x1 - x0, y1 - y0])
    return np.array(boxes, np.float32)


@torch.no_grad()
def pose(imgs, boxes):
    out = []
    for i in range(0, len(imgs), POSE_BATCH):
        bi, bb = imgs[i : i + POSE_BATCH], boxes[i : i + POSE_BATCH]
        inp = pp(bi, boxes=[[b] for b in bb], return_tensors="pt").to("cuda")
        res = pp.post_process_pose_estimation(pm(**inp), boxes=[[b] for b in bb])
        for r in res:
            out.append(np.concatenate([r[0]["keypoints"].cpu().numpy(), r[0]["scores"].cpu().numpy()[:, None]], 1))
    return np.stack(out).astype(np.float32)                 # (N, 17, 3)


def windows_features(frames, starts):
    mids = [frames[s + CLIP_LEN // 2] for s in starts]
    boxes = detect(mids)
    pose_imgs, pose_boxes = [], []
    for s, b in zip(starts, boxes):
        for k in range(0, CLIP_LEN, POSE_STRIDE):
            pose_imgs.append(frames[s + k]); pose_boxes.append(b)
    kp = pose(pose_imgs, np.array(pose_boxes)).reshape(len(starts), CLIP_LEN // POSE_STRIDE, 17, 3)
    return kp, boxes


def main(which):
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    if which == "train":
        paths = sorted(p for p in glob.glob(f"{DATA_DIR}/train/*/*/*") if p.lower().endswith(".mov"))
        gen = prefetch(read_video, paths, n_workers=4, depth=6)
        KP, BOX, meta = [], [], []
        for k, (p, (frames, n_frames)) in enumerate(zip(paths, gen)):
            starts = list(range(0, n_frames - CLIP_LEN + 1, WIN_STRIDE))
            kp, boxes = windows_features(frames, starts)
            KP.append(kp); BOX.append(boxes)
            parts = os.path.normpath(p).split(os.sep)
            angle, behavior = os.path.splitext(parts[-1])[0].split("_", 1)
            for s in starts:
                meta.append(dict(video=f"{parts[-2]}/{parts[-1]}", label=int(parts[-3] == "cheat"),
                                 subject=int(parts[-2].split("_")[1]), angle=angle, behavior=behavior, start=s))
            if k % 20 == 0:
                print(f"{k}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
        np.save(f"{CACHE_DIR}/pose_train_s{WIN_STRIDE}.npy", np.concatenate(KP))
        np.save(f"{CACHE_DIR}/box_train_s{WIN_STRIDE}.npy", np.concatenate(BOX))
        pd.DataFrame(meta).to_csv(f"{CACHE_DIR}/pose_train_s{WIN_STRIDE}_meta.csv", index=False)
    else:
        ids = sorted(os.listdir(f"{DATA_DIR}/test"))
        gen = prefetch(read_seq, [f"{DATA_DIR}/test/{i}" for i in ids], n_workers=6, depth=48)
        KP, BOX, buf = [], [], []
        def flush():
            merged = {j * CLIP_LEN + i: f for j, fr in enumerate(buf) for i, f in fr.items()}
            kp, boxes = windows_features(merged, [j * CLIP_LEN for j in range(len(buf))])
            KP.append(kp); BOX.append(boxes); buf.clear()
        for k, frames in enumerate(gen):
            buf.append(frames)
            if len(buf) == 16:
                flush()
            if k % 200 == 0:
                print(f"{k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
        if buf:
            flush()
        np.save(f"{CACHE_DIR}/pose_test.npy", np.concatenate(KP))
        np.save(f"{CACHE_DIR}/box_test.npy", np.concatenate(BOX))
        pd.DataFrame({"id": ids}).to_csv(f"{CACHE_DIR}/pose_test_meta.csv", index=False)
    print(f"done {which} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["test", "train"]:
        main(w)
