"""Ekstraksi embedding DINOv2 (frozen) untuk frame train dan test. Hasil disimpan di cache/dino/.

Dipakai oleh main.ipynb (kode yang sama disalin ke notebook); file ini untuk menjalankan ekstraksi panjang di background.
"""
import os, glob, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # folder fayadh/
DATA_DIR = os.path.join(ROOT, "..", "data")
CACHE_DIR = os.path.join(ROOT, "cache", "dino")
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "facebook/dinov2-small"
IN_W, IN_H = 448, 252           # kelipatan patch 14 -> grid 32 x 18
FRAME_STRIDE = 6                # 5 frame per detik
TEST_W, TEST_H = 640, 360
JPEG_QUALITY = 90
GRID_R, GRID_C = 3, 4
BATCH = 32
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TAG = MODEL_NAME.split("/")[-1]


def prep(frame_bgr, from_video):
    if from_video:
        frame_bgr = cv2.resize(frame_bgr, (TEST_W, TEST_H), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return rgb


def read_video_frames(path):
    cap, out, i = cv2.VideoCapture(path), [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % FRAME_STRIDE == 0:
            _, f = cap.retrieve()
            out.append(prep(f, True))
        i += 1
    cap.release()
    return np.stack(out), i


def read_test_frames(folder):
    files = sorted(glob.glob(f"{folder}/*.jpg"))[::FRAME_STRIDE]
    return np.stack([prep(cv2.imread(p), False) for p in files])


@torch.no_grad()
def embed(model, frames):
    feats = []
    for i in range(0, len(frames), BATCH):
        x = (frames[i : i + BATCH].astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).cuda()
        h = model(pixel_values=x).last_hidden_state            # (B, 1 + 18*32, D)
        cls, patch = h[:, 0], h[:, 1:]
        B, _, D = patch.shape
        grid = patch.reshape(B, IN_H // 14, IN_W // 14, D).permute(0, 3, 1, 2)
        grid = torch.nn.functional.adaptive_avg_pool2d(grid, (GRID_R, GRID_C)).flatten(2).transpose(1, 2)
        feats.append(torch.cat([cls[:, None], patch.mean(1, keepdim=True), grid], 1).half().cpu().numpy())
    return np.concatenate(feats)                                  # (N, 2 + 12, D)


def main(which):
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).cuda().eval()
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    if which == "train":
        paths = sorted(p for p in glob.glob(f"{DATA_DIR}/train/*/*/*") if p.lower().endswith(".mov"))
        gen = Parallel(n_jobs=6, prefer="threads", return_as="generator")(delayed(read_video_frames)(p) for p in paths)
        embs, meta = [], []
        for k, (p, (frames, n_frames)) in enumerate(zip(paths, gen)):
            e = embed(model, frames)
            embs.append(e)
            parts = os.path.normpath(p).split(os.sep)
            angle, behavior = os.path.splitext(parts[-1])[0].split("_", 1)
            meta.append(dict(video=f"{parts[-2]}/{parts[-1]}", category=parts[-3], label=int(parts[-3] == "cheat"),
                             subject=int(parts[-2].split("_")[1]), angle=angle, behavior=behavior,
                             n_frames=n_frames, n_emb=len(e)))
            if k % 20 == 0:
                print(f"{k}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
        np.save(f"{CACHE_DIR}/dino_{TAG}_train.npy", np.concatenate(embs))
        pd.DataFrame(meta).to_csv(f"{CACHE_DIR}/dino_{TAG}_train_meta.csv", index=False)
    else:
        ids = sorted(os.listdir(f"{DATA_DIR}/test"))
        gen = Parallel(n_jobs=6, prefer="threads", return_as="generator", batch_size=8)(
            delayed(read_test_frames)(f"{DATA_DIR}/test/{i}") for i in ids)
        embs = []
        for k, frames in enumerate(gen):
            embs.append(embed(model, frames))
            if k % 200 == 0:
                print(f"{k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
        np.save(f"{CACHE_DIR}/dino_{TAG}_test.npy", np.stack(embs))
        pd.DataFrame({"id": ids}).to_csv(f"{CACHE_DIR}/dino_{TAG}_test_meta.csv", index=False)
    print(f"done {which} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for w in sys.argv[2:] or ["test", "train"]:
        main(w)
