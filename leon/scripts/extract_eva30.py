# Per-frame features (CLS + mean patch) of a frozen image backbone from timm for EVERY frame of every train video and all 30 frames of every test sequence, same layout as extract_dino30.py:
# cache/<OUT_DIR>/{test.npy (1515, 30, D), train/<video with __>.npy (n_frames, D)}. Resumable. Train frames go through the test delivery format first (640x360 INTER_AREA + JPEG q90), test jpgs are read as delivered.
# Default: EVA-02 Large (timm eva02_large_patch14_448.mim_m38m_ft_in22k_in1k, MIT, 305M), input 224x392 (16 x 28 patches of 14, dynamic image size), D = 2048 (1024 CLS + 1024 mean patch), bf16 autocast, fp16 on disk.
# env: MODEL (timm name), OUT_DIR, IMG_H, IMG_W (multiples of the patch size), LIMIT (n train videos and 2 test parts, speed test)
# usage (from leon/): python scripts/extract_eva30.py [test|train|both]
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, timm, torch

DATA = Path("../data"); MODEL = os.environ.get("MODEL", "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"); IMG_HW = (int(os.environ.get("IMG_H", 224)), int(os.environ.get("IMG_W", 392)))
CACHE = Path("..") / os.environ.get("OUT_DIR", "cache/eva02_l_ft_224x392"); LIMIT = int(os.environ.get("LIMIT", 0)); DEVICE = "cuda"
(CACHE / "train").mkdir(parents=True, exist_ok=True); (CACHE / "test_parts").mkdir(exist_ok=True)
WHAT = sys.argv[1] if len(sys.argv) > 1 else "both"
cv2.setNumThreads(2)
backbone = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=IMG_HW, dynamic_img_size=True).to(DEVICE).eval()
MEAN = torch.tensor(backbone.pretrained_cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(backbone.pretrained_cfg["std"], device=DEVICE).view(1, 3, 1, 1)
print("model", MODEL, "input", IMG_HW, "mean", backbone.pretrained_cfg["mean"], flush=True)


def to_input(img_bgr):
    return cv2.cvtColor(cv2.resize(img_bgr, (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def recode(img_bgr):                                      # train frames in the test delivery format
    img = cv2.resize(img_bgr, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)


@torch.inference_mode()
def embed(frames, batch_size=32):
    out = []
    for i in range(0, len(frames), batch_size):
        x = torch.from_numpy(np.stack(frames[i:i + batch_size])).to(DEVICE).permute(0, 3, 1, 2).float().div(255)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            tokens = backbone.forward_features((x - MEAN) / STD)
        feat = torch.cat([tokens[:, 0], tokens[:, backbone.num_prefix_tokens:].mean(1)], dim=1).float()
        assert torch.isfinite(feat).all(), "features contain NaN/inf"
        out.append(feat.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def read_video(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, img = cap.read()
        if not ok: break
        frames.append(to_input(recode(img)))
    cap.release()
    return frames


def read_seq(seq):
    files = sorted((DATA / "test" / seq).glob("*.jpg")); assert len(files) == 30, (seq, len(files))
    return [to_input(cv2.imread(str(p))) for p in files]


def run(items, loader, saver, exists, label):
    t0, done = time.time(), 0
    todo = [it for it in items if not exists(it)]
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(loader, todo[0]) if todo else None
        for k, it in enumerate(todo):
            frames = nxt.result(); nxt = pool.submit(loader, todo[k + 1]) if k + 1 < len(todo) else None
            saver(it, embed(frames)); done += len(frames)
            if k % 10 == 0: print(f"{label} {k + 1}/{len(todo)} | {done / (time.time() - t0):.1f} frames/s | {time.time() - t0:.0f}s", flush=True)


if WHAT in ("test", "both"):
    ids = list(pd.read_csv(DATA / "sample_submission.csv")["id"]); parts = [ids[i:i + 50] for i in range(0, len(ids), 50)][:(2 if LIMIT else None)]
    pf = lambda p: CACHE / "test_parts" / f"part_{ids.index(p[0]):05d}.npy"
    run(parts, lambda p: sum([read_seq(s) for s in p], []), lambda p, f: np.save(pf(p), f.reshape(len(p), 30, -1)), lambda p: pf(p).exists(), "test")
    if not LIMIT and not (CACHE / "test.npy").exists():
        np.save(CACHE / "test.npy", np.concatenate([np.load(pf(p)) for p in parts])); print("test.npy", np.load(CACHE / "test.npy", mmap_mode="r").shape, flush=True)

if WHAT in ("train", "both"):
    rows = [(f"{p.parent.parent.name}/{p.parent.name}/{p.stem}", str(p)) for p in sorted(DATA.glob("train/*/*/*.MOV"))]
    if LIMIT: rows = rows[:LIMIT]
    out = lambda r: CACHE / "train" / (r[0].replace("/", "__") + ".npy")
    run(rows, lambda r: read_video(r[1]), lambda r, f: np.save(out(r), f), lambda r: out(r).exists(), "train")
print("done", flush=True)
