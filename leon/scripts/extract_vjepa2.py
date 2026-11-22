# Member F1 features: pretrained V-JEPA 2 ViT-g (facebook/vjepa2-vitg-fpc64-256, frozen, video model) on every 1-second clip, using ALL 30 frames.
# The 30 frames go in as ONE clip (tubelet size 2 -> 15 time steps), 256x256 squashed, ImageNet-normalised.
# Train frames go through the test delivery format first (640x360 INTER_AREA + JPEG q90), test frames are read as delivered.
# Output per clip: (16, 1408) fp16 = 15 per-time-step mean-patch vectors (2 frames each) + the mean of all patches.
# usage (from leon/): python extract_levjepa.py   (resumable)
import os, sys, time, glob
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, torch
from transformers import AutoModel
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data"; OUT = ROOT / "cache/vjepa2_g_256_30"
(OUT / "train").mkdir(parents=True, exist_ok=True); (OUT / "test_parts").mkdir(exist_ok=True)
cv2.setNumThreads(2)
model = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256").half().cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1) * 255


def prep(img_bgr):
    return cv2.cvtColor(cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)


@torch.inference_mode()
def embed(clips):                                     # clips: (n, 30, 256, 256, 3) uint8 -> (n, 16, 1408) fp16
    out = []
    for s in range(0, len(clips), 1):
        x = torch.from_numpy(clips[s:s + 1]).cuda().permute(0, 4, 1, 2, 3).float()      # (b, 3, 30, 256, 256)
        x = ((x - MEAN) / STD).half().permute(0, 2, 1, 3, 4)                              # (b, 30, 3, 256, 256)
        h = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state          # (b, 15*256, 1408)
        b = x.shape[0]
        pm = h.view(b, 15, 256, -1).float().mean(2)                                      # (b, 15, 1408)
        o = torch.cat([pm, pm.mean(1, keepdim=True)], 1)                                 # (b, 16, 1408)
        assert torch.isfinite(o).all()
        out.append(o.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def read_video(path):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, img = cap.read()
        if not ok: break
        fr.append(prep(recode(img)))
    cap.release()
    n = len(fr) // 30
    return np.stack(fr[:n * 30]).reshape(n, 30, 256, 256, 3)


def read_seq(seq):
    files = sorted((DATA / "test" / seq).glob("*.jpg")); assert len(files) == 30
    return np.stack([prep(cv2.imread(str(p))) for p in files])


def run(items, loader, saver, exists, label, group=1):
    t0, done, todo = time.time(), 0, [it for it in items if not exists(it)]
    with ThreadPoolExecutor(2) as pool:
        futs = [pool.submit(loader, it) for it in todo[:3]]
        for k, it in enumerate(todo):
            data = futs.pop(0).result()
            if k + 3 < len(todo): futs.append(pool.submit(loader, todo[k + 3]))
            data = data if data.ndim == 5 else data[None]
            saver(it, embed(data)); done += len(data)
            if k % 10 == 0: print(f"{label} {k + 1}/{len(todo)} | {done / (time.time() - t0):.2f} clips/s | {time.time() - t0:.0f}s", flush=True)


sub = pd.read_csv(DATA / "sample_submission.csv"); ids = list(sub["id"]); parts = [ids[i:i + 25] for i in range(0, len(ids), 25)]
pf = lambda p: OUT / "test_parts" / f"part_{ids.index(p[0]):05d}.npy"
run(parts, lambda p: np.stack([read_seq(s) for s in p]), lambda p, f: np.save(pf(p), f), lambda p: pf(p).exists(), "test")
if not (OUT / "test.npy").exists():
    np.save(OUT / "test.npy", np.concatenate([np.load(pf(p)) for p in parts])); print("test.npy saved", flush=True)
rows = [(f"{p.parent.parent.name}/{p.parent.name}/{p.stem}", str(p)) for p in sorted(DATA.glob("train/*/*/*.MOV"))]
out = lambda r: OUT / "train" / (r[0].replace("/", "__") + ".npy")
run(rows, lambda r: read_video(r[1]), lambda r, f: np.save(out(r), f), lambda r: out(r).exists(), "train")
print("done", flush=True)
