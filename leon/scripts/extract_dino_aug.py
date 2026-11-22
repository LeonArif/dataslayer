# Augmented copy of the frozen DINOv3 ViT-H+ train features (for training the frozen-feature members E3 / D / A on more varied clips).
# For every labelled train clip (cache/members_2026-11-20/clips.csv, 1 second = 30 frames) the SAME random augmentation is applied to all 30 frames
# (the random generator is re-seeded with one seed per clip before every frame, so the hidden patches, the weather and the flip are the same in the whole
# second, like a real occlusion or weather), then the frozen DINOv3 gives CLS + mean patch (2560) per frame, exactly as extract30.py does.
# Train frames go through the test delivery format first (640x360 INTER_AREA + JPEG q90), then 400x224. Augmentation tokens: env AUG (default hide+weather+hflip).
# Output: cache/dinov3_aug<COPY>_224x400/train/<video with __>.npz = seconds (n,), feat (n, 30, 2560) fp16. Resumable per video.
# usage (from leon/): COPY=1 python scripts/extract_dino_aug.py      (env LIMIT=n videos for a speed check, nothing saved)
import os, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd, timm, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft2_augs

ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data"
COPY, AUG, LIMIT = int(os.environ.get("COPY", 1)), os.environ.get("AUG", "hide+weather+hflip"), int(os.environ.get("LIMIT", 0))
OUT = ROOT / f"cache/dinov3_aug{COPY}_224x400/train"; OUT.mkdir(parents=True, exist_ok=True)
IMG_HW, DEVICE = (224, 400), "cuda"
OPS = ft2_augs.parse(AUG)
cv2.setNumThreads(2)
backbone = timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=IMG_HW).to(DEVICE).eval()
MEAN = torch.tensor(backbone.pretrained_cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(backbone.pretrained_cfg["std"], device=DEVICE).view(1, 3, 1, 1)
NPRE = backbone.num_prefix_tokens
print("augmentation", OPS, "| copy", COPY, flush=True)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
    return cv2.cvtColor(cv2.resize(img, (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def load(video, seconds):
    cap, frames = cv2.VideoCapture(str(DATA / "train" / (video + ".MOV"))), []
    while True:
        ok, img = cap.read()
        if not ok: break
        frames.append(img)
    cap.release()
    secs = [s for s in seconds if (s + 1) * 30 <= len(frames)]
    return secs, [np.stack([recode(frames[s * 30 + i]) for i in range(30)]) for s in secs]


@torch.inference_mode()
def clip_features(clip, seed):                          # clip (30, 224, 400, 3) uint8 -> (30, 2560) fp16
    x = torch.from_numpy(clip).to(DEVICE).permute(0, 3, 1, 2).float().div(255)
    out = []
    for f in range(30):
        torch.manual_seed(seed); np.random.seed(seed)    # same random draw for every frame of the second
        out.append(ft2_augs.augment(x[f:f + 1], None, OPS)[0])
    x = (torch.cat(out) - MEAN) / STD
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        t = backbone.forward_features(x)
    return torch.cat([t[:, 0], t[:, NPRE:].mean(1)], 1).float().cpu().numpy().astype(np.float16)


clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")[["video", "second"]]
groups = [(v, sorted(g["second"].astype(int))) for v, g in clips.groupby("video")]
todo = [(v, s) for v, s in groups if not (OUT / (v.replace("/", "__") + ".npz")).exists()]
if LIMIT: todo = todo[:LIMIT]
t0, done = time.time(), 0
with ThreadPoolExecutor(2) as pool:
    futs = [pool.submit(load, v, s) for v, s in todo[:2]]
    for k, (v, s) in enumerate(todo):
        secs, vids = futs.pop(0).result()
        if k + 2 < len(todo): futs.append(pool.submit(load, *todo[k + 2]))
        feats = np.stack([clip_features(c, zlib.crc32(f"{v}|{sec}|{COPY}".encode()) % (2 ** 31)) for sec, c in zip(secs, vids)])
        done += len(secs)
        if not LIMIT: np.savez(OUT / (v.replace("/", "__") + ".npz"), seconds=np.array(secs), feat=feats)
        print(f"video {k + 1}/{len(todo)} {v} clips {len(secs)}/{len(s)} | {done / (time.time() - t0):.2f} clips/s ({30 * done / (time.time() - t0):.1f} frames/s) | {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
