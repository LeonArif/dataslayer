# Augmented copy of the frozen LeVJEPA-1.5-Large features (for training the frozen LeVJEPA members D1 / E5 on more varied clips).
# For every labelled train clip (row of cache/members_2026-11-20/clips.csv, 30 frames from levjepa_clips.py) the SAME random augmentation is applied to all 30 frames
# (the random generator is re-seeded with one seed per clip before every frame), then the clip goes through the frozen model exactly like extract_levjepa.py:
# two interleaved 15-frame halves, 224x224, ImageNet normalised -> 30 per-frame mean-patch vectors in frame order + the [cls] token of each half = (32, 1024) fp16.
# Output: cache/levjepa_aug<COPY>_224_30/train.npy (6166, 32, 1024), row i = row i of clips.csv; resumable (chunks of 4 clips, progress in done.txt).
# env: COPY (default 1), AUG (default hide+weather: no flip, because the members also use pose, which is taken from the original clip), LIMIT=n clips (speed test, nothing saved)
# usage (from leon/): COPY=1 python scripts/extract_levjepa_aug.py
import os, sys, time, zlib
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModel
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft2_augs

ROOT = Path(__file__).resolve().parents[2]
COPY, AUG, LIMIT = int(os.environ.get("COPY", 1)), os.environ.get("AUG", "hide+weather"), int(os.environ.get("LIMIT", 0))
OPS = ft2_augs.parse(AUG)
OUT = ROOT / f"cache/levjepa_aug{COPY}_224_30"; OUT.mkdir(parents=True, exist_ok=True)
FR = np.load(ROOT / "cache/levjepa_clips/train_clips.npy", mmap_mode="r"); n = len(FR)
path = OUT / "train.npy"
mm = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=(n, 32, 1024))
done_f = OUT / "done.txt"; start = int(done_f.read_text()) if done_f.exists() else 0
model = AutoModel.from_pretrained("galilai-group/LeVJEPA-1.5-Large", trust_remote_code=True).half().cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
print("augmentation", OPS, "| copy", COPY, "| clips", n, "| start at", start, flush=True)


@torch.inference_mode()
def embed(arr, seeds):                                 # arr (b, 30, 224, 224, 3) uint8 -> (b, 32, 1024) fp16
    x = torch.from_numpy(arr).cuda().permute(0, 1, 4, 2, 3).float().div(255)         # (b, 30, 3, H, W)
    out = []
    for b in range(len(x)):
        fr = []
        for f in range(30):
            torch.manual_seed(seeds[b]); np.random.seed(seeds[b])                    # same draw on every frame of the clip
            fr.append(ft2_augs.augment(x[b, f:f + 1], None, OPS)[0])
        out.append(torch.cat(fr))
    x = ((torch.stack(out) - MEAN) / STD).permute(0, 2, 1, 3, 4).half()             # (b, 3, 30, H, W)
    halves = torch.cat([x[:, :, 0::2], x[:, :, 1::2]])                               # (2b, 3, 15, H, W)
    h = model(pixel_values=halves).last_hidden_state                                 # (2b, 1 + 15 * 196, 1024)
    b = x.shape[0]
    pm = h[:, 1:].view(2 * b, 15, 196, -1).float().mean(2); cls = h[:, 0].float()
    fr = torch.zeros(b, 30, pm.shape[-1], device="cuda"); fr[:, 0::2], fr[:, 1::2] = pm[:b], pm[b:]
    o = torch.cat([fr, cls[:b, None], cls[b:, None]], 1)
    assert torch.isfinite(o).all()
    return o.cpu().numpy().astype(np.float16)


t0, k0 = time.time(), start
end = min(n, start + LIMIT) if LIMIT else n
for s in range(start, end, 4):
    idx = list(range(s, min(end, s + 4)))
    res = embed(np.stack([FR[i] for i in idx]), [zlib.crc32(f"{i}|{COPY}".encode()) % 2 ** 31 for i in idx])
    if not LIMIT:
        mm[idx[0]:idx[-1] + 1] = res; mm.flush(); done_f.write_text(str(idx[-1] + 1))
    if (s - start) % 100 == 0:
        print(f"clip {idx[-1] + 1}/{n} | {(idx[-1] + 1 - k0) / (time.time() - t0):.2f} clips/s | {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
