# Member features: pretrained V-JEPA 2.1 (env SIZE=g: ViT-g 1B, vjepa2_1_vitg_384.pt; SIZE=l: ViT-L 300M, vjepa2_1_vitl_dist_vitG_384.pt, distilled from ViT-G; Meta checkpoint from dl.fbaipublicfiles.com, frozen, video model) on every 1-second clip, using ALL 30 frames.
# Model code: Meta's repo (cache/vjepa2_1_download/repo/vjepa2-main, read before use); the checkpoint is loaded with weights_only=True.
# The 30 frames go in as ONE clip (tubelet size 2 -> 15 time steps), 256x256 squashed (the model was trained at 256 then cooled down at 384, RoPE handles the size), ImageNet-normalised.
# Train frames go through the test delivery format first (640x360 INTER_AREA + JPEG q90), test frames are read as delivered.
# Output per clip: (16, 1408) fp16 = 15 per-time-step mean-patch vectors (2 frames each) + the mean of all patches.
# usage (from leon/): python scripts/extract_vjepa21.py   (resumable)
import os, sys, time, glob
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data"; SIZE = os.environ.get("SIZE", "g"); OUT = ROOT / f"cache/vjepa21_{SIZE}_256_30"
(OUT / "train").mkdir(parents=True, exist_ok=True); (OUT / "test_parts").mkdir(exist_ok=True)
cv2.setNumThreads(2)
REPO = ROOT / "cache/vjepa2_1_download/repo/vjepa2-main"; sys.path.insert(0, str(REPO))
from src.hub.backbones import _make_vjepa2_1_model
if SIZE == "g":
    CK, KEY, NAME, KW = "vjepa2_1_vitg_384.pt", "target_encoder", "vjepa2_1_vit_giant_384", dict(predictor_num_mask_tokens=8, n_output_distillation=4)
else:
    CK, KEY, NAME, KW = "vjepa2_1_vitl_dist_vitG_384.pt", "ema_encoder", "vjepa2_1_vit_large_384", dict(predictor_depth=12, predictor_num_mask_tokens=8, n_output_distillation=1, teacher_embed_dim=1664)
model, _ = _make_vjepa2_1_model(model_name=NAME, img_size=256, return_all_tokens=True, pretrained=False, **KW)
ck = torch.load(ROOT / "cache/vjepa2_1_download/weights" / CK, map_location="cpu", weights_only=True, mmap=True)
print("checkpoint keys", list(ck.keys()), flush=True)
sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ck[KEY].items()}
print("load", model.load_state_dict(sd, strict=False), flush=True)      # pos_embed may be present while RoPE is used
del ck, sd
model = model.half().cuda().eval()
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
        x = ((x - MEAN) / STD).half().permute(0, 2, 1, 3, 4)                              # (b, 30, 3, 256, 256), model wants (b, 3, T, H, W)
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(x.permute(0, 2, 1, 3, 4))                                              # (b, 3, 30, 256, 256) -> (b, 15*256, 1408)
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
