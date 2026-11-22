# Member features: pretrained VideoMAE v2 giant (OpenGVLab/VideoMAEv2-giant, self-supervised, pixel-reconstruction pretraining, frozen) on every 1-second clip, using ALL 30 frames.
# The model code from the Hugging Face snapshot (read before use) is executed without trust_remote_code: only the VisionTransformer part, so the easydict package is not needed.
# The 30 frames go in as ONE clip (tubelet size 2 -> 15 time steps; the sine-cosine position table is built for 30 frames, the model was pretrained with 16), 224x224 squashed, ImageNet-normalised.
# Train frames go through the test delivery format first (640x360 INTER_AREA + JPEG q90), test frames are read as delivered.
# Output per clip: (16, 1408) fp16 = 15 per-time-step mean-patch vectors (after the final LayerNorm) + the mean of those.
# usage (from leon/): python scripts/extract_videomaev2.py   (resumable; env LIMIT=n clips for a quick check)
import os, sys, time, json
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
from safetensors.torch import load_file
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data"; OUT = ROOT / "cache/videomaev2_g_224_30"
(OUT / "train").mkdir(parents=True, exist_ok=True); (OUT / "test_parts").mkdir(exist_ok=True)
cv2.setNumThreads(2)
SNAP = Path.home() / ".cache/huggingface/hub/models--OpenGVLab--VideoMAEv2-giant/snapshots/d27568eb41ccb2d41bb191fc2e3fe5aad74942d4"
src = (SNAP / "modeling_videomaev2.py").read_text(encoding="utf-8")
src = src[:src.index("class VideoMAEv2(PreTrainedModel):")]
src = src.replace("from transformers import AutoConfig, PreTrainedModel\n", "").replace("from .modeling_config import VideoMAEv2Config\n", "")
ns = {"__name__": "videomaev2_local"}; exec(compile(src, "modeling_videomaev2.py", "exec"), ns)


def attn_forward(self, x):                           # same maths as the original Attention.forward, with fused attention
    B, N, C = x.shape
    bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias)) if self.q_bias is not None else None
    qkv = F.linear(x, self.qkv.weight, bias).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
    return self.proj(o.transpose(1, 2).reshape(B, N, -1))


ns["Attention"].forward = attn_forward
cfg = json.loads((SNAP / "config.json").read_text(encoding="utf-8"))["model_config"]; cfg["num_frames"] = 30
model = ns["VisionTransformer"](**cfg)
sd = {k[len("model."):]: v for k, v in load_file(str(SNAP / "model.safetensors")).items() if k.startswith("model.")}
print("load", model.load_state_dict(sd, strict=False), flush=True)          # decoder / mask token / encoder_to_decoder are unexpected, nothing should be missing
del sd
model = model.half().cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1) * 255


def prep(img_bgr):
    return cv2.cvtColor(cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)


@torch.inference_mode()
def embed(clips):                                     # clips: (n, 30, 224, 224, 3) uint8 -> (n, 16, 1408) fp16
    out = []
    for s in range(len(clips)):
        x = torch.from_numpy(clips[s:s + 1]).cuda().permute(0, 4, 1, 2, 3).float()      # (1, 3, 30, 224, 224)
        x = ((x - MEAN) / STD).half()
        with torch.autocast("cuda", dtype=torch.float16):
            t = model.patch_embed(x)                                                      # (1, 15*256, 1408), time-major
            t = t + model.pos_embed.to(t.device, t.dtype)
            for b in model.blocks:
                t = b(t)
            t = model.norm(t)
        pm = t.view(1, 15, 256, -1).float().mean(2)                                      # (1, 15, 1408)
        o = torch.cat([pm, pm.mean(1, keepdim=True)], 1)                                 # (1, 16, 1408)
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
    return np.stack(fr[:n * 30]).reshape(n, 30, 224, 224, 3)


def read_seq(seq):
    files = sorted((DATA / "test" / seq).glob("*.jpg")); assert len(files) == 30
    return np.stack([prep(cv2.imread(str(p))) for p in files])


def run(items, loader, saver, exists, label):
    t0, done, todo = time.time(), 0, [it for it in items if not exists(it)]
    with ThreadPoolExecutor(2) as pool:
        futs = [pool.submit(loader, it) for it in todo[:3]]
        for k, it in enumerate(todo):
            data = futs.pop(0).result()
            if k + 3 < len(todo): futs.append(pool.submit(loader, todo[k + 3]))
            data = data if data.ndim == 5 else data[None]
            saver(it, embed(data)); done += len(data)
            if k % 10 == 0: print(f"{label} {k + 1}/{len(todo)} | {done / (time.time() - t0):.2f} clips/s | {time.time() - t0:.0f}s", flush=True)


if os.environ.get("LIMIT"):                           # quick check: first clips of the first test parts, nothing saved
    seqs = list(pd.read_csv(DATA / "sample_submission.csv")["id"])[:int(os.environ["LIMIT"])]
    t0 = time.time(); o = embed(np.stack([read_seq(s) for s in seqs])); print("check", o.shape, o.dtype, float(np.abs(o.astype(np.float32)).mean()), f"{len(seqs) / (time.time() - t0):.2f} clips/s")
    print(np.round(o[0, :3, :6].astype(np.float32), 3)); sys.exit()
sub = pd.read_csv(DATA / "sample_submission.csv"); ids = list(sub["id"]); parts = [ids[i:i + 25] for i in range(0, len(ids), 25)]
pf = lambda p: OUT / "test_parts" / f"part_{ids.index(p[0]):05d}.npy"
run(parts, lambda p: np.stack([read_seq(s) for s in p]), lambda p, f: np.save(pf(p), f), lambda p: pf(p).exists(), "test")
if not (OUT / "test.npy").exists():
    np.save(OUT / "test.npy", np.concatenate([np.load(pf(p)) for p in parts])); print("test.npy saved", flush=True)
rows = [(f"{p.parent.parent.name}/{p.parent.name}/{p.stem}", str(p)) for p in sorted(DATA.glob("train/*/*/*.MOV"))]
out = lambda r: OUT / "train" / (r[0].replace("/", "__") + ".npy")
run(rows, lambda r: read_video(r[1]), lambda r, f: np.save(out(r), f), lambda r: out(r).exists(), "train")
print("done", flush=True)
