# Fine-tuning step 1: run the frozen bottom 20 of the 32 ViT-H+ blocks once and store their output tokens.
# Train: every 15th frame (2 per second). Test: frames 01 and 16 of every 1-second clip (also 2 per second).
# Tokens are stored as bfloat16 bits (uint16) because activations exceed the float16 range.
import sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, timm, torch

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data", ROOT / "cache/ft_hplus_block20"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT, IMG_HW, STEP, BS = 20, (224, 400), 15, 32
DEVICE = "cuda"
model = timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=IMG_HW).to(DEVICE).eval()
MEAN = torch.tensor(model.pretrained_cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(model.pretrained_cfg["std"], device=DEVICE).view(1, 3, 1, 1)
N_TOK = model.num_prefix_tokens + (IMG_HW[0] // 16) * (IMG_HW[1] // 16)


def to_input(img):
    return cv2.cvtColor(cv2.resize(img, (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def bottom(frames):
    out = []
    for i in range(0, len(frames), BS):
        x = torch.from_numpy(np.stack(frames[i:i + BS])).to(DEVICE).permute(0, 3, 1, 2).float().div(255)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            t = model.patch_embed((x - MEAN) / STD)
            t, rope = model._pos_embed(t)
            t = model.norm_pre(t)
            for blk in model.blocks[:SPLIT]:
                t = blk(t, rope=rope)
        t = t.to(torch.bfloat16)
        assert torch.isfinite(t).all()
        out.append(t.view(torch.int16).cpu().numpy().view(np.uint16))
    return np.concatenate(out)


rows = []
for p in sorted(DATA.glob("train/*/*/*.MOV")) + sorted(DATA.glob("train/*/*/*.mov")):
    rows.append(dict(path=str(p), video=f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"))
train = pd.DataFrame(rows).drop_duplicates("video").reset_index(drop=True)
counts, fps_list = [], []
for p in train["path"]:
    cap = cv2.VideoCapture(p)
    counts.append(len(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), STEP)) + 2)
    fps_list.append(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
cap_total = int(sum(counts))
mm = np.lib.format.open_memmap(OUT / "train_tokens.npy", mode="w+", dtype=np.uint16, shape=(cap_total, N_TOK, model.embed_dim))
meta, pos, t0 = [], 0, time.time()
for k, (p, v, fps) in enumerate(zip(train["path"], train["video"], fps_list)):
    cap = cv2.VideoCapture(p)
    frames, idx, i = [], [], 0
    while cap.grab():
        if i % STEP == 0:
            ok, img = cap.retrieve()
            if ok:
                frames.append(to_input(img)); idx.append(i)
        i += 1
    cap.release()
    tok = bottom(frames)
    mm[pos:pos + len(tok)] = tok
    meta += [dict(row=pos + j, video=v, frame=f, sec=f / fps) for j, f in enumerate(idx)]
    pos += len(tok)
    if k % 20 == 0:
        print(f"train {k}/{len(train)} frames {pos} | {time.time() - t0:.0f}s", flush=True)
mm.flush(); del mm
pd.DataFrame(meta).to_csv(OUT / "train_meta.csv", index=False)
print("train rows used", pos, "of", cap_total, flush=True)

sub = pd.read_csv(DATA / "sample_submission.csv")
mm = np.lib.format.open_memmap(OUT / "test_tokens.npy", mode="w+", dtype=np.uint16, shape=(len(sub) * 2, N_TOK, model.embed_dim))
for s in range(0, len(sub), 64):
    frames = [to_input(cv2.imread(str(DATA / "test" / q / f))) for q in sub["id"][s:s + 64] for f in ("01.jpg", "16.jpg")]
    mm[2 * s:2 * s + len(frames)] = bottom(frames)
    if s % 640 == 0:
        print(f"test {s}/{len(sub)} | {time.time() - t0:.0f}s", flush=True)
mm.flush()
print("done", round(time.time() - t0), "s")
