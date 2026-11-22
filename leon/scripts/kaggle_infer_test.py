# Test inference of the Kaggle 100%-train fine-tune (kaggle_15fps_aug/kaggle_finetune_15fps_aug_full.ipynb) on ALL 30 frames of every test sequence, for the averaged (EMA) weights and the
# unaveraged final weights in one pass (the frozen first 20 blocks run once per frame). Output per variant: head log-probs (1515, 30, 8) and the cheat probability per sequence.
# env: DIR (folder with the 4 downloaded files, default cache/kaggle_15fps_aug_full), TAG (default k12_15fps_aug_full), IMG_H / IMG_W (default 224 / 400)
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd, timm, torch

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / os.environ.get("DIR", "cache/kaggle_15fps_aug_full"); TAG = os.environ.get("TAG", "k12_15fps_aug_full")
K, IMG_HW, DEVICE, CHUNK = 12, (int(os.environ.get("IMG_H", 224)), int(os.environ.get("IMG_W", 400))), "cuda", 32
VARIANTS = {"ema": (f"{TAG}_weights.pt", f"{TAG}_head.pt"), "final": (f"{TAG}_final_weights.pt", f"{TAG}_final_head.pt")}
PARTS = DIR / "test_parts"; PARTS.mkdir(exist_ok=True)
cv2.setNumThreads(2)
base = timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=IMG_HW).to(DEVICE).eval()
MEAN = torch.tensor(base.pretrained_cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(base.pretrained_cfg["std"], device=DEVICE).view(1, 3, 1, 1)
NPRE, NF = base.num_prefix_tokens, 32 - K
for b in base.blocks[:NF]: b.to(torch.bfloat16)
frozen, top = base.blocks[:NF], base.blocks[NF:]
W, H = {}, {}
for v, (wf, hf) in VARIANTS.items():
    W[v] = torch.load(DIR / wf, weights_only=True)
    h = torch.load(DIR / hf, weights_only=True, map_location="cpu"); H[v] = [h[k].float().to(DEVICE) for k in ["0.running_mean", "0.running_var", "2.weight", "2.bias"]]
    assert len(W[v]["blocks"]) == K and W[v]["K"] == K
head = lambda v, x: torch.log_softmax(((x - H[v][0]) / torch.sqrt(H[v][1] + 1e-5)) @ H[v][2].T + H[v][3], 1)


def load_variant(v):
    for i, b in enumerate(top): b.load_state_dict({k: x.float() for k, x in W[v]["blocks"][NF + i].items()})
    base.norm.load_state_dict({k: x.float() for k, x in W[v]["norm"].items()})


def read_seq(seq):
    files = sorted((ROOT / "data/test" / seq).glob("*.jpg")); assert len(files) == 30
    return [cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for p in files]


@torch.inference_mode()
def middle(frames):
    out = []
    for s in range(0, len(frames), 32):
        x = torch.from_numpy(np.stack(frames[s:s + 32])).to(DEVICE).permute(0, 3, 1, 2).float().div(255)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            t = base.patch_embed((x - MEAN) / STD); t, rope = base._pos_embed(t); t = base.norm_pre(t)
            for b in frozen: t = b(t, rope=rope)
        out.append(t.to(torch.bfloat16))
    return torch.cat(out), rope


@torch.inference_mode()
def upper(mid, rope, v):
    lp = []
    for s in range(0, len(mid), 32):
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            t = mid[s:s + 32]
            for b in top: t = b(t, rope=rope)
            t = base.norm(t)
            lp.append(head(v, torch.cat([t[:, 0], t[:, NPRE:].mean(1)], 1).float()).cpu().numpy())
    return np.concatenate(lp)


ids = list(pd.read_csv(ROOT / "data/sample_submission.csv")["id"])
chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
pf = lambda c: PARTS / f"part_{ids.index(c[0]):05d}.npz"
todo = [c for c in chunks if not pf(c).exists()]
t0 = time.time()
with ThreadPoolExecutor(2) as pool:
    futs = [pool.submit(lambda c: sum([read_seq(s) for s in c], []), c) for c in todo[:2]]
    for k, c in enumerate(todo):
        frames = futs.pop(0).result()
        if k + 2 < len(todo): futs.append(pool.submit(lambda c: sum([read_seq(s) for s in c], []), todo[k + 2]))
        mid, rope = middle(frames); res = {}
        for v in VARIANTS:
            load_variant(v); res[v] = upper(mid, rope, v).reshape(len(c), 30, 8)
        np.savez(pf(c), **res); del mid
        print(f"chunk {k + 1}/{len(todo)} | {time.time() - t0:.0f}s", flush=True)
for v in VARIANTS:
    lp = np.concatenate([np.load(pf(c))[v] for c in chunks]); np.save(DIR / f"{TAG}_test_logp_{v}.npy", lp)
    print(v, lp.shape, "mean cheat prob", float(np.exp(lp[:, :, :4]).sum(2).mean()))
print("done", flush=True)
# usage (from leon/): python scripts/kaggle_infer_test.py   (resumable per chunk of 32 sequences; about 40 min on an RTX 4070 laptop GPU)
