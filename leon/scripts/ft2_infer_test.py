# Test features of the fine-tuned backbones (ft2_train.py) on ALL 30 frames of every test sequence.
# The first 32-K blocks are frozen and identical for the 5 fold backbones, so they run once per frame; then the saved trained blocks of each fold run on the
# same intermediate tokens. Output: cache/ft2/<run>/k<K>_test_feat.npy, shape (5 folds, 1515 sequences in sample_submission order, 30 frames, 2560) fp16.
# Resumable per chunk of 32 sequences.
# usage (from leon/): python scripts/ft2_infer_test.py K RUN
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd, timm, torch

ROOT = Path(__file__).resolve().parents[2]
K, RUN = int(sys.argv[1]), sys.argv[2]
OUT = ROOT / "cache/ft2" / RUN
PARTS = OUT / f"k{K}_test_parts"; PARTS.mkdir(exist_ok=True)
IMG_HW, DEVICE, CHUNK = (224, 400), "cuda", 32
cv2.setNumThreads(2)
MODEL = "vit_huge_plus_patch16_dinov3.lvd1689m"
base = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=IMG_HW).to(DEVICE).eval()
MEAN = torch.tensor(base.pretrained_cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(base.pretrained_cfg["std"], device=DEVICE).view(1, 3, 1, 1)
NPRE, NF = base.num_prefix_tokens, 32 - K
for b in base.blocks[:NF]: b.to(torch.bfloat16)
frozen, top = base.blocks[:NF], base.blocks[NF:]
W = [torch.load(OUT / f"k{K}_f{f}_weights.pt", weights_only=True) for f in range(5)]


def load_fold(f):
    for i, b in enumerate(top):
        b.load_state_dict({k: v.float() for k, v in W[f]["blocks"][NF + i].items()})
    base.norm.load_state_dict({k: v.float() for k, v in W[f]["norm"].items()})


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
            for b in frozen:
                t = b(t, rope=rope)
        out.append(t.to(torch.bfloat16))
    return torch.cat(out), rope


@torch.inference_mode()
def upper(mid, rope):
    fe = []
    for s in range(0, len(mid), 32):
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            t = mid[s:s + 32]
            for b in top:
                t = b(t, rope=rope)
            t = base.norm(t)
            fe.append(torch.cat([t[:, 0], t[:, NPRE:].mean(1)], 1).float().cpu().numpy().astype(np.float16))
    return np.concatenate(fe)


ids = list(pd.read_csv(ROOT / "data/sample_submission.csv")["id"])
chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
pf = lambda c: PARTS / f"part_{ids.index(c[0]):05d}.npy"
todo = [c for c in chunks if not pf(c).exists()]
t0 = time.time()
with ThreadPoolExecutor(2) as pool:
    futs = [pool.submit(lambda c: sum([read_seq(s) for s in c], []), c) for c in todo[:2]]
    for k, c in enumerate(todo):
        frames = futs.pop(0).result()
        if k + 2 < len(todo): futs.append(pool.submit(lambda c: sum([read_seq(s) for s in c], []), todo[k + 2]))
        mid, rope = middle(frames)
        res = []
        for f in range(5):
            load_fold(f); res.append(upper(mid, rope).reshape(len(c), 30, -1))
        np.save(pf(c), np.stack(res)); del mid
        print(f"chunk {k + 1}/{len(todo)} | {time.time() - t0:.0f}s", flush=True)
np.save(OUT / f"k{K}_test_feat.npy", np.concatenate([np.load(pf(c)) for c in chunks], axis=1))
print("done", np.load(OUT / f"k{K}_test_feat.npy", mmap_mode="r").shape, flush=True)
