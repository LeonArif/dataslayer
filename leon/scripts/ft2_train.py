# Fine-tune the top K of the 32 DINOv3 ViT-H+ blocks (+ final norm + an 8-action head) on train frames, 5 folds by video subject (the folds of main_legit).
# Input = uint8 frames in the test delivery format (ft2_frames.py). The first 32-K blocks are frozen and run on the fly, so images can be augmented
# and K can be anything up to about 24 on an 8 GB GPU: the trainable blocks use bitsandbytes 8-bit AdamW stepped inside the backward pass
# (no stored gradients, no gradient clipping), with gradient checkpointing.
# Frame labels: before the action starts the person writes -> Writing_Answers; after the start -> the video's action (by-eye starts).
# Per fold it saves: weights of the trained blocks + norm (fp16), the 2560-d features (CLS + mean patch) of ALL train frames from that fold's backbone,
# and the 8-action log-probabilities of the held-out frames. Test features come from ft2_infer_test.py.
# usage (from leon/): python scripts/ft2_train.py K [EPOCHS]     env: FT_OUT (run name), AUG (augmentation tokens or groups, see ft2_augs.py; 0 = none), FOLDS (e.g. 0 or 0,1; default all),
# VAL_ONLY=1 (screening: only predict the held-out frames, save no weights / features), TSEED (training seed, default 42), BENCH=n (speed test, n steps),
# ALLFRAMES=1 (train on ALL 30 frames per second: every epoch draws one random labelled frame from each half second out of cache/ft_frames_testfmt/train_frames_all.npy
# made by ft2_frames_all.py; same number of training images per epoch as before, the held-out / feature pass still uses the every-15th-frame rows)
import sys, time, math, os
from pathlib import Path
import numpy as np, pandas as pd, timm, torch, torch.nn as nn, torch.nn.functional as F
import bitsandbytes as bnb
import ft2_augs
from torch.utils.checkpoint import checkpoint
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
FR = ROOT / os.environ.get("FRAMES_DIR", "cache/ft_frames_testfmt")
OUT = ROOT / "cache/ft2" / os.environ.get("FT_OUT", "run"); OUT.mkdir(parents=True, exist_ok=True)
IMG_HW, DEVICE, SEED = (int(os.environ.get("IMG_H", 224)), int(os.environ.get("IMG_W", 400))), "cuda", 42     # env IMG_H / IMG_W with FRAMES_DIR = a frame cache made by ft2_frames.py at that size
K = int(sys.argv[1]); EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
BS, LR_BLOCKS, LR_HEAD, WD = 16, float(os.environ.get("LR_BLOCKS", 2e-5)), float(os.environ.get("LR_HEAD", 1e-3)), 0.05
AUG, BENCH = os.environ.get("AUG", "0"), int(os.environ.get("BENCH", 0))
OPS = ft2_augs.parse(AUG)
FOLDS = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
VAL_ONLY, TSEED = int(os.environ.get("VAL_ONLY", 0)), int(os.environ.get("TSEED", 42))
ALL = int(os.environ.get("ALLFRAMES", 0))
LLRD, EMA_FRAC = float(os.environ.get("LLRD", 1.0)), float(os.environ.get("EMA_FRAC", 0))   # layer-wise LR decay per block below the top one (1 = same LR everywhere); EMA window as a fraction of the steps (0 = off)
CHEAT_ACTIONS = ["Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
NORMAL_ACTIONS = ["Checking_Smartwatch", "Looking_Around", "Thinking", "Writing_Answers"]
ACTIONS = CHEAT_ACTIONS + NORMAL_ACTIONS
WRITE = ACTIONS.index("Writing_Answers")

# ---- train table and folds exactly as in main_legit
rows = []
for path in sorted((ROOT / "data").glob("train/*/*/*.MOV")):
    view, action = path.stem.split("_", 1)
    rows.append({"video": f"{path.parent.parent.name}/{path.parent.name}/{path.stem}", "label": int(path.parent.parent.name == "cheat"),
                 "subject": int(path.parent.name.split("_")[1]), "view": view, "action": action})
train = pd.DataFrame(rows)
train["action_id"] = train["action"].map(ACTIONS.index)
train["fold"] = -1
for fold, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, train["action_id"], train["subject"])):
    train.loc[va, "fold"] = fold
eye = pd.read_csv(ROOT / "leon/train_action_start_eye_all.csv").set_index("video")["start_eye"]
train["start"] = [eye.get(v.replace("/side_", "/front_"), np.nan) + (0.5 if vw == "side" else 0.0) for v, vw in zip(train.video, train.view)]

meta = pd.read_csv(FR / "train_meta.csv").merge(train[["video", "fold", "action_id", "start"]], on="video", how="left")
assert meta["fold"].notna().all() and (meta["row"].values == np.arange(len(meta))).all()


NEWLAB = int(os.environ.get("NEWLAB", 0))        # 1 = cheat videos labelled per second from leon/annotation/clip_labels_ai.csv (cheat = the video's action, normal = writing, unsure = not trained on); other videos as before
if NEWLAB:
    _nl = pd.read_csv(ROOT / "leon/annotation/clip_labels_ai.csv"); _lab = {(v, s): l for v, s, l in zip(_nl.video, _nl.second, _nl.label_new)}


def frame_labels(df):
    lab = np.full(len(df), -1)
    for i, (v, a, s, t, fr) in enumerate(zip(df.video, df.action_id, df.start, df.sec, df.frame)):
        if NEWLAB and v.startswith("cheat/") and (v, int(fr) // 30) in _lab:
            l = _lab[(v, int(fr) // 30)]; lab[i] = a if l == 1 else (WRITE if l == 0 else -1); continue
        if a == WRITE:
            lab[i] = WRITE
        elif not np.isnan(s):
            lab[i] = a if t >= s + 0.5 else (WRITE if t < s - 0.5 else -1)
        else:
            lab[i] = a if t >= 9.75 else (WRITE if t < 4.25 else -1)
    return lab


lab = frame_labels(meta)
meta["y"] = lab
if ALL:
    meta_all = pd.read_csv(FR / "train_meta_all.csv").merge(train[["video", "fold", "action_id", "start"]], on="video", how="left")
    assert meta_all["fold"].notna().all()
    meta_all["y"] = frame_labels(meta_all)
    rows_all, hs_all = meta_all["row"].values, pd.factorize(meta_all["video"])[0] * 100000 + meta_all["frame"].values // 15
    FRAMES_ALL = np.load(FR / "train_frames_all.npy", mmap_mode="r")
    print("ALL frames", len(meta_all), "labelled", (meta_all.y >= 0).sum(), flush=True)
FRAMES = np.load(FR / "train_frames.npy", mmap_mode="r")
print("frames", len(meta), "labelled", (lab >= 0).sum(), "per class", np.bincount(lab[lab >= 0], minlength=8), flush=True)

MODEL = "vit_huge_plus_patch16_dinov3.lvd1689m"
_probe = timm.create_model(MODEL, pretrained=False, num_classes=0, img_size=IMG_HW)
MEAN = torch.tensor(_probe.pretrained_cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(_probe.pretrained_cfg["std"], device=DEVICE).view(1, 3, 1, 1)
EMBED, NPRE = _probe.embed_dim, _probe.num_prefix_tokens
del _probe


def raw_images(idx, arr=None):
    arr = FRAMES if arr is None else arr
    return torch.from_numpy(np.stack([arr[i] for i in idx])).to(DEVICE).permute(0, 3, 1, 2).float().div(255)


def batch_images(idx):
    return (raw_images(idx) - MEAN) / STD


class Net(nn.Module):
    def __init__(self, base, k):
        super().__init__()
        self.base, self.nfrozen = base, 32 - k
        self.train_blocks = nn.ModuleList(base.blocks[self.nfrozen:])
        self.head = nn.Sequential(nn.BatchNorm1d(2 * EMBED, affine=False), nn.Dropout(0.2), nn.Linear(2 * EMBED, len(ACTIONS)))
        for p in base.parameters(): p.requires_grad_(False)
        for p in list(self.train_blocks.parameters()) + list(base.norm.parameters()): p.requires_grad_(True)
        for b in base.blocks[:self.nfrozen]: b.to(torch.bfloat16)

    def forward(self, x, feats=False):
        base = self.base
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            t = base.patch_embed(x); t, rope = base._pos_embed(t); t = base.norm_pre(t)
            for b in base.blocks[:self.nfrozen]:
                t = b(t, rope=rope)
        t = t.detach()
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            for b in self.train_blocks:
                t = checkpoint(b, t, rope=rope, use_reentrant=False) if self.training else b(t, rope=rope)
            t = base.norm(t)
            f = torch.cat([t[:, 0], t[:, NPRE:].mean(1)], 1)
        return (self.head(f.float()), f.float()) if feats else self.head(f.float())


@torch.no_grad()
def predict(m, idx_all):
    m.eval(); out, fe = [], []
    for s in range(0, len(idx_all), 32):
        lo, f = m(batch_images(idx_all[s:s + 32]), feats=True)
        out.append(F.log_softmax(lo, 1).float().cpu().numpy()); fe.append(f.cpu().numpy().astype(np.float16))
    return np.concatenate(out), np.concatenate(fe)


for fold in FOLDS:
    if VAL_ONLY and (OUT / f"k{K}_f{fold}_val.npy").exists():
        continue
    if not VAL_ONLY and (OUT / f"k{K}_f{fold}_weights.pt").exists() and (OUT / f"k{K}_f{fold}_feat_train.npy").exists():
        continue
    torch.manual_seed(TSEED + fold); np.random.seed(TSEED + fold)
    base = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=IMG_HW).to(DEVICE).eval()
    m = Net(base, K).to(DEVICE)
    blocks_params = list(m.train_blocks.parameters()) + list(base.norm.parameters())
    n_train = sum(p.numel() for p in blocks_params)
    print(f"K={K} fold={fold} trainable block params {n_train / 1e6:.1f}M | aug {OPS}", flush=True)
    assert n_train > K * 20e6
    opts = {p: bnb.optim.AdamW8bit([p], lr=LR_BLOCKS, weight_decay=WD) for p in blocks_params}
    def hook(p):
        opts[p].step(); opts[p].zero_grad(set_to_none=True)
    handles = [p.register_post_accumulate_grad_hook(hook) for p in blocks_params]
    head_opt = torch.optim.AdamW(m.head.parameters(), lr=LR_HEAD, weight_decay=WD)
    scale = {p: LLRD ** (K - 1 - i) for i, b in enumerate(m.train_blocks) for p in b.parameters()}; scale.update({p: 1.0 for p in base.norm.parameters()})
    ema_params = blocks_params + list(m.head.parameters()); ema = [p.detach().clone().float() for p in ema_params] if EMA_FRAC > 0 else None

    va = np.where((meta.fold == fold).values)[0]
    if ALL:
        ytr = meta_all.y.values; tr_all = np.where((meta_all.fold != fold).values & (ytr >= 0))[0]; y_all = torch.tensor(ytr, device=DEVICE)
        def sample_epoch():
            perm = np.random.permutation(tr_all); sel = perm[np.unique(hs_all[perm], return_index=True)[1]]
            wr, act = sel[ytr[sel] == WRITE], sel[ytr[sel] != WRITE]
            return np.random.permutation(np.r_[act, np.random.choice(wr, len(wr) // 3, replace=False)])
        ep0 = sample_epoch(); cnt = np.bincount(ytr[ep0], minlength=8); n_epoch = len(ep0)
        fetch = lambda b: raw_images(rows_all[b], FRAMES_ALL)
    else:
        tr = np.where((meta.fold != fold).values & (meta.y >= 0).values)[0]
        y_all = torch.tensor(meta.y.values, device=DEVICE)
        tr_write, tr_act = tr[meta.y.values[tr] == WRITE], tr[meta.y.values[tr] != WRITE]
        n_write = len(tr_write) // 3
        cnt = np.bincount(meta.y.values[tr_act], minlength=8); cnt[WRITE] = n_write; n_epoch = len(tr_act) + n_write
        fetch = raw_images
    w = torch.tensor((cnt.sum() / (8 * cnt)) ** 0.5, dtype=torch.float32, device=DEVICE)
    steps = EPOCHS * math.ceil(n_epoch / BS); warm = max(1, steps // 20)
    lr_at = lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))
    t0, step = time.time(), 0
    for ep in range(EPOCHS):
        m.train(); m.base.blocks[:m.nfrozen].eval()
        if ALL:
            perm = ep0 if ep == 0 else sample_epoch()
        else:
            perm = np.random.permutation(np.r_[tr_act, np.random.choice(tr_write, n_write, replace=False)])
        tot = 0.0
        for s in range(0, len(perm), BS):
            b = np.sort(perm[s:s + BS]); f = lr_at(step)
            for p_, o in opts.items(): o.param_groups[0]["lr"] = LR_BLOCKS * f * scale[p_]
            head_opt.param_groups[0]["lr"] = LR_HEAD * f
            yb = y_all[torch.from_numpy(b).to(DEVICE)]; ysoft = None
            if OPS:
                xb, ysoft = ft2_augs.augment(fetch(b), F.one_hot(yb, len(ACTIONS)).float(), OPS); xb = (xb - MEAN) / STD
            else:
                xb = (fetch(b) - MEAN) / STD
            loss = F.cross_entropy(m(xb), yb if ysoft is None else ysoft, weight=w, label_smoothing=0.1)
            head_opt.zero_grad(set_to_none=True); loss.backward(); head_opt.step()
            step += 1; tot += loss.item() * len(b)
            if ema is not None:
                d = min(1 - 1 / max(1.0, EMA_FRAC * steps), (1 + step) / (10 + step))
                with torch.no_grad(): torch._foreach_mul_(ema, d); torch._foreach_add_(ema, [p.detach().float() for p in ema_params], alpha=1 - d)
            if BENCH and step == BENCH:
                torch.cuda.synchronize(); print(f"BENCH K={K}: {BENCH * BS / (time.time() - t0):.1f} frames/s, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True); sys.exit()
            if step % 100 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.3f} | {time.time() - t0:.0f}s | peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
        print(f"K={K} fold={fold} epoch={ep} loss={tot / len(perm):.4f} | {time.time() - t0:.0f}s", flush=True)
    for h in handles: h.remove()
    if VAL_ONLY:
        pv, _ = predict(m, va)
        if ema is not None:      # also score the averaged weights (saved next to the run as <run>_ema)
            keep = [p.detach().clone() for p in ema_params]
            with torch.no_grad():
                for p_, e in zip(ema_params, ema): p_.copy_(e.to(p_.dtype))
            pe, _ = predict(m, va); ye = meta.y.values[va]; oke = ye >= 0
            print(f"K={K} fold={fold} EMA held-out frame accuracy {(pe.argmax(1)[oke] == ye[oke]).mean():.4f}", flush=True)
            (OUT.parent / (OUT.name + "_ema")).mkdir(exist_ok=True); np.save(OUT.parent / (OUT.name + "_ema") / f"k{K}_f{fold}_val.npy", np.c_[va, pe])
            with torch.no_grad():
                for p_, k_ in zip(ema_params, keep): p_.copy_(k_)
    else:
        lp_all, fe_all = predict(m, np.arange(len(meta))); pv = lp_all[va]
    yv = meta.y.values[va]; ok = yv >= 0
    print(f"K={K} fold={fold} held-out frame accuracy {(pv.argmax(1)[ok] == yv[ok]).mean():.4f} | {time.time() - t0:.0f}s", flush=True)
    if VAL_ONLY:
        np.save(OUT / f"k{K}_f{fold}_val.npy", np.c_[va, pv])
        del m, base, opts, head_opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        continue
    np.save(OUT / f"k{K}_f{fold}_feat_train.npy", fe_all); np.save(OUT / f"k{K}_f{fold}_val.npy", np.c_[va, pv])
    torch.save({"K": K, "blocks": {32 - K + i: {k: v.half().cpu() for k, v in b.state_dict().items()} for i, b in enumerate(m.train_blocks)},
                "norm": {k: v.half().cpu() for k, v in base.norm.state_dict().items()}}, OUT / f"k{K}_f{fold}_weights.pt")
    del m, base, opts, head_opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print("done")
