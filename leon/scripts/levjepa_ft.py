# Fine-tune the top K of the 24 LeVJEPA-1.5-Large blocks (+ final norm + an 8-action head) on 1-second train clips, with augmentation.
# Input per sample = one half of a clip (15 frames: even or odd frames, 224x224, test delivery format, from levjepa_clips.py), random half at every step;
# the same random augmentation is applied to all 15 frames (the generator is re-seeded with one seed per clip before every frame).
# Head: BatchNorm + Dropout + Linear on [cls, mean patch] (2048) -> 8 actions like B; cheat probability = sum of the first 4 class probabilities.
# Clip score = mean of the two halves (all 30 frames used). Clip labels: cheat clips -> the video's action, normal clips -> Writing_Answers before the action
# (normal action videos: action when the clip centre is after 9.75 s, Writing before 4.25 s, in between = not trained on, as the frame labels of B).
# The first 24-K blocks are frozen and run on the fly in bf16, the top K use bitsandbytes 8-bit AdamW stepped inside the backward pass, gradient checkpointing.
# usage (from leon/): python scripts/levjepa_ft.py K [EPOCHS]
# env: FOLD (held-out fold 0-4, or ALL = train on all clips, default 0), AUG (default hide+weather+hflip, 0 = none), EMA_FRAC (default 0.2), LR_BLOCKS, LR_HEAD, BS (8),
#      FT_OUT (run name), TSEED, BENCH=n (speed / memory test with n steps, saves nothing), CHECK=1 (compare the forward with the Hugging Face model, then exit), NOTEST=1 (skip test inference)
# Output: cache/levjepa_ft/<run>/ weights_{ema,final}.pt, val_logp_{ema,final}.npy (n_val, 2 halves, 8) + val_rows.npy, test_logp_{ema,final}.npy (1515, 2, 8), summary.json
import os, sys, time, math, json, zlib
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import bitsandbytes as bnb
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft2_augs

ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data"
K = int(sys.argv[1]); EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
FOLD = os.environ.get("FOLD", "0"); BS = int(os.environ.get("BS", 8))
AUG = os.environ.get("AUG", "hide+weather+hflip"); OPS = ft2_augs.parse(AUG)
EMA_FRAC = float(os.environ.get("EMA_FRAC", 0.2)); LR_BLOCKS, LR_HEAD, WD = float(os.environ.get("LR_BLOCKS", 2e-5)), float(os.environ.get("LR_HEAD", 1e-3)), 0.05
TSEED, BENCH, CHECK, NOTEST = int(os.environ.get("TSEED", 42)), int(os.environ.get("BENCH", 0)), int(os.environ.get("CHECK", 0)), int(os.environ.get("NOTEST", 0))
HEAD = os.environ.get("HEAD", "lr")                       # lr = linear on [cls, mean patch]; gru / tf = BiGRU / transformer over the 15 per-frame mean-patch vectors
RUN = os.environ.get("FT_OUT", f"k{K}_f{FOLD}_{HEAD}_aug"); OUT = ROOT / "cache/levjepa_ft" / RUN; OUT.mkdir(parents=True, exist_ok=True)
DEVICE, DEPTH, T, HP = "cuda", 24, 15, 14
CHEAT = ["Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
ACTIONS = CHEAT + ["Checking_Smartwatch", "Looking_Around", "Thinking", "Writing_Answers"]; WRITE = 7
cv2.setNumThreads(2)

# ---- clips, labels
clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")
eye = pd.read_csv(ROOT / "leon/train_action_start_eye_all.csv").set_index("video")["start_eye"]
start = np.array([eye.get(v.replace("/side_", "/front_"), np.nan) + (0.5 if vw == "side" else 0.0) for v, vw in zip(clips.video, clips.view)])
centre, aid = clips["second"].values + 0.5, clips["action"].map(ACTIONS.index).values
y = np.full(len(clips), -1)
for i in range(len(clips)):
    if aid[i] == WRITE: y[i] = WRITE
    elif not np.isnan(start[i]): y[i] = aid[i] if centre[i] >= start[i] + 0.5 else (WRITE if centre[i] < start[i] - 0.5 else -1)
    else: y[i] = aid[i] if centre[i] >= 9.75 else (WRITE if centre[i] < 4.25 else -1)
assert ((y >= 0) & (y < 4) == (clips.label.values == 1) & (y >= 0)).all()          # a cheat class only on clips labelled cheat
ycheat = clips.label.values
va = np.where(clips.fold.values == int(FOLD))[0] if FOLD != "ALL" else np.array([], dtype=int)
tr = np.where((clips.fold.values != (int(FOLD) if FOLD != "ALL" else -9)) & (y >= 0))[0]
tr_write, tr_act = tr[y[tr] == WRITE], tr[y[tr] != WRITE]
n_write = len(tr_write) // 3
cnt = np.bincount(y[tr_act], minlength=8); cnt[WRITE] = n_write; n_epoch = len(tr_act) + n_write
print(f"run {RUN} | K={K} epochs={EPOCHS} BS={BS} fold={FOLD} aug={OPS} | train clips {len(tr)} (per epoch {n_epoch}, classes {cnt.tolist()}) | held-out clips {len(va)}", flush=True)

CLIPS_PATH = ROOT / "cache/levjepa_clips/train_clips.npy"
FRAMES = np.load(CLIPS_PATH, mmap_mode="r") if CLIPS_PATH.exists() else None
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 1, 3, 1, 1)


def to_input(arr, seeds=None):                      # arr (b, 15, 224, 224, 3) uint8 -> (b, 3, 15, 224, 224) normalised, augmented with one draw per clip
    x = torch.from_numpy(arr).to(DEVICE).permute(0, 1, 4, 2, 3).float().div(255)      # (b, 15, 3, H, W)
    if seeds is not None and OPS:
        out = []
        for b in range(len(x)):
            fr = []
            for f in range(x.shape[1]):
                torch.manual_seed(seeds[b]); np.random.seed(seeds[b])               # same random draw on every frame of the clip
                fr.append(ft2_augs.augment(x[b, f:f + 1], None, OPS)[0])
            out.append(torch.cat(fr))                                                # each augment() result is (1, 3, H, W) -> (15, 3, H, W)
        x = torch.stack(out)
    return ((x - MEAN) / STD).permute(0, 2, 1, 3, 4).contiguous()


class LRHead(nn.Module):
    def __init__(s):
        super().__init__(); s.net = nn.Sequential(nn.BatchNorm1d(2 * 1024, affine=False), nn.Dropout(0.2), nn.Linear(2 * 1024, len(ACTIONS)))

    def forward(s, cls, pf): return s.net(torch.cat([cls, pf.mean(1)], 1))


class SeqHead(nn.Module):                               # the E5 recipe inside the model: per-frame vectors, centred on the clip mean, small BiGRU / transformer
    def __init__(s, kind, d=256):
        super().__init__(); s.kind = kind
        s.inp = nn.Sequential(nn.LayerNorm(2 * 1024), nn.Dropout(0.2), nn.Linear(2 * 1024, d))
        if kind == "gru": s.seq = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        else: s.pos = nn.Parameter(torch.zeros(1, T, d)); s.seq = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.2, batch_first=True, norm_first=True), 2)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, len(ACTIONS)))

    def forward(s, cls, pf):
        m = pf.mean(1, keepdim=True); x = s.inp(torch.cat([pf - m, m.expand_as(pf)], 2))
        x = s.seq(x)[0] if s.kind == "gru" else s.seq(x + s.pos)
        return s.out(x.mean(1))


class Net(nn.Module):
    def __init__(self, enc, k):
        super().__init__()
        self.enc, self.nfrozen = enc, DEPTH - k
        self.train_blocks = nn.ModuleList(enc.blocks[self.nfrozen:])
        self.head = LRHead() if HEAD == "lr" else SeqHead(HEAD)
        for p in enc.parameters(): p.requires_grad_(False)
        for p in list(self.train_blocks.parameters()) + list(enc.norm.parameters()): p.requires_grad_(True)
        for b in enc.blocks[:self.nfrozen]: b.to(torch.bfloat16)
        N = T * HP * HP
        fid = torch.arange(N) // (HP * HP)
        mask = torch.zeros(N + 1, N + 1, dtype=torch.bool); mask[0, :] = True; mask[1:, 1:] = fid[:, None] >= fid[None, :]       # block-causal, cls reads everything (as the model's own mask)
        self.register_buffer("mask", mask[None, None].to(DEVICE), persistent=False)

    def forward(self, x, feats=False):
        enc, kw = self.enc, dict(T=T, H_patches=HP, W_patches=HP, token_ids=None, attn_mask=self.mask)
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            t = enc.patch_embed(x)
            t = torch.cat([enc.cls_token.expand(t.shape[0], -1, -1).to(t.dtype), t], 1)
            for b in enc.blocks[:self.nfrozen]: t = b(t, **kw)
        t = t.detach()
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            for b in self.train_blocks:
                t = checkpoint(b, t, use_reentrant=False, **kw) if self.training else b(t, **kw)
            t = enc.norm(t)
        cls, pf = t[:, 0].float(), t[:, 1:].float().reshape(t.shape[0], T, HP * HP, -1).mean(2)
        out = self.head(cls, pf)
        return (out, torch.cat([cls, pf.mean(1)], 1)) if feats else out


hf = AutoModel.from_pretrained("galilai-group/LeVJEPA-1.5-Large", trust_remote_code=True)
enc = hf.encoder.to(DEVICE).eval()
m = Net(enc, K).to(DEVICE)
blocks_params = list(m.train_blocks.parameters()) + list(enc.norm.parameters())
n_train = sum(p.numel() for p in blocks_params)
print(f"trainable block params {n_train / 1e6:.1f}M", flush=True)
assert n_train > K * 12e6

if CHECK:                                            # same features as the Hugging Face forward (fp16) on random-ish input
    x = torch.randn(1, 3, T, 224, 224, device=DEVICE)
    m.eval()
    with torch.no_grad():
        _, mine = m(x, feats=True)
        ref = AutoModel.from_pretrained("galilai-group/LeVJEPA-1.5-Large", trust_remote_code=True).half().to(DEVICE).eval()(pixel_values=x.half()).last_hidden_state.float()
        refv = torch.cat([ref[:, 0], ref[:, 1:].mean(1)], 1)
    print("check: cosine", F.cosine_similarity(mine, refv).item(), "| max abs diff", (mine - refv).abs().max().item(), flush=True); sys.exit()

torch.manual_seed(TSEED); np.random.seed(TSEED)
opts = {p: bnb.optim.AdamW8bit([p], lr=LR_BLOCKS, weight_decay=WD) for p in blocks_params}


def hook(p):
    opts[p].step(); opts[p].zero_grad(set_to_none=True)


handles = [p.register_post_accumulate_grad_hook(hook) for p in blocks_params]
head_opt = torch.optim.AdamW(m.head.parameters(), lr=LR_HEAD, weight_decay=WD)
ema_params = blocks_params + list(m.head.parameters()); ema = [p.detach().clone().float() for p in ema_params] if EMA_FRAC > 0 else None
w = torch.tensor((cnt.sum() / (8 * cnt)) ** 0.5, dtype=torch.float32, device=DEVICE)
y_t = torch.tensor(y, device=DEVICE)
steps = EPOCHS * math.ceil(n_epoch / BS); warm = max(1, steps // 20)
lr_at = lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))
rng = np.random.RandomState(TSEED)


def synth(b):                                         # random input when the clip cache is not built yet (BENCH only)
    return rng.randint(0, 255, (len(b), T, 224, 224, 3), dtype=np.uint8)


t0, step = time.time(), 0
for ep in range(EPOCHS):
    m.train(); m.enc.blocks[:m.nfrozen].eval()
    perm = rng.permutation(np.r_[tr_act, rng.choice(tr_write, n_write, replace=False)])
    tot = 0.0
    for s in range(0, len(perm), BS):
        b = np.sort(perm[s:s + BS]); f = lr_at(step)
        for o in opts.values(): o.param_groups[0]["lr"] = LR_BLOCKS * f
        head_opt.param_groups[0]["lr"] = LR_HEAD * f
        half = rng.randint(0, 2, len(b))
        arr = np.stack([FRAMES[i, h::2] for i, h in zip(b, half)]) if FRAMES is not None else synth(b)
        xb = to_input(arr, [zlib.crc32(f"{ep}|{i}|{TSEED}".encode()) % 2 ** 31 for i in b])
        yb = y_t[torch.from_numpy(b).to(DEVICE)]
        loss = F.cross_entropy(m(xb), yb, weight=w, label_smoothing=0.1)
        head_opt.zero_grad(set_to_none=True); loss.backward(); head_opt.step()
        step += 1; tot += loss.item() * len(b)
        if ema is not None:
            d = min(1 - 1 / max(1.0, EMA_FRAC * steps), (1 + step) / (10 + step))
            with torch.no_grad(): torch._foreach_mul_(ema, d); torch._foreach_add_(ema, [p.detach().float() for p in ema_params], alpha=1 - d)
        if BENCH and step == BENCH:
            torch.cuda.synchronize(); print(f"BENCH K={K} BS={BS}: {(time.time() - t0) / BENCH:.2f} s/step ({BENCH * BS / (time.time() - t0):.1f} clips/s), peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, epoch = {math.ceil(n_epoch / BS)} steps", flush=True); sys.exit()
        if step % 25 == 0:
            print(f"  step {step}/{steps} loss {loss.item():.3f} | {time.time() - t0:.0f}s | peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"epoch {ep} loss {tot / len(perm):.4f} | {time.time() - t0:.0f}s", flush=True)
for h in handles: h.remove()
train_s = time.time() - t0


@torch.no_grad()
def predict(arr_fn, n):                               # arr_fn(i) -> (30, 224, 224, 3) uint8 ; returns (n, 2, 8) log-probs of the two halves
    m.eval(); out = np.zeros((n, 2, 8), dtype=np.float32)
    for s in range(0, n, 4):
        ids = list(range(s, min(n, s + 4))); a = np.stack([arr_fn(i) for i in ids])
        for h in range(2):
            lo = m(to_input(a[:, h::2]))
            out[s:s + len(ids), h] = F.log_softmax(lo, 1).float().cpu().numpy()
    return out


def read_seq(seq):
    files = sorted((DATA / "test" / seq).glob("*.jpg")); assert len(files) == 30
    return np.stack([cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for p in files])


def score(lp): return np.exp(lp[:, :, :4]).sum(2).mean(1)


def best_f1(yy, p): return max(f1_score(yy, p > t, average="macro") for t in np.arange(0.05, 0.96, 0.025))


summary = {"run": RUN, "K": K, "epochs": EPOCHS, "fold": FOLD, "aug": AUG, "train_seconds": train_s, "steps": steps}
sub_ids = list(pd.read_csv(DATA / "sample_submission.csv")["id"]) if not NOTEST else []
keep = [p.detach().clone() for p in ema_params]
for tag in (["ema", "final"] if ema is not None else ["final"]):
    if tag == "ema":
        with torch.no_grad():
            for p_, e in zip(ema_params, ema): p_.copy_(e.to(p_.dtype))
    else:
        with torch.no_grad():
            for p_, k_ in zip(ema_params, keep): p_.copy_(k_)
    torch.save({"K": K, "blocks": {DEPTH - K + i: {k: v.half().cpu() for k, v in b.state_dict().items()} for i, b in enumerate(m.train_blocks)},
                "norm": {k: v.half().cpu() for k, v in enc.norm.state_dict().items()}, "head": {k: v.cpu() for k, v in m.head.state_dict().items()}}, OUT / f"weights_{tag}.pt")
    if len(va):
        lp = predict(lambda i: FRAMES[va[i]], len(va)); np.save(OUT / f"val_logp_{tag}.npy", lp); np.save(OUT / "val_rows.npy", va)
        p = score(lp); auc, f1 = roc_auc_score(ycheat[va], p), best_f1(ycheat[va], p)
        summary[f"{tag}_val_auc"], summary[f"{tag}_val_f1"] = float(auc), float(f1)
        print(f"{tag}: held-out clips AUC {auc:.4f} | best F1 {f1:.4f}", flush=True)
    if sub_ids:
        lp = predict(lambda i: read_seq(sub_ids[i]), len(sub_ids)); np.save(OUT / f"test_logp_{tag}.npy", lp)
        print(f"{tag}: test predicted, mean score {score(lp).mean():.3f}", flush=True)
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print("done", flush=True)
