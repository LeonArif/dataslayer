# Fine-tuning step 2: train the top K of the 32 ViT-H+ blocks (+ final norm + an 8-action head) on frames, 5 folds
# by subject (same folds as main_legit). Input = cached output of block 20 (ft_cache.py).
# Frame labels: before the action starts the person writes -> Writing_Answers; after the start -> the video's action.
# Start: by-eye start for cheat front videos (side = front + 0.5 s); other videos: < 4.25 s writing, >= 9.75 s action.
# Saves the 8-action log-probabilities of every held-out train frame and every test frame.
# usage: python ft_train.py K [K ...]
import sys, time, math, os
from pathlib import Path
import numpy as np, pandas as pd, timm, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "cache/ft_hplus_block20"
OUT = C / os.environ.get("FT_OUT", "preds_v2"); OUT.mkdir(exist_ok=True)
SPLIT, IMG_HW, DEVICE, SEED = 20, (224, 400), "cuda", 42
EPOCHS, BS, LR_BLOCKS, LR_HEAD, WD = int(sys.argv[2]) if len(sys.argv) > 2 else 2, 16, 2e-5, 1e-3, 0.05
import os
BENCH = int(os.environ.get('BENCH', 0))
KS = [int(k) for k in sys.argv[1].split(",")]
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

meta = pd.read_csv(C / "train_meta.csv").merge(train[["video", "fold", "action_id", "start"]], on="video", how="left")
assert meta["fold"].notna().all()
lab = np.full(len(meta), -1)
for i, (a, s, t) in enumerate(zip(meta.action_id, meta.start, meta.sec)):
    if a == WRITE:
        lab[i] = WRITE
    elif not np.isnan(s):
        lab[i] = a if t >= s + 0.5 else (WRITE if t < s - 0.5 else -1)
    else:
        lab[i] = a if t >= 9.75 else (WRITE if t < 4.25 else -1)
meta["y"] = lab
from rowreader import RowReader
tok_tr, tok_te = RowReader(C / "train_tokens.npy"), RowReader(C / "test_tokens.npy")
print("frames", len(meta), "labelled", (lab >= 0).sum(), "per class", np.bincount(lab[lab >= 0], minlength=8), flush=True)

base = timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=IMG_HW).eval()
with torch.no_grad():
    _, ROPE = base._pos_embed(base.patch_embed(torch.zeros(1, 3, *IMG_HW)))
ROPE = ROPE.to(DEVICE)
TOP_STATE = {i: {k: v.clone() for k, v in base.blocks[i].state_dict().items()} for i in range(SPLIT, 32)}
NORM_STATE = {k: v.clone() for k, v in base.norm.state_dict().items()}
NPRE = base.num_prefix_tokens


def load(tok, idx):
    x = torch.from_numpy(np.ascontiguousarray(tok[np.sort(idx)]).view(np.int16)).view(torch.bfloat16)
    return x.to(DEVICE, non_blocking=True)


class Top(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.frozen = nn.ModuleList([base.blocks[i] for i in range(SPLIT, 32 - k)])
        self.train_blocks = nn.ModuleList([base.blocks[i] for i in range(32 - k, 32)])
        self.norm = base.norm
        self.head = nn.Sequential(nn.BatchNorm1d(2 * base.embed_dim, affine=False), nn.Dropout(0.2), nn.Linear(2 * base.embed_dim, len(ACTIONS)))
        for p in self.frozen.parameters(): p.requires_grad_(False)
        for p in self.train_blocks.parameters(): p.requires_grad_(True)
        for p in self.norm.parameters(): p.requires_grad_(k > 0)
        self.frozen.to(torch.bfloat16)

    def forward(self, x, feats=False):
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            for b in self.frozen:
                x = b(x, rope=ROPE)
        x = x.detach()
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            for b in self.train_blocks:
                x = checkpoint(b, x, rope=ROPE, use_reentrant=False) if self.training else b(x, rope=ROPE)
            x = self.norm(x)
            f = torch.cat([x[:, 0], x[:, NPRE:].mean(1)], 1)
        return (self.head(f.float()), f.float()) if feats else self.head(f.float())


@torch.no_grad()
def predict(m, tok, idx):
    m.eval(); out, fe = [], []
    for s in range(0, len(idx), 32):
        lo, f = m(load(tok, idx[s:s + 32]), feats=True)
        out.append(F.log_softmax(lo, 1).float().cpu().numpy()); fe.append(f.cpu().numpy().astype(np.float16))
    return np.concatenate(out), np.concatenate(fe)


for K in KS:
    for fold in range(5):
        f_val, f_test = OUT / f"k{K}_f{fold}_val.npy", OUT / f"k{K}_f{fold}_test.npy"
        if (OUT / f"k{K}_f{fold}_feat_test.npy").exists():
            continue
        for i in range(SPLIT, 32):
            base.blocks[i].float(); base.blocks[i].load_state_dict(TOP_STATE[i])
        base.norm.load_state_dict(NORM_STATE)
        torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
        m = Top(K).to(DEVICE)
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f'K={K} fold={fold} trainable params {n_train / 1e6:.1f}M', flush=True)
        assert n_train > K * 20e6
        tr = np.where((meta.fold != fold).values & (meta.y >= 0).values)[0]
        va = np.where((meta.fold == fold).values)[0]
        y_all = torch.tensor(meta.y.values, device=DEVICE)
        tr_write, tr_act = tr[meta.y.values[tr] == WRITE], tr[meta.y.values[tr] != WRITE]
        n_write = len(tr_write) // 3
        cnt = np.bincount(meta.y.values[tr_act], minlength=8); cnt[WRITE] = n_write
        w = torch.tensor((cnt.sum() / (8 * cnt)) ** 0.5, dtype=torch.float32, device=DEVICE)
        params = [{"params": m.head.parameters(), "lr": LR_HEAD}]
        if K > 0:
            params.append({"params": list(m.train_blocks.parameters()) + list(m.norm.parameters()), "lr": LR_BLOCKS})
        opt = torch.optim.AdamW(params, weight_decay=WD)
        steps = EPOCHS * math.ceil((len(tr_act) + n_write) / BS); warm = max(1, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
        t0, step = time.time(), 0
        for ep in range(EPOCHS):
            m.train(); m.frozen.eval(); perm = np.random.permutation(np.r_[tr_act, np.random.choice(tr_write, n_write, replace=False)]); tot = 0.0
            for s in range(0, len(perm), BS):
                b = np.sort(perm[s:s + BS])
                loss = F.cross_entropy(m(load(tok_tr, b)), y_all[torch.from_numpy(b).to(DEVICE)], weight=w, label_smoothing=0.1)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sched.step(); step += 1; tot += loss.item() * len(b)
                if BENCH and step == BENCH:
                    torch.cuda.synchronize(); print(f'BENCH K={K}: {BENCH * BS / (time.time() - t0):.1f} frames/s, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB', flush=True); sys.exit()
            print(f"K={K} fold={fold} epoch={ep} loss={tot / len(perm):.4f} | {time.time() - t0:.0f}s", flush=True)
        lp_all, fe_all = predict(m, tok_tr, np.arange(len(meta)))
        pv = lp_all[va]
        np.save(OUT / f"k{K}_f{fold}_feat_train.npy", fe_all)
        yv = meta.y.values[va]; ok = yv >= 0
        print(f"K={K} fold={fold} held-out frame accuracy {(pv.argmax(1)[ok] == yv[ok]).mean():.4f} | {time.time() - t0:.0f}s | gpu peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        np.save(f_val, np.c_[va, pv])
        lp_te, fe_te = predict(m, tok_te, np.arange(len(tok_te)))
        np.save(OUT / f"k{K}_f{fold}_feat_test.npy", fe_te)
        np.save(f_test, lp_te)
        if K > 0:
            torch.save({"K": K, "blocks": {i: {k: v.to(torch.bfloat16).cpu() for k, v in base.blocks[i].state_dict().items()} for i in range(32 - K, 32)},
                        "norm": {k: v.to(torch.bfloat16).cpu() for k, v in base.norm.state_dict().items()}}, OUT / f"k{K}_f{fold}_weights.pt")
        del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print("done")
