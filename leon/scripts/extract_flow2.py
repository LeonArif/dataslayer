"""VERSION 2 of extract_flow.py: 27 channels per frame (the 12 of version 1 + shoulders, elbows, torso, each dx, dy, magnitude) and the raw flow field of every frame pooled to 32x32 (fp16) so any region feature can be derived later without rerunning RAFT.
Output cache/flow2/{flow_series_train (6166,30,27), flow_series_test (1515,30,27), field_train (6166,30,2,32,32) fp16, field_test (1515,30,2,32,32)}.
Original description of version 1: Optical-flow motion channels per frame (RAFT-small, torchvision, weights C_T_V2 from download.pytorch.org) for every labelled train clip and all 30 frames of every test sequence.
Per clip: the person box (cached RT-DETR box of pose_vitpose_all, expanded by 15 %) is cropped from every frame and resized to 256x256; RAFT runs on the 29 consecutive pairs; the flow of pair k -> k+1 is assigned to frame k (the last frame repeats the 28th pair).
12 channels per frame (all in crop pixels per frame step): mean flow dx, dy and magnitude over (1) the head (circle around the mean of nose, eyes, ears, radius 0.8 x ear-to-ear distance, at least 10 px), (2) the left wrist (circle r 12 px), (3) the right wrist (r 12 px), (4) the whole person crop.
Keypoints are the cached ViTPose base ones (30 frames per window), points with confidence < 0.3 fall back to the clip median of that point.
Output (same layout as cache/gaze_lle): cache/flow/flow_series_train.npy (6166, 30, 12) in the row order of cache/members_2026-11-20/clips.csv, cache/flow/flow_series_test.npy (1515, 30, 12) in sample_submission order.
Train resumable per video (cache/flow/train/<video with __>.npy). env: LIMIT = n train videos and 2 test parts (speed test, nothing final saved). usage (from leon/): python scripts/extract_flow.py"""
import os, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "cache/flow2"; (OUT / "train").mkdir(parents=True, exist_ok=True); LIMIT = int(os.environ.get("LIMIT", 0))
S, PAD, CONF = 256, 0.15, 0.3
model = raft_small(weights=Raft_Small_Weights.C_T_V2).cuda().eval(); cv2.setNumThreads(2)
YY, XX = torch.meshgrid(torch.arange(S, device="cuda").float(), torch.arange(S, device="cuda").float(), indexing="ij")


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def clip_flow(frames, kp, box):                              # frames: 30 RGB uint8 (360, 640, 3); kp (30, 17, 3); box xywh -> ((30, 27), (30, 2, 32, 32))
    x, y, w, h = [float(v) for v in box]; cx, cy = x + w / 2, y + h / 2; w, h = w * (1 + 2 * PAD), h * (1 + 2 * PAD)
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2)); x1, y1 = int(min(640, cx + w / 2)), int(min(360, cy + h / 2))
    if x1 - x0 < 16 or y1 - y0 < 16: x0, y0, x1, y1 = 0, 0, 640, 360
    sx, sy = S / (x1 - x0), S / (y1 - y0)
    crops = np.stack([cv2.resize(f[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_AREA) for f in frames])
    t = torch.from_numpy(crops).cuda().permute(0, 3, 1, 2).float() / 255 * 2 - 1
    fl = model(t[:-1].contiguous(), t[1:].contiguous())[-1]            # (29, 2, S, S)
    k = np.array(kp, np.float32); med = np.nanmedian(np.where((k[..., 2:3] >= CONF), k[..., :2], np.nan), axis=0)
    xy = np.where(k[..., 2:3] >= CONF, k[..., :2], np.where(np.isnan(med), 0, med)[None])
    xy = (xy - np.array([x0, y0], np.float32)) * np.array([sx, sy], np.float32)           # keypoints in crop pixels (30, 17, 2)
    xy_t = torch.from_numpy(xy[:29]).cuda()                                                      # (29, 17, 2)
    ctr = torch.stack([xy_t[:, :5].mean(1), xy_t[:, 9], xy_t[:, 10], xy_t[:, 5], xy_t[:, 6], xy_t[:, 7], xy_t[:, 8], xy_t[:, [5, 6, 11, 12]].mean(1)], 1)       # head, Lwrist, Rwrist, Lshoulder, Rshoulder, Lelbow, Relbow, torso (29, 8, 2)
    rad = torch.full((29, 8), 12.0, device="cuda"); rad[:, 0] = torch.clamp(0.8 * torch.linalg.norm(xy_t[:, 3] - xy_t[:, 4], dim=-1), min=10.0); rad[:, 3:5] = 16.0; rad[:, 5:7] = 14.0; rad[:, 7] = 28.0
    m = (((XX[None, None] - ctr[..., 0, None, None]) ** 2 + (YY[None, None] - ctr[..., 1, None, None]) ** 2) <= rad[..., None, None] ** 2).float()          # (29, 8, S, S)
    cnt = m.sum((2, 3)).clamp(min=1); reg = torch.einsum("tkhw,tchw->tkc", m, fl) / cnt[..., None]                                                          # (29, 8, 2)
    reg = torch.cat([reg, fl.mean((2, 3))[:, None]], 1)                                                                                                    # + whole crop (29, 9, 2)
    order = [0, 1, 2, 8, 3, 4, 5, 6, 7]                                                                                                                    # head, Lwrist, Rwrist, body (the 12 of version 1 in this order), then shoulders, elbows, torso
    reg = reg[:, order]; ch = torch.cat([reg, torch.linalg.norm(reg, dim=-1, keepdim=True)], -1).reshape(29, -1).cpu().numpy()                             # (29, 27)
    out = np.concatenate([ch, ch[-1:]]).astype(np.float32)
    fld = torch.nn.functional.avg_pool2d(fl, 8).cpu().numpy().astype(np.float16); fld = np.concatenate([fld, fld[-1:]])                                    # (30, 2, 32, 32)
    return out, fld


clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")
meta = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_train_s30_meta.csv"); kpall = np.load(ROOT / "cache/pose_vitpose_all/pose_train_s30.npy", mmap_mode="r"); boxall = np.load(ROOT / "cache/pose_vitpose_all/box_train_s30.npy")
meta["key"] = [("cheat/" if l == 1 else "normal/") + v.replace(".MOV", "") for v, l in zip(meta.video, meta.label)]; look = {(k, s): i for i, (k, s) in enumerate(zip(meta.key, meta.start))}
t0 = time.time()
ids = list(pd.read_csv(ROOT / "data/sample_submission.csv")["id"]); tm = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_test_meta.csv"); trow = {i: k for k, i in enumerate(tm["id"])}
kpt = np.load(ROOT / "cache/pose_vitpose_all/pose_test.npy", mmap_mode="r"); boxt = np.load(ROOT / "cache/pose_vitpose_all/box_test.npy")
(OUT / "test_parts").mkdir(exist_ok=True); parts = [ids[i:i + 50] for i in range(0, len(ids), 50)]
for pi, part in enumerate(parts[:(2 if LIMIT else None)]):
    f = OUT / "test_parts" / f"part_{pi:03d}.npy"
    if f.exists(): continue
    res = []
    for sid in part:
        files = sorted((ROOT / "data/test" / sid).glob("*.jpg")); assert len(files) == 30
        res.append(clip_flow([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files], np.asarray(kpt[trow[sid]]), boxt[trow[sid]]))
    np.save(f, np.stack([r[0] for r in res])); np.save(f.with_name(f.stem + "_field.npy"), np.stack([r[1] for r in res])); print(f"test part {pi + 1}/{len(parts)} | {time.time() - t0:.0f}s", flush=True)
if not LIMIT and not (OUT / "flow_series_test.npy").exists(): (np.save(OUT / "flow_series_test.npy", np.concatenate([np.load(OUT / "test_parts" / f"part_{i:03d}.npy") for i in range(len(parts))])), np.save(OUT / "field_test.npy", np.concatenate([np.load(OUT / "test_parts" / f"part_{i:03d}_field.npy") for i in range(len(parts))])), print("test series saved", flush=True))
groups = [(v, g) for v, g in clips.groupby("video", sort=False) if not (OUT / "train" / (v.replace("/", "__") + ".npy")).exists()]
if LIMIT: groups = groups[:LIMIT]
for k, (v, g) in enumerate(groups):
    cap = cv2.VideoCapture(str(ROOT / "data/train" / (v + ".MOV"))); frames = []
    while True:
        ok, im = cap.read()
        if not ok: break
        frames.append(recode(im))
    cap.release(); res = []
    for sec in g.second.astype(int):
        i = look[(v, sec * 30)]; res.append(clip_flow(frames[sec * 30:sec * 30 + 30], np.asarray(kpall[i]), boxall[i]))
    if not LIMIT: np.save(OUT / "train" / (v.replace("/", "__") + "_field.npy"), np.stack([r[1] for r in res])); np.save(OUT / "train" / (v.replace("/", "__") + ".npy"), np.stack([r[0] for r in res]))
    if k % 10 == 0: print(f"train {k + 1}/{len(groups)} | {time.time() - t0:.0f}s", flush=True)
if not LIMIT:
    for suf, name in (("", "flow_series_train.npy"), ("_field", "field_train.npy")):
        rows = {v: np.load(OUT / "train" / (v.replace("/", "__") + suf + ".npy")) for v in clips.video.unique()}; ctr = {v: 0 for v in rows}; series = []
        for v in clips.video:
            series.append(rows[v][ctr[v]]); ctr[v] += 1
        np.save(OUT / name, np.stack(series)); print(name, "saved", np.stack(series).shape, flush=True)
print("done", flush=True)
