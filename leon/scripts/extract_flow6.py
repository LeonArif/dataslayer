"""VERSION 6 of extract_flow4.py: SEA-RAFT (princeton-vl, BSD-3, config spring-M, weights MemorySlices/Tartan-C-T-TSKH-spring540x960-M from Hugging Face, 4 iterations) instead of torchvision RAFT-large; crops resized to 224x224 (v4: 160); same regions, same 60 channels, pairs k -> k+1. Output cache/flow6.
VERSION 4 of extract_flow3.py: RAFT-LARGE (torchvision C_T_SKHT_V2) instead of RAFT-small, crops resized to 160x160, head side 3 x ear distance in 48..192 px, wrist side 128 px (larger hand and paper region). Output cache/flow4 (same 60 channels).
VERSION 3 text: : detail flow of three small crops (head, left wrist, right wrist; head side = 3 x ear distance in 48..160 px, wrist side 96 px of the 640x360 frame, resized to 128x128, same box on both frames of a pair) with RAFT-small:
per region 3x3 grid of mean (dx, dy) + mean and std of the magnitude = 20 values, 60 per frame, output cache/flow3/flow_series_{train,test}.npy; field_* files are dummies.
VERSION 2 text: : 27 channels per frame (the 12 of version 1 + shoulders, elbows, torso, each dx, dy, magnitude) and the raw flow field of every frame pooled to 32x32 (fp16) so any region feature can be derived later without rerunning RAFT.
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
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "cache/flow6"; (OUT / "train").mkdir(parents=True, exist_ok=True); LIMIT = int(os.environ.get("LIMIT", 0))
R, CONF = 224, 0.3
import sys, json, argparse
SEA = os.environ.get("SEA_RAFT_DIR", "C:/Users/leouw/AppData/Local/Temp/claude/2eaf2cf1aa6e220d/20cd342f-a458-47fa-97b9-7d3619f8d62b/scratchpad/searaft/repo")
sys.path.insert(0, SEA + "/core")
from raft import RAFT                                     # SEA-RAFT code (read before use)
_args = argparse.Namespace(**json.load(open(SEA + "/config/eval/spring-M.json")))
model = RAFT.from_pretrained("MemorySlices/Tartan-C-T-TSKH-spring540x960-M", args=_args).cuda().eval(); cv2.setNumThreads(2)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def clip_flow(frames, kp, box):                              # frames: 30 RGB uint8 (360, 640, 3); kp (30, 17, 3) -> ((30, 60), dummy)
    X = torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float()                # 0..255, SEA-RAFT normalises inside                # (30, 3, 360, 640)
    k = np.array(kp, np.float32); med = np.nanmedian(np.where((k[..., 2:3] >= CONF), k[..., :2], np.nan), axis=0)
    xy = np.where(k[..., 2:3] >= CONF, k[..., :2], np.where(np.isnan(med), 0, med)[None])                  # (30, 17, 2) full-frame pixels
    ear = float(np.median(np.linalg.norm(xy[:, 3] - xy[:, 4], axis=-1))); side_h = float(np.clip(3.0 * ear, 48, 192))
    regions = [(xy[:29, :5].mean(1), side_h), (xy[:29, 9], 128.0), (xy[:29, 10], 128.0)]                       # head, left wrist, right wrist: centre of frame k, same box on frames k and k+1
    feats = []
    for c, side in regions:
        bx = torch.from_numpy(np.concatenate([np.arange(29)[:, None], c - side / 2, c + side / 2], 1).astype(np.float32)).cuda()
        a_ = roi_align(X[:-1], bx, (R, R), 1.0, 2, True); b_ = roi_align(X[1:], bx, (R, R), 1.0, 2, True)
        fl = model(a_.contiguous(), b_.contiguous(), iters=4, test_mode=True)['flow'][-1] * (side / R)                                         # (29, 2, R, R) in original pixels per frame step
        mag = torch.linalg.norm(fl, dim=1)
        feats.append(torch.cat([torch.nn.functional.adaptive_avg_pool2d(fl, 3).reshape(29, 18), mag.mean((1, 2))[:, None], mag.std((1, 2))[:, None]], 1))
    out = torch.cat(feats, 1).cpu().numpy().astype(np.float32); out = np.concatenate([out, out[-1:]])        # (30, 60)
    return out, np.zeros((30, 1), np.float16)

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
