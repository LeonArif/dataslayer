"""Three new per-frame signal families for the sequence members, same layout as the flow caches (cache/<name>/flow_series_{train,test}.npy, (6166, 30, C) and (1515, 30, C), row order of
cache/members_2026-11-20/clips.csv and sample_submission), so they plug into the E3 / E58 members through FLOW=1 FLOWDIR=../cache/<name>. env MODE selects the family:
 depth  (cache/depth, 27 ch): Depth Anything V2 Small (relative inverse depth, larger = closer) on the whole 640x360 frame (resized to 294x518), per frame normalised by its 2nd / 98th percentile to 0..1, sampled at the cached
        ViTPose keypoints (7x7 mean). Channels: torso depth (mean of shoulders + hips), nose - torso, left ear - right ear, left - right shoulder, left - right hip, both elbows - torso, both wrists - torso,
        head region mean - torso, 4x4 grid of the person box (mean depth - torso), std of the depth inside the person box. Gives forward lean, head turn in depth, hands in front of or behind the body.
 track  (cache/track, 31 ch): CoTracker3 (offline, scaled_offline.pth, window 60) queried at frame 0 on the 17 keypoints + a 6x6 grid in the person box, tracked through all 30 frames. Per group (head, left wrist,
        right wrist, left elbow, right elbow, shoulders, box grid) the displacement from frame 0 (dx, dy) and the travelled path length so far, both divided by the person box height, and the mean visibility = 28;
        + spread of the grid displacement (non-rigid motion) + left / right wrist to head distance.
 objdet (cache/objdet, 26 ch): RT-DETR (the same PekingU/rtdetr_r50vd_coco_o365 as the person boxes) on every frame; for the classes cell phone, book, clock, laptop: top score, score x box centre (x, y) relative to the
        person box, score x box area relative to the person box, score x distance to the nearest wrist and to the head (in person box heights) = 6 each = 24; + number of persons with score > 0.5 minus 1 and the top 'remote' score.
Frames are decoded and recoded in the test delivery format (640x360, JPEG q90). Resumable per video / test part. env LIMIT = n train videos and 2 test parts (speed test, nothing final saved).
usage (from leon/): MODE=depth python scripts/extract_newsig.py"""
import os, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parents[2]; MODE = os.environ["MODE"]; OUT = ROOT / f"cache/{MODE}"; (OUT / "train").mkdir(parents=True, exist_ok=True); (OUT / "test_parts").mkdir(exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", 0)); CONF = 0.3; cv2.setNumThreads(2)
if MODE == "depth":
    from transformers import AutoModelForDepthEstimation
    dm = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").cuda().half().eval()
    IM_MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); IM_STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
elif MODE == "track":
    sys.path.insert(0, str(ROOT / "cache/cotracker3/co-tracker"))
    from cotracker.predictor import CoTrackerPredictor
    tracker = CoTrackerPredictor(checkpoint=str(ROOT / "cache/cotracker3/scaled_offline.pth"), offline=True, v2=False, window_len=60).cuda().eval()
elif MODE == "objdet":
    from transformers import RTDetrForObjectDetection
    det = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").cuda().half().eval()
else:
    raise SystemExit("MODE must be depth, track or objdet")


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def keypoints(kp):                                            # (30, 17, 3) -> (30, 17, 2) with low-confidence points replaced by the clip median of that point
    k = np.array(kp, np.float32); med = np.nanmedian(np.where(k[..., 2:3] >= CONF, k[..., :2], np.nan), axis=0)
    return np.where(k[..., 2:3] >= CONF, k[..., :2], np.where(np.isnan(med), 0, med)[None])


@torch.inference_mode()
def depth_feats(frames, kp, box):
    X = torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float() / 255; xy = torch.from_numpy(keypoints(kp)).cuda()      # (30, 17, 2) full-frame pixels
    d = dm(pixel_values=((F.interpolate(X, (294, 518), mode="bilinear", align_corners=False) - IM_MEAN) / IM_STD).half()).predicted_depth.float()    # (30, 294, 518)
    if d.dim() == 3: d = d[:, None]
    d = F.interpolate(d, (360, 640), mode="bilinear", align_corners=False)[:, 0]
    q = torch.quantile(d.reshape(30, -1)[:, ::7], torch.tensor([0.02, 0.98], device="cuda"), dim=1); dn = ((d - q[0][:, None, None]) / (q[1] - q[0] + 1e-6)[:, None, None]).clamp(-0.2, 1.2)
    sm = F.avg_pool2d(dn[:, None], 7, 1, 3)[:, 0]; ix = xy[..., 0].round().long().clamp(0, 639); iy = xy[..., 1].round().long().clamp(0, 359)
    at = lambda j: sm[torch.arange(30), iy[:, j], ix[:, j]]
    torso = (at(5) + at(6) + at(11) + at(12)) / 4
    hc = xy[:, :5].mean(1); hx = hc[:, 0].round().long().clamp(0, 639); hy = hc[:, 1].round().long().clamp(0, 359)
    head = F.avg_pool2d(dn[:, None], 21, 1, 10)[:, 0][torch.arange(30), hy, hx]
    b = torch.tensor([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=torch.float32, device="cuda"); bx = torch.cat([torch.arange(30, device="cuda")[:, None].float(), b[None].expand(30, 4)], 1)
    g = roi_align(dn[:, None], bx, (4, 4), 1.0, 2, True).reshape(30, 16); fine = roi_align(dn[:, None], bx, (16, 16), 1.0, 2, True).reshape(30, -1)
    ch = torch.stack([torso, at(0) - torso, at(3) - at(4), at(5) - at(6), at(11) - at(12), at(7) - torso, at(8) - torso, at(9) - torso, at(10) - torso, head - torso], 1)
    return torch.cat([ch, g - torso[:, None], fine.std(1)[:, None]], 1).cpu().numpy().astype(np.float32)        # (30, 27)


@torch.inference_mode()
def track_feats(frames, kp, box):
    v = torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float()[None]; xy = keypoints(kp)                                         # (1, 30, 3, 360, 640)
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[0] + box[2]), float(box[1] + box[3]); s = max(y2 - y1, 1.0)
    gx, gy = np.meshgrid(np.linspace(x1, x2, 6), np.linspace(y1, y2, 6)); grid = np.stack([gx.ravel(), gy.ravel()], 1)
    pts = np.concatenate([xy[0], grid], 0).astype(np.float32); qs = torch.from_numpy(np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1))[None].cuda()
    tr, vis = tracker(v, queries=qs); tr = tr[0].float().cpu().numpy(); vis = vis[0].float().cpu().numpy()                                       # (30, N, 2), (30, N)
    groups = [list(range(0, 5)), [9], [10], [7], [8], [5, 6], list(range(17, 53))]; out = []
    for gi in groups:
        p = tr[:, gi].mean(1); disp = (p - p[0]) / s; path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]) / s
        out += [disp[:, 0], disp[:, 1], path, vis[:, gi].mean(1)]
    gd = (tr[:, 17:53] - tr[:1, 17:53]) / s; out.append(np.linalg.norm(gd - gd.mean(1, keepdims=True), axis=2).mean(1))
    head = tr[:, :5].mean(1); out += [np.linalg.norm(tr[:, 9] - head, axis=1) / s, np.linalg.norm(tr[:, 10] - head, axis=1) / s]
    return np.stack(out, 1).astype(np.float32)                                                                                              # (30, 31)


CLS = [67, 73, 74, 63]                                         # cell phone, book, clock, laptop


@torch.inference_mode()
def objdet_feats(frames, kp, box):
    X = torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float() / 255; xy = keypoints(kp)
    o = det(pixel_values=F.interpolate(X, (640, 640), mode="bilinear", align_corners=False).half()); pr = o.logits.float().sigmoid().cpu().numpy(); bb = o.pred_boxes.float().cpu().numpy()   # (30, 300, 80), (30, 300, 4) cxcywh in 0..1
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[0] + box[2]), float(box[1] + box[3]); s = max(y2 - y1, 1.0); bw = max(x2 - x1, 1.0); res = np.zeros((30, 26), np.float32)
    for t in range(30):
        head = xy[t, :5].mean(0); wr = xy[t, 9:11]; col = 0
        for c in CLS:
            j = int(pr[t, :, c].argmax()); sc = float(pr[t, j, c]); cx, cy, w, h = bb[t, j]; cx, cy, w, h = cx * 640, cy * 360, w * 640, h * 360
            dw = float(np.min(np.linalg.norm(wr - np.array([cx, cy]), axis=1))) / s; dh = float(np.linalg.norm(head - np.array([cx, cy]))) / s
            res[t, col:col + 6] = [sc, sc * (cx - x1) / bw, sc * (cy - y1) / s, sc * (w * h) / (bw * s), sc * dw, sc * dh]; col += 6
        res[t, 24] = max(float((pr[t, :, 0] > 0.5).sum()) - 1.0, 0.0); res[t, 25] = float(pr[t, :, 65].max())
    return res


FEATS = dict(depth=depth_feats, track=track_feats, objdet=objdet_feats)[MODE]
clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")
meta = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_train_s30_meta.csv"); kpall = np.load(ROOT / "cache/pose_vitpose_all/pose_train_s30.npy", mmap_mode="r"); boxall = np.load(ROOT / "cache/pose_vitpose_all/box_train_s30.npy")
meta["key"] = [("cheat/" if l == 1 else "normal/") + v.replace(".MOV", "") for v, l in zip(meta.video, meta.label)]; look = {(k, s): i for i, (k, s) in enumerate(zip(meta.key, meta.start))}
t0 = time.time()
ids = list(pd.read_csv(ROOT / "data/sample_submission.csv")["id"]); tm = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_test_meta.csv"); trow = {i: k for k, i in enumerate(tm["id"])}
kpt = np.load(ROOT / "cache/pose_vitpose_all/pose_test.npy", mmap_mode="r"); boxt = np.load(ROOT / "cache/pose_vitpose_all/box_test.npy"); parts = [ids[i:i + 50] for i in range(0, len(ids), 50)]
for pi, part in enumerate(parts[:(2 if LIMIT else None)]):
    f = OUT / "test_parts" / f"part_{pi:03d}.npy"
    if f.exists(): continue
    res = []
    for sid in part:
        files = sorted((ROOT / "data/test" / sid).glob("*.jpg")); assert len(files) == 30
        res.append(FEATS([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files], np.asarray(kpt[trow[sid]]), boxt[trow[sid]]))
    np.save(f, np.stack(res)); print(f"test part {pi + 1}/{len(parts)} | {time.time() - t0:.0f}s", flush=True)
if not LIMIT and not (OUT / "flow_series_test.npy").exists():
    np.save(OUT / "flow_series_test.npy", np.concatenate([np.load(OUT / "test_parts" / f"part_{i:03d}.npy") for i in range(len(parts))])); print("test series saved", flush=True)
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
        i = look[(v, sec * 30)]; res.append(FEATS(frames[sec * 30:sec * 30 + 30], np.asarray(kpall[i]), boxall[i]))
    if not LIMIT: np.save(OUT / "train" / (v.replace("/", "__") + ".npy"), np.stack(res))
    if k % 10 == 0: print(f"train {k + 1}/{len(groups)} | {time.time() - t0:.0f}s", flush=True)
if not LIMIT:
    rows = {v: np.load(OUT / "train" / (v.replace("/", "__") + ".npy")) for v in clips.video.unique()}; ctr = {v: 0 for v in rows}; series = []
    for v in clips.video:
        series.append(rows[v][ctr[v]]); ctr[v] += 1
    np.save(OUT / "flow_series_train.npy", np.stack(series)); print("train series saved", np.stack(series).shape, flush=True)
print("done", flush=True)
