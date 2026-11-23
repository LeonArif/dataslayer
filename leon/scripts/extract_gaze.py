# Gaze-LLE (Ryan et al., CVPR 2025; decoder checkpoint gazelle_dinov2_vitl14_inout + frozen DINOv2 ViT-L backbone from the public hub; FAQ #56 allows it as a feature extractor) for every frame of every labelled train clip
# and of every test sequence. The head box of every frame comes from the cached ViTPose keypoints (nose, eyes, ears). Frames are in the test delivery format (640x360, JPEG q90; test jpgs are read as delivered).
# Per frame 6 numbers: gaze point x, y (argmax of the 64x64 heatmap, 0..1 of the frame), soft gaze x, y (heatmap-weighted mean), in-frame score, heatmap peak.
# Output: cache/gaze_lle/train/<video with __>.npz (seconds, feat (n, 30, 6)) and cache/gaze_lle/test.npy (1515, 30, 6) in sample_submission order. Resumable. env LIMIT=n videos / 20 test parts (speed test).
# Setup (once): git clone https://github.com/fkryan/gazelle cache/gazelle_repo/src, and download the decoder checkpoint
#   https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14_inout.pt to cache/gazelle_repo/weights/ (loaded with weights_only=True).
# usage (from leon/): python scripts/extract_gaze.py
import os, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "cache/gazelle_repo/src"))
from gazelle.model import get_gazelle_model
LIMIT = int(os.environ.get("LIMIT", 0)); OUT = ROOT / "cache/gaze_lle"; (OUT / "train").mkdir(parents=True, exist_ok=True); (OUT / "test_parts").mkdir(exist_ok=True)
cv2.setNumThreads(2)
model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
model.load_gazelle_state_dict(torch.load(ROOT / "cache/gazelle_repo/weights/gazelle_dinov2_vitl14_inout.pt", map_location="cpu", weights_only=True))
model = model.cuda().eval()


def head_box(kp):                                         # kp (17, 3), 640x360 pixels -> normalised (xmin, ymin, xmax, ymax)
    pts = kp[:5][kp[:5, 2] > 0.3]
    if len(pts) < 2: pts = kp[:5]
    x0, x1, y0, y1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(); w = max(x1 - x0, 24.0); h = max(y1 - y0, 24.0)
    cx = (x0 + x1) / 2; x0, x1 = cx - w * 0.9, cx + w * 0.9; y0, y1 = y0 - h * 1.1, y1 + h * 0.9
    return (max(0, x0) / 640, max(0, y0) / 360, min(640, x1) / 640, min(360, y1) / 360)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def gaze_clip(rgb_frames, kps):                          # 30 RGB frames (360, 640, 3) + keypoints (30, 17, 3) -> (30, 6)
    imgs = torch.stack([transform(f) for f in rgb_frames]).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model({"images": imgs, "bboxes": [[head_box(k)] for k in kps]})
    hm = torch.stack([h[0] for h in out["heatmap"]]).float().cpu().numpy().reshape(30, 64, 64)
    flat = hm.reshape(30, -1); am = flat.argmax(1); gy, gx = np.divmod(am, 64)
    w = flat / (flat.sum(1, keepdims=True) + 1e-8); yy, xx = np.divmod(np.arange(4096), 64)
    return np.stack([(gx + 0.5) / 64, (gy + 0.5) / 64, (w * (xx + 0.5)).sum(1) / 64, (w * (yy + 0.5)).sum(1) / 64, torch.stack([h[0] for h in out["inout"]]).float().cpu().numpy(), flat.max(1)], 1).astype(np.float32)


clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")
meta = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_train_s30_meta.csv"); kpall = np.load(ROOT / "cache/pose_vitpose_all/pose_train_s30.npy", mmap_mode="r")
meta["key"] = [("cheat/" if l == 1 else "normal/") + v.replace(".MOV", "") for v, l in zip(meta.video, meta.label)]; look = {(k, s): i for i, (k, s) in enumerate(zip(meta.key, meta.start))}
t0 = time.time()
# ---- test first (all 30 frames of every sequence, as delivered)
ids = list(pd.read_csv(ROOT / "data/sample_submission.csv")["id"]); tm = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_test_meta.csv"); trow = {i: k for k, i in enumerate(tm["id"])}; kpt = np.load(ROOT / "cache/pose_vitpose_all/pose_test.npy", mmap_mode="r")
parts = [ids[i:i + 50] for i in range(0, len(ids), 50)]
for pi, part in enumerate(parts[:(2 if LIMIT else None)]):
    f = OUT / "test_parts" / f"part_{pi:03d}.npy"
    if f.exists(): continue
    res = []
    for sid in part:
        files = sorted((ROOT / "data/test" / sid).glob("*.jpg")); assert len(files) == 30
        res.append(gaze_clip([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files], np.asarray(kpt[trow[sid]])))
    np.save(f, np.stack(res)); print(f"test part {pi + 1}/{len(parts)} | {time.time() - t0:.0f}s", flush=True)
if not LIMIT and not (OUT / "test.npy").exists(): np.save(OUT / "test.npy", np.concatenate([np.load(OUT / "test_parts" / f"part_{i:03d}.npy") for i in range(len(parts))])); print("test.npy saved", flush=True)
# ---- train clips, per video
groups = [(v, g) for v, g in clips.groupby("video") if not (OUT / "train" / (v.replace("/", "__") + ".npz")).exists()]
if LIMIT: groups = groups[:LIMIT]
for k, (v, g) in enumerate(groups):
    cap = cv2.VideoCapture(str(ROOT / "data/train" / (v + ".MOV"))); frames = []
    while True:
        ok, im = cap.read()
        if not ok: break
        frames.append(im)
    cap.release(); secs, feats = [], []
    for sec in sorted(g["second"].astype(int)):
        if (sec + 1) * 30 > len(frames): continue
        kps = np.asarray(kpall[look[(v, sec * 30)]]); feats.append(gaze_clip([recode(frames[sec * 30 + i]) for i in range(30)], kps)); secs.append(sec)
    np.savez(OUT / "train" / (v.replace("/", "__") + ".npz"), seconds=np.array(secs), feat=np.stack(feats))
    print(f"video {k + 1}/{len(groups)} {v} clips {len(secs)}/{len(g)} | {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
