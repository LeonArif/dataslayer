# Head pose (pitch, yaw, roll in degrees) per frame with 6DRepNet (Hempel et al., 2022; RepVGG-B1g2, weights 6DRepNet_300W_LP_AFLW2000.pth, a pretrained feature extractor as in FAQ #56) for every frame of every labelled train clip
# and of every test sequence. The head crop of every frame comes from the cached ViTPose keypoints (nose, eyes, ears): a square around the face points. Frames are in the test delivery format (640x360, JPEG q90; test jpgs are read as delivered).
# Output: cache/headpose/train/<video with __>.npz (seconds, feat (n, 30, 3)) and cache/headpose/test.npy (1515, 30, 3) in sample_submission order. Resumable. env LIMIT=n videos / 2 test parts (speed test).
# Setup (once): the package files of sixdrepnet 0.1.6 unpacked (not installed) into cache/sixdrepnet_pkg/ and the weights downloaded to cache/sixdrepnet_weights/
#   https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth (loaded with weights_only=True).
# usage (from leon/): python scripts/extract_headpose.py
import os, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "cache/sixdrepnet_pkg/sixdrepnet"))
from model import SixDRepNet
import utils as sixd_utils
LIMIT = int(os.environ.get("LIMIT", 0)); OUT = ROOT / os.environ.get("HP_OUT", "cache/headpose"); (OUT / "train").mkdir(parents=True, exist_ok=True); (OUT / "test_parts").mkdir(exist_ok=True)
cv2.setNumThreads(2)
model = SixDRepNet(backbone_name="RepVGG-B1g2", backbone_file="", deploy=True, pretrained=False)
model.load_state_dict(torch.load(ROOT / "cache/sixdrepnet_weights/6DRepNet_300W_LP_AFLW2000.pth", map_location="cpu", weights_only=True))
model = model.cuda().eval()
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def face_crop(rgb, kp):                                  # rgb (360, 640, 3), kp (17, 3) pixels -> 224x224 crop around the face points
    pts = kp[:5][kp[:5, 2] > 0.3]
    if len(pts) < 2: pts = kp[:5]
    x0, x1, y0, y1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    side = max((x1 - x0) * 1.5, (y1 - y0) * 2.2, 24.0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.1 * side
    M = np.array([[224 / side, 0, 112 - cx * 224 / side], [0, 224 / side, 112 - cy * 224 / side]], np.float32)
    return cv2.warpAffine(rgb, M, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def pose_clip(rgb_frames, kps):                          # 30 RGB frames + keypoints (30, 17, 3) -> (30, 3) pitch, yaw, roll in degrees
    crops = np.stack([(face_crop(f, k).astype(np.float32) / 255 - MEAN) / STD for f, k in zip(rgb_frames, kps)]).transpose(0, 3, 1, 2)
    R = model(torch.from_numpy(crops).cuda())
    return (sixd_utils.compute_euler_angles_from_rotation_matrices(R) * 180 / np.pi).float().cpu().numpy().astype(np.float32)


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
        res.append(pose_clip([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files], np.asarray(kpt[trow[sid]])))
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
        kps = np.asarray(kpall[look[(v, sec * 30)]]); feats.append(pose_clip([recode(frames[sec * 30 + i]) for i in range(30)], kps)); secs.append(sec)
    np.savez(OUT / "train" / (v.replace("/", "__") + ".npz"), seconds=np.array(secs), feat=np.stack(feats))
    print(f"video {k + 1}/{len(groups)} {v} clips {len(secs)}/{len(g)} | {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
