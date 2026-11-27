"""Head-crop DINOv3 features: for every frame of every labelled train clip and every frame of the 1,515 test sequences, a square crop around the head
(centre = mean of nose, eyes and ears of that frame from the cached ViTPose keypoints, points with confidence < 0.3 fall back to the clip median; side = 3.2 x the
median ear-to-ear distance of the clip, clipped to 48..160 px of the 640x360 test-format frame) is resized to 224x224 and passed through the frozen DINOv3 ViT-H+
(same model and pooling as extract_dino30.py: CLS + mean patch = 2560, fp16). The head is only about 60 to 80 px in the frame, so this gives the model more pixels on
the face and head direction than the whole-frame features (224x400 for the whole person).
Frames are decoded and recoded in the test delivery format (resize 640x360, JPEG quality 90), like the other extractors.
Output (same layout as cache/dinov3_hplus_224x400_step1_bf16): cache/dinov3_headcrop_224/train/<video with __>.npy (n_windows x 30, 2560) and test.npy (1515, 30, 2560).
Resumable per video. env LIMIT = n train videos (speed test, nothing final saved). usage (from leon/): python scripts/extract_headcrop.py"""
import os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd, timm, torch
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "cache/dinov3_headcrop_224"; (OUT / "train").mkdir(parents=True, exist_ok=True); (OUT / "test_parts").mkdir(exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", 0)); CONF, R, DEVICE = 0.3, 224, "cuda"
cv2.setNumThreads(2)
backbone = timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=(R, R)).to(DEVICE).eval()
MEAN = torch.tensor(backbone.pretrained_cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(backbone.pretrained_cfg["std"], device=DEVICE).view(1, 3, 1, 1)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def clip_features(frames, kp):                               # frames: 30 RGB uint8 (360, 640, 3); kp (30, 17, 3) -> (30, 2560) fp16
    X = torch.from_numpy(np.stack(frames)).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    k = np.array(kp, np.float32); med = np.nanmedian(np.where((k[..., 2:3] >= CONF), k[..., :2], np.nan), axis=0)
    xy = np.where(k[..., 2:3] >= CONF, k[..., :2], np.where(np.isnan(med), 0, med)[None])                  # (30, 17, 2)
    ear = float(np.median(np.linalg.norm(xy[:, 3] - xy[:, 4], axis=-1))); side = float(np.clip(3.2 * ear, 48, 160))
    c = xy[:, :5].mean(1)                                                                                  # head centre per frame
    bx = torch.from_numpy(np.concatenate([np.arange(30)[:, None], c - side / 2, c + side / 2], 1).astype(np.float32)).to(DEVICE)
    crops = roi_align(X, bx, (R, R), 1.0, 2, True)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        tok = backbone.forward_features((crops - MEAN) / STD)
    feat = torch.cat([tok[:, 0], tok[:, backbone.num_prefix_tokens:].mean(1)], dim=1).float()
    assert torch.isfinite(feat).all(), "features contain NaN/inf"
    return feat.cpu().numpy().astype(np.float16)


meta = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_train_s30_meta.csv"); kpall = np.load(ROOT / "cache/pose_vitpose_all/pose_train_s30.npy", mmap_mode="r")
meta["key"] = [("cheat/" if l == 1 else "normal/") + v.replace(".MOV", "") for v, l in zip(meta.video, meta.label)]
clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv"); videos = list(clips.video.unique())
t0 = time.time()

# ---- test sequences (all 30 frames, jpg as delivered)
ids = list(pd.read_csv(ROOT / "data/sample_submission.csv")["id"]); tm = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_test_meta.csv"); trow = {i: k for k, i in enumerate(tm["id"])}
kpt = np.load(ROOT / "cache/pose_vitpose_all/pose_test.npy", mmap_mode="r"); parts = [ids[i:i + 50] for i in range(0, len(ids), 50)]
for pi, part in enumerate(parts[:(2 if LIMIT else None)]):
    f = OUT / "test_parts" / f"part_{pi:03d}.npy"
    if f.exists():
        continue
    res = []
    for sid in part:
        files = sorted((ROOT / "data/test" / sid).glob("*.jpg")); assert len(files) == 30
        res.append(clip_features([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files], np.asarray(kpt[trow[sid]])))
    np.save(f, np.stack(res)); print(f"test part {pi + 1}/{len(parts)} | {time.time() - t0:.0f}s", flush=True)
if not LIMIT and not (OUT / "test.npy").exists():
    np.save(OUT / "test.npy", np.concatenate([np.load(OUT / "test_parts" / f"part_{i:03d}.npy") for i in range(len(parts))])); print("test.npy saved", flush=True)

# ---- train videos
def load_video(v):
    cap = cv2.VideoCapture(str(ROOT / "data/train" / (v + ".MOV"))); frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(recode(im))
    cap.release()
    return frames


todo = [v for v in videos if not (OUT / "train" / (v.replace("/", "__") + ".npy")).exists()]
if LIMIT:
    todo = todo[:LIMIT]
with ThreadPoolExecutor(1) as pool:
    nxt = pool.submit(load_video, todo[0]) if todo else None
    for k, v in enumerate(todo):
        frames = nxt.result(); nxt = pool.submit(load_video, todo[k + 1]) if k + 1 < len(todo) else None
        idx = np.where(meta.key.values == v)[0]; idx = idx[np.argsort(meta.start.values[idx])]
        n = min(len(frames) // 30, len(idx)); out = [clip_features(frames[c * 30:(c + 1) * 30], np.asarray(kpall[idx[c]])) for c in range(n)]
        if not LIMIT:
            np.save(OUT / "train" / (v.replace("/", "__") + ".npy"), np.concatenate(out))
        if k % 5 == 0:
            print(f"train {k + 1}/{len(todo)} | {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
