"""Whole-body pose (133 keypoints incl. 21 points per hand) with RTMW (rtmlib RTMPose class, model rtmw-dw-x-l cocktail14 384x288 onnx, GPU via onnxruntime) on all 30 frames of every labelled train clip and every test sequence, using the cached RT-DETR person box (xywh in 640x360 test-format pixels) of the clip for all 30 frames.
Runs in the separate venv (scratchpad/rtmw_env: rtmlib, onnxruntime-gpu, opencv, numpy, pandas), not in the main environment. Weights are trained on COCO-WholeBody etc. (some non-commercial datasets): research use only, noted in logs.md.
Output cache/rtmw/{kp_train (6166,30,133,3) fp16 = x, y, score; kp_test (1515,30,133,3)}; train frames re-encoded as 640x360 JPEG q90 like the test format. Resumable per video. env LIMIT=n for a speed test (saves nothing final). usage (from leon/): <venv python> scripts/extract_rtmw.py
Keypoint layout (COCO-WholeBody): 0-16 body, 17-22 feet, 23-90 face, 91-111 left hand, 112-132 right hand."""
import os, time
os.add_dll_directory(r"C:/Users/leouw/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/torch/lib")   # cuDNN 9 / CUDA 12 DLLs shipped with torch, for onnxruntime-gpu 1.22 in the venv
from pathlib import Path
import cv2, numpy as np, pandas as pd
from rtmlib import RTMPose

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "cache/rtmw"; (OUT / "train").mkdir(parents=True, exist_ok=True); LIMIT = int(os.environ.get("LIMIT", 0))
URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
pose = RTMPose(onnx_model=URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda"); cv2.setNumThreads(2)


def recode(img):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)           # BGR


def clip_kp(frames, box):                                  # frames: 30 BGR (360, 640, 3), box xywh -> (30, 133, 3) fp16
    x, y, w, h = [float(v) for v in box]; bb = [[max(x, 0), max(y, 0), min(x + w, 640), min(y + h, 360)]]; out = []
    for f in frames:
        k, s = pose(f, bboxes=bb); out.append(np.concatenate([k[0], s[0][:, None]], 1))
    return np.stack(out).astype(np.float16)


clips = pd.read_csv(ROOT / "cache/members_2026-11-20/clips.csv")
meta = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_train_s30_meta.csv"); boxall = np.load(ROOT / "cache/pose_vitpose_all/box_train_s30.npy")
meta["key"] = [("cheat/" if l == 1 else "normal/") + v.replace(".MOV", "") for v, l in zip(meta.video, meta.label)]; look = {(k, s): i for i, (k, s) in enumerate(zip(meta.key, meta.start))}
t0 = time.time(); ids = list(pd.read_csv(ROOT / "data/sample_submission.csv")["id"]); tm = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_test_meta.csv"); trow = {i: k for k, i in enumerate(tm["id"])}; boxt = np.load(ROOT / "cache/pose_vitpose_all/box_test.npy")
(OUT / "test_parts").mkdir(exist_ok=True); parts = [ids[i:i + 50] for i in range(0, len(ids), 50)]
for pi, part in enumerate(parts[:(2 if LIMIT else None)]):
    f = OUT / "test_parts" / f"part_{pi:03d}.npy"
    if f.exists(): continue
    res = []
    for sid in part:
        files = sorted((ROOT / "data/test" / sid).glob("*.jpg")); assert len(files) == 30
        res.append(clip_kp([cv2.imread(str(p)) for p in files], boxt[trow[sid]]))
    np.save(f, np.stack(res)); print(f"test part {pi + 1}/{len(parts)} | {time.time() - t0:.0f}s", flush=True)
if not LIMIT and not (OUT / "kp_test.npy").exists(): np.save(OUT / "kp_test.npy", np.concatenate([np.load(OUT / "test_parts" / f"part_{i:03d}.npy") for i in range(len(parts))])); print("test saved", flush=True)
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
        i = look[(v, sec * 30)]; res.append(clip_kp(frames[sec * 30:sec * 30 + 30], boxall[i]))
    if not LIMIT: np.save(OUT / "train" / (v.replace("/", "__") + ".npy"), np.stack(res))
    if k % 10 == 0: print(f"train {k + 1}/{len(groups)} | {time.time() - t0:.0f}s", flush=True)
if not LIMIT:
    rows = {v: np.load(OUT / "train" / (v.replace("/", "__") + ".npy")) for v in clips.video.unique()}; ctr = {v: 0 for v in rows}; series = []
    for v in clips.video:
        series.append(rows[v][ctr[v]]); ctr[v] += 1
    np.save(OUT / "kp_train.npy", np.stack(series)); print("kp_train saved", np.stack(series).shape, flush=True)
print("done", flush=True)
