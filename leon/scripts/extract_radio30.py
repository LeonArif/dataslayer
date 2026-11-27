# C-RADIOv4-H (nvidia/C-RADIOv4-H, NVIDIA Open Model License, custom code read before use: no network or exec calls except optional adaptors that are NOT loaded, adaptor_names is None so no SigLIP2/text part) features for EVERY frame, same layout as extract_dino30.py:
# per frame the model summary vector (cls-like) + mean of the spatial patch features, fp16. Image only, no text, no zero-shot. Input 224x400 RGB in 0..1 (the model has its own input conditioner for mean/std).
# Output: cache/radio_h_224x400/{test.npy (1515, 30, D), train/<video>.npy (n_frames, D)}. Resumable. usage (from leon/): python scripts/extract_radio30.py [test|train|both]
import sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, torch
from transformers import AutoModel

DATA, IMG_HW, DEVICE = Path("../data"), (224, 400), "cuda"
CACHE = Path("../cache/radio_h_224x400")
(CACHE / "train").mkdir(parents=True, exist_ok=True); (CACHE / "test_parts").mkdir(exist_ok=True)
WHAT = sys.argv[1] if len(sys.argv) > 1 else "both"
cv2.setNumThreads(2)
backbone = AutoModel.from_pretrained("nvidia/C-RADIOv4-H", trust_remote_code=True).to(DEVICE).eval()


def to_input(img_bgr):
    return cv2.cvtColor(cv2.resize(img_bgr, (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def embed(frames, batch_size=32):
    out = []
    for i in range(0, len(frames), batch_size):
        x = torch.from_numpy(np.stack(frames[i:i + batch_size])).to(DEVICE).permute(0, 3, 1, 2).float().div(255)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            summary, spatial = backbone(x)
        feat = torch.cat([summary, spatial.mean(1)], dim=1).float()
        assert torch.isfinite(feat).all(), "features contain NaN/inf"
        out.append(feat.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def read_video(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(to_input(img))
    cap.release()
    return frames


def read_seq(seq):
    files = sorted((DATA / "test" / seq).glob("*.jpg"))
    assert len(files) == 30, (seq, len(files))
    return [to_input(cv2.imread(str(p))) for p in files]


def run(items, loader, saver, exists, label):
    t0, done = time.time(), 0
    todo = [it for it in items if not exists(it)]
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(loader, todo[0]) if todo else None
        for k, it in enumerate(todo):
            frames = nxt.result()
            nxt = pool.submit(loader, todo[k + 1]) if k + 1 < len(todo) else None
            saver(it, embed(frames)); done += len(frames)
            if k % 10 == 0:
                print(f"{label} {k + 1}/{len(todo)} | {done / (time.time() - t0):.1f} frames/s | {time.time() - t0:.0f}s", flush=True)


if WHAT in ("test", "both"):
    sub = pd.read_csv(DATA / "sample_submission.csv")
    ids = list(sub["id"]); parts = [ids[i:i + 50] for i in range(0, len(ids), 50)]
    pf = lambda p: CACHE / "test_parts" / f"part_{ids.index(p[0]):05d}.npy"
    run(parts, lambda p: sum([read_seq(s) for s in p], []), lambda p, f: np.save(pf(p), f.reshape(len(p), 30, -1)), lambda p: pf(p).exists(), "test")
    if not (CACHE / "test.npy").exists():
        np.save(CACHE / "test.npy", np.concatenate([np.load(pf(p)) for p in parts]))
        print("test.npy", np.load(CACHE / "test.npy", mmap_mode="r").shape, flush=True)

if WHAT in ("train", "both"):
    rows = []
    for path in sorted(DATA.glob("train/*/*/*.MOV")):
        rows.append((f"{path.parent.parent.name}/{path.parent.name}/{path.stem}", str(path)))
    out = lambda r: CACHE / "train" / (r[0].replace("/", "__") + ".npy")
    run(rows, lambda r: read_video(r[1]), lambda r, f: np.save(out(r), f), lambda r: out(r).exists(), "train")
print("done", flush=True)
