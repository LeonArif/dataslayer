# Contact sheet of TRAIN videos for per-second labelling by the AI assistant (FAQ (3) #83: annotation with a VLM / AI assistant is allowed on TRAIN data only; never use test frames here).
# One cell per second: 3 frames (offsets 4, 14, 24 of that second) cropped around the person (median person box of the video from cache/pose_vitpose_all/box_train_s30.npy),
# seconds from FIRST (default 3) to the end, two columns of seconds side by side. The second number is written on the left of every cell.
# usage (from leon/): python scripts/annotation_sheet.py "cheat/subject_8/front_Repeated_Peeking" out.jpg [FIRST]
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
video, out = sys.argv[1], Path(sys.argv[2]); FIRST = int(sys.argv[3]) if len(sys.argv) > 3 else 3
assert video.startswith(("cheat/", "normal/")), "train videos only"
OFFS, TW, TH = [4, 14, 24], 168, 190
meta = pd.read_csv(ROOT / "cache/pose_vitpose_all/pose_train_s30_meta.csv"); boxes = np.load(ROOT / "cache/pose_vitpose_all/box_train_s30.npy")
key = video.split("/", 1)[1] + ".MOV"; sel = np.where(meta["video"].values == key)[0]
bx, by, bw, bh = np.median(boxes[sel], axis=0)                                  # (x, y, w, h) in 640x360 coordinates
cap = cv2.VideoCapture(str(ROOT / "data/train" / (video + ".MOV"))); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); secs = n // 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); sc = W / 640
ch = min(H, max(bh * 1.12, bw * 1.12 / (TW / TH)) * sc); cw = ch * TW / TH
cx, cy = (bx + bw / 2) * sc, (by + bh / 2) * sc
x0 = int(np.clip(cx - cw / 2, 0, W - cw)); y0 = int(np.clip(cy - ch / 2, 0, H - ch)); x1, y1 = int(x0 + cw), int(y0 + ch)
show = list(range(FIRST, secs)); rows = (len(show) + 1) // 2; LAB = 34; BLOCK = LAB + len(OFFS) * TW
sheet = np.full((rows * TH, 2 * BLOCK, 3), 255, np.uint8)
need = {s * 30 + o: (s, k) for s in show for k, o in enumerate(OFFS)}
i = 0
while cap.grab():
    if i in need:
        ok, img = cap.retrieve()
        if ok:
            s, k = need[i]; j = show.index(s); r, c = j % rows, j // rows
            sheet[r * TH:(r + 1) * TH, c * BLOCK + LAB + k * TW:c * BLOCK + LAB + (k + 1) * TW] = cv2.resize(img[y0:y1, x0:x1], (TW, TH), interpolation=cv2.INTER_AREA)
    i += 1
for j, s in enumerate(show):
    r, c = j % rows, j // rows
    cv2.putText(sheet, str(s), (c * BLOCK + 2, r * TH + TH // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.line(sheet, (c * BLOCK, r * TH), (c * BLOCK + BLOCK, r * TH), (0, 0, 255), 1)
cv2.line(sheet, (BLOCK, 0), (BLOCK, sheet.shape[0]), (255, 0, 0), 2)
cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82]); print(out.name, "seconds", FIRST, "to", secs - 1, sheet.shape)
