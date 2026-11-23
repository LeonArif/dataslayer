# Make contact sheets for worklist orders A..B (inclusive): python annotation/make_batch.py A B   (from leon/)
import subprocess, sys, pandas as pd
w = pd.read_csv("annotation/worklist.csv"); a, b = int(sys.argv[1]), int(sys.argv[2])
for r in w[(w.order >= a) & (w.order <= b)].itertuples():
    first = 3 if r.seconds == r.seconds and str(r.seconds) != "" and not pd.isna(r.seconds) else 0       # videos not in the clip table (no start was read) are viewed from second 0
    subprocess.run([sys.executable, "scripts/annotation_sheet.py", r.video, f"annotation/sheets/s{int(r.order):03d}.jpg", str(first)], check=True)
