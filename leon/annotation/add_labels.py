# Append per-second labels (AI assistant, TRAIN frames only, FAQ (3) #83) to annotation/annotations.csv.
# usage (from leon/): python annotation/add_labels.py ORDER "3-6:0 7-8:1 9:u 10-15:0" ["note"]     (0 normal, 1 cheat, u unsure; seconds before the first viewed second are written as 0 with viewed=0)
import csv, sys, pandas as pd
w = pd.read_csv("annotation/worklist.csv"); r = w[w.order == int(sys.argv[1])].iloc[0]; video = r.video
n = int(float(r.seconds)) if str(r.seconds) not in ("", "nan") else None
spec = {}
for tok in sys.argv[2].split():
    rng, lab = tok.split(":"); a, b = (rng.split("-") + [rng])[:2] if "-" in rng else (rng, rng)
    for s in range(int(a), int(b) + 1): spec[s] = {"0": 0, "1": 1, "u": -1}[lab]
first = min(spec); last = max(spec)
if n is not None: assert last == n - 1, (video, "last second", last, "expected", n - 1)
assert sorted(spec) == list(range(first, last + 1)), "gap in the seconds"
done = set(pd.read_csv("annotation/annotations.csv")["video"])
assert video not in done, "video already labelled"
with open("annotation/annotations.csv", "a", newline="") as f:
    wr = csv.writer(f)
    for s in range(0, last + 1):
        wr.writerow([video, s, spec.get(s, 0), int(s in spec), sys.argv[3] if len(sys.argv) > 3 and s in spec else ""])
print(video, "labelled", {k: sum(v == k for v in spec.values()) for k in (0, 1, -1)})
