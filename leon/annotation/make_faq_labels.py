# FAQ-continuity repair of the per-second labels of the cheat FRONT videos (annotation/annotations.csv -> annotation/annotations_faq.csv). The original file is not touched.
# Why: the first pass labelled a second cheat only when the cheating behaviour was visibly happening in it and left pauses and preparation as normal or unsure. The FAQ answers say:
#  - label = context and dominant behaviour in the 30 frames; preparation movements and pauses are NOT cheat by themselves, but they DO count when they are clearly part of the sequence of cheating actions (#88.1 to #88.3, hands under the desk are not cheat alone);
#  - under desk counts from the moment the head bends down; reaching into a pocket or bag is normal, using the notes is cheat (earlier answers).
# Rules (per video, seconds in order; C cheat, . normal, ? unsure; first / last = first / last cheat second of the first pass):
#  1. every unsure second inside the sequence (between the first and the last cheat second) becomes cheat;
#  2. every normal run of at most 2 seconds inside the sequence becomes cheat (a 1 to 2 s pause between two peeks, #88.1); longer normal runs stay normal;
#  3. the run of unsure seconds directly before the first cheat second becomes cheat (preparation that is part of the sequence: hand moving to the watch #88.2, head bending down);
#  4. the run of unsure seconds directly after the last cheat second becomes cheat (the sequence has not ended: notes still held, head still coming back up #88.3).
# Normal seconds outside the sequence and unsure seconds that touch no cheat second stay as they are. The rules are mechanical, no new frames were looked at; they only apply the FAQ wording to the labels already made.
import re, pandas as pd
a = pd.read_csv("annotation/annotations.csv"); sym = {1: "C", 0: ".", -1: "?"}; inv = {"C": 1, ".": 0, "?": -1}; out = []
for v, g in a.groupby("video", sort=False):
    g = g.sort_values("second"); s = list("".join(sym[x] for x in g.label)); seq = "".join(s); f, l = seq.find("C"), seq.rfind("C")
    if f >= 0:
        for m in re.finditer(r"[.?]+", seq[f:l + 1]):
            a0, b0 = f + m.start(), f + m.end(); run = seq[a0:b0]
            for i in range(a0, b0):
                if s[i] == "?" or len(run) <= 2: s[i] = "C"
        i = f - 1
        while i >= 0 and seq[i] == "?": s[i] = "C"; i -= 1
        i = l + 1
        while i < len(seq) and seq[i] == "?": s[i] = "C"; i += 1
    g = g.copy(); g["label_orig"] = g.label; g["label"] = [inv[c] for c in s]; out.append(g)
b = pd.concat(out); b.to_csv("annotation/annotations_faq.csv", index=False)
print("seconds changed:", int((b.label != b.label_orig).sum()), "of", len(b), "| first pass: cheat", int((b.label_orig == 1).sum()), "normal", int((b.label_orig == 0).sum()), "unsure", int((b.label_orig == -1).sum()),
      "| FAQ repair: cheat", int((b.label == 1).sum()), "normal", int((b.label == 0).sum()), "unsure", int((b.label == -1).sum()))
b["act"] = b.video.str.extract(r"front_(.*)$")
print(b.groupby("act").apply(lambda d: pd.Series({"cheat first pass": int((d.label_orig == 1).sum()), "cheat repaired": int((d.label == 1).sum()), "unsure left": int((d.label == -1).sum())}), include_groups=False).to_string())
