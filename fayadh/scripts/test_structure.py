"""Rekonstruksi video test dari sequence 1 detik (scratch, kode yang sama dipakai di main.ipynb)."""
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # folder fayadh/
DATA_DIR = os.path.join(ROOT, "..", "data")
CACHE_DIR = os.path.join(ROOT, "cache", "main")


def edge_frames(ids, k, size=(96, 54)):
    return np.stack([
        cv2.resize(cv2.imread(f"{DATA_DIR}/test/{i}/{k:02d}.jpg"), size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        for i in ids
    ])


def build_chains(ids, max_link=2.0):
    first, last = edge_frames(ids, 1), edge_frames(ids, 30)
    d = pairwise_distances(last, first, metric="l1") / last.shape[1]
    np.fill_diagonal(d, np.inf)
    succ, pred = d.argmin(1), d.argmin(0)
    n = len(ids)
    nxt = {i: succ[i] for i in range(n) if pred[succ[i]] == i and d[i, succ[i]] < max_link}
    has_prev = set(nxt.values())
    chains = []
    for i in range(n):
        if i in has_prev:
            continue
        c = [i]
        while c[-1] in nxt and len(c) <= n:
            c.append(nxt[c[-1]])
        chains.append(c)
    rows = [(ids[i], ci, pos) for ci, c in enumerate(chains) for pos, i in enumerate(c)]
    return pd.DataFrame(rows, columns=["id", "chain", "pos"]), d


if __name__ == "__main__":
    ids = sorted(os.listdir(f"{DATA_DIR}/test"))
    chains, d = build_chains(ids)
    print(chains["chain"].nunique(), chains.groupby("chain").size().value_counts().sort_index().to_dict())
    os.makedirs(CACHE_DIR, exist_ok=True)
    chains.to_csv(f"{CACHE_DIR}/test_chains.csv", index=False)
