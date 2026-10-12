"""
Makes a tiny fake dataset in the exact Elliptic file format.
Use it to check the code runs before (or without) downloading the real data:
    python tests/make_fake_data.py data_fake
The numbers are random - never report results from it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(out_dir, n=3000, n_feat=165, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tx = np.arange(100000, 100000 + n)
    steps = np.sort(rng.integers(1, 50, n))
    X = rng.normal(size=(n, n_feat))
    cls = rng.choice(["1", "2", "unknown"], size=n, p=[0.02, 0.21, 0.77])
    X[cls == "1", :5] += 1.5  # make illicit slightly separable

    pd.DataFrame(np.column_stack([tx, steps, X])).to_csv(
        out / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame({"txId": tx, "class": cls}).to_csv(out / "elliptic_txs_classes.csv", index=False)
    src = rng.choice(tx, size=int(n * 1.15))
    dst = rng.choice(tx, size=int(n * 1.15))
    pd.DataFrame({"txId1": src, "txId2": dst}).to_csv(out / "elliptic_txs_edgelist.csv", index=False)
    return out


if __name__ == "__main__":
    print("wrote", make(sys.argv[1] if len(sys.argv) > 1 else "data_fake"))
