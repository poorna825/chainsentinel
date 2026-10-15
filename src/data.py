"""
Loading the Elliptic Bitcoin dataset and splitting it by time.

The dataset comes as three CSV files (put them in data/):
  elliptic_txs_features.csv  - no header. col 0 = txId, col 1 = time step (1-49),
                               then 165 anonymised features (93 local + 72 aggregated)
  elliptic_txs_classes.csv   - header: txId,class   class is "1" (illicit), "2" (licit) or "unknown"
  elliptic_txs_edgelist.csv  - header: txId1,txId2  one row per money flow between transactions
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FAKE_DATA_DIR = REPO_ROOT / "data_fake"

# Standard split used in the original paper: train on early time steps, test on later ones.
TRAIN_STEPS = range(1, 35)   # 1..34
TEST_STEPS = range(35, 50)   # 35..49

LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}  # 1 = illicit, 0 = licit, -1 = unlabelled


def load_elliptic(data_dir=DATA_DIR):
    """Return (features, edges). features has txId, time_step, label, f1..f165.

    Falls back to FAKE_DATA_DIR with a warning if the real CSVs aren't present.
    Results produced from fake data are meaningless and must never be reported.
    """
    data_dir = Path(data_dir)
    if not (data_dir / "elliptic_txs_features.csv").exists():
        print(f"WARNING: no real data in {data_dir}, falling back to fake data in "
              f"{FAKE_DATA_DIR}. Results from fake data are meaningless - do not report them.")
        data_dir = FAKE_DATA_DIR

    feats = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    n_feat = feats.shape[1] - 2
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, n_feat + 1)]

    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    classes["label"] = classes["class"].astype(str).map(LABEL_MAP)

    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")

    df = feats.merge(classes[["txId", "label"]], on="txId", how="left")
    cols = ["txId", "time_step", "label"] + [c for c in df.columns if c.startswith("f")]
    return df[cols], edges


def feature_columns(df):
    return [c for c in df.columns if c.startswith("f")]


def time_split(df, labelled_only=True):
    """Split by time step, never randomly. Random splits leak the future into training."""
    if labelled_only:
        df = df[df["label"] != -1]
    train = df[df["time_step"].isin(TRAIN_STEPS)]
    test = df[df["time_step"].isin(TEST_STEPS)]
    return train, test
