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


def add_degree_features(df, edges):
    """Add in_deg/out_deg/total_deg from the directed transaction graph.

    edges is txId1 -> txId2 (money flow direction), so out_deg counts a
    transaction's appearances as txId1, in_deg its appearances as txId2.
    Named without an "f" prefix so feature_columns() keeps ignoring them -
    callers that want these must ask for them explicitly.
    """
    df = df.copy()
    out_deg = edges["txId1"].value_counts()
    in_deg = edges["txId2"].value_counts()
    df["out_deg"] = df["txId"].map(out_deg).fillna(0).astype(int)
    df["in_deg"] = df["txId"].map(in_deg).fillna(0).astype(int)
    df["total_deg"] = df["out_deg"] + df["in_deg"]
    return df


def add_neighbor_pred_features(df, edges, pred_col):
    """Add neighbor_mean_pred/neighbor_max_pred from a stage-1 prediction
    column already present on df (collective classification: use a first
    model's predictions on a node's graph neighbors as features for a second
    model on that node).

    Neighbors are undirected (a node's neighbors are everyone it shares an
    edge with, regardless of edge direction) and exclude self-loops. Nodes
    with no edges get 0/0 rather than NaN. No has_neighbors flag: every
    transaction in this dataset has at least one edge (checked directly),
    so that flag would be a constant column with nothing to split on.
    """
    pred = df.set_index("txId")[pred_col]
    nbrs = pd.concat([
        edges.rename(columns={"txId1": "txId", "txId2": "neighbor"})[["txId", "neighbor"]],
        edges.rename(columns={"txId2": "txId", "txId1": "neighbor"})[["txId", "neighbor"]],
    ])
    nbrs = nbrs[nbrs["txId"] != nbrs["neighbor"]]
    nbrs["neighbor_pred"] = nbrs["neighbor"].map(pred)
    agg = nbrs.groupby("txId")["neighbor_pred"].agg(["mean", "max"])

    df = df.copy()
    df["neighbor_mean_pred"] = df["txId"].map(agg["mean"]).fillna(0)
    df["neighbor_max_pred"] = df["txId"].map(agg["max"]).fillna(0)
    return df


def time_split(df, labelled_only=True):
    """Split by time step, never randomly. Random splits leak the future into training."""
    if labelled_only:
        df = df[df["label"] != -1]
    train = df[df["time_step"].isin(TRAIN_STEPS)]
    test = df[df["time_step"].isin(TEST_STEPS)]
    return train, test
