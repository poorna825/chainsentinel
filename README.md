# ChainSentinel

Graph-based fraud detection on the Elliptic Bitcoin dataset, with drift monitoring that warns when the model starts going stale.

> Work in progress — a personal learning project, built incrementally and directly in `src/`.

## Setup (once)

```bash
git clone <your-repo-url> && cd chainsentinel
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

1. Download the Elliptic dataset from Kaggle: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Unzip and put these three files in `data/`:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_classes.csv`
   - `elliptic_txs_edgelist.csv`

`data/` is git-ignored, so the ~700 MB of CSVs never get pushed.

No data yet? Make fake practice data in the same format to check that everything runs:

```bash
python tests/make_fake_data.py data_fake
```

## Run

The core logic lives in `src/` and is built up step by step (data loading, metrics, models, and
later graph features, GNNs, and drift monitoring). Notebooks under `notebooks/` are used for
exploration and visualization, not as the primary deliverable.

## Structure

```
data/         raw CSVs (git-ignored)
notebooks/    exploration/visualization only
notes/        notes in my own words
solutions/    legacy exercise solutions from an earlier notebook-curriculum approach
src/          the real project code: data loading, metrics, models, (later) features, drift
app/          Streamlit dashboard (later, once the pipeline exists)
results/      saved charts and tables
tests/        fake-data generator and tests
```

## Progress

- [x] Data loading + time-based train/test split
- [x] Illicit-class metrics (precision/recall/F1, PR-AUC)
- [x] Baseline models (logistic regression, random forest)
- [ ] Graph features
- [ ] First GNN (GCN)
- [ ] GraphSAGE, GAT, hybrid models
- [ ] Drift detection
- [ ] Explainability (SHAP)
- [ ] Dashboard
- [ ] Polish

## Data source

Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* (2019). https://arxiv.org/abs/1908.02591
