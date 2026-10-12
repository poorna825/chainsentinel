# Fraud Graph Drift

Graph-based fraud detection on the Elliptic Bitcoin dataset, with drift monitoring that warns when the model starts going stale.

> Work in progress — currently in **Week 1** of a 10-week learning build.

## Setup (once)

```bash
git clone <your-repo-url> && cd fraud-graph-drift
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

```bash
jupyter notebook notebooks/01_foundations.ipynb
```

## Structure

```
data/         raw CSVs (git-ignored)
notebooks/    one notebook per week
notes/        weekly notes in my own words
solutions/    exercise solutions (try first!)
src/          reusable code: data loading, features, models, drift
app/          Streamlit dashboard (Week 9)
results/      saved charts and tables
tests/        fake-data generator and tests
```

## Progress

- [ ] Week 1 — Foundations: metrics, imbalance, time split
- [ ] Week 2 — EDA
- [ ] Week 3 — Classic ML baselines
- [ ] Week 4 — Graph features
- [ ] Week 5 — First GNN (GCN)
- [ ] Week 6 — GraphSAGE, GAT, hybrid
- [ ] Week 7 — Drift detection
- [ ] Week 8 — Explainability (SHAP)
- [ ] Week 9 — Dashboard
- [ ] Week 10 — Polish

## Data source

Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* (2019). https://arxiv.org/abs/1908.02591
