# ChainSentinel

Flags illicit Bitcoin transactions using machine learning, trained on the public
[Elliptic dataset](https://arxiv.org/abs/1908.02591) (~200k real transactions, a subset labeled
licit/illicit). Will eventually add drift monitoring to warn when a deployed model goes stale.

> Personal learning project, not production software. Built incrementally, directly in `src/`.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download the Elliptic dataset from
[Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and put
`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` in `data/`
(git-ignored). No data yet? Generate fake practice data instead:

```bash
python tests/make_fake_data.py data_fake
```

## Structure

```
src/          project code: data loading, metrics, models, (later) graph features, drift
notebooks/    exploration/visualization only
data/         raw CSVs (git-ignored)
results/      saved charts and tables
tests/        fake-data generator
```

## How it works

- **`src/data.py`** — loads the Elliptic CSVs and splits train/test **by time**, not randomly
  (a random split would leak future transactions into training).
- **`src/metrics.py`** — illicit transactions are a small minority (~2% overall, ~10% of
  labeled), so plain accuracy is misleading. Tracks illicit-class precision/recall/F1 and PR-AUC
  instead.
- **`src/model.py`** — baseline logistic regression and random forest, weighted for the class
  imbalance.

## Roadmap

- [x] Data loading + time-based split, illicit-class metrics, baseline models
- [ ] Graph-based features
- [ ] GNN models (GCN, GraphSAGE, GAT)
- [ ] Drift detection
- [ ] Explainability (SHAP) + dashboard

## Data source

Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks
for Financial Forensics* (2019). https://arxiv.org/abs/1908.02591
