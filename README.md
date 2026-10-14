# ChainSentinel

ChainSentinel flags illicit Bitcoin transactions using machine learning. Bitcoin transactions form
a graph — money flows from one transaction to the next — and some of those transactions are known
to be tied to illicit activity (scams, dark markets, ransomware). This project trains models to
tell illicit transactions apart from legitimate ones, and will eventually watch for **drift** —
early warning signs that a deployed model's predictions are becoming unreliable as transaction
patterns shift over time.

It's built on the [Elliptic dataset](#data-source), a public dataset of ~200k real Bitcoin
transactions where a subset are labeled licit or illicit.

> This is a personal learning project, not production software. It's built incrementally,
> directly in `src/`, as a series of small working pieces rather than all at once.

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

## Structure

```
data/         raw CSVs (git-ignored, ~700 MB)
src/          the actual project code: data loading, metrics, models, (later) graph features, drift
notebooks/    exploration and visualization (not where the core logic lives)
notes/        notes in my own words
solutions/    answers from an earlier version of this project that used notebook exercises
app/          Streamlit dashboard (later, once the pipeline exists)
results/      saved charts and tables
tests/        fake-data generator and tests
```

## How it works, so far

- **Loading & splitting** (`src/data.py`) — reads the raw Elliptic CSVs into one table, and splits
  it into train/test **by time** (earlier transactions train the model, later ones test it) rather
  than randomly. A random split would let the model "see the future," which isn't realistic.
- **Metrics** (`src/metrics.py`) — illicit transactions are a small minority (~2% of all
  transactions, ~10% of labeled ones), so plain accuracy is misleading — a model that always
  guesses "licit" would score high and catch zero fraud. Instead this project tracks precision,
  recall, and F1 for the illicit class specifically, plus PR-AUC.
- **Models** (`src/model.py`) — baseline classifiers (logistic regression, random forest) trained
  on the transaction features, weighted to account for the class imbalance.

## Roadmap

- [x] Data loading + time-based train/test split
- [x] Illicit-class metrics (precision/recall/F1, PR-AUC)
- [x] Baseline models (logistic regression, random forest)
- [ ] Graph-based features (using the transaction network, not just per-transaction stats)
- [ ] Graph neural network models (GCN, GraphSAGE, GAT)
- [ ] Drift detection — alerting when the model's predictions start to look unreliable
- [ ] Explainability (SHAP)
- [ ] Dashboard for exploring results

## Data source

Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* (2019). https://arxiv.org/abs/1908.02591
