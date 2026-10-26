"""
Headline run: train the random forest on time steps 1-34, then lay out drift
signals next to real PR-AUC for every later time step (35-49).

Run from the repo root:  python src/report.py
Writes results/monitoring_report.csv.
"""
import pandas as pd

from data import REPO_ROOT, TEST_STEPS, feature_columns, load_elliptic, time_split
from drift import monitoring_report
from model import train_random_forest

RESULTS_DIR = REPO_ROOT / "results"


def main():
    df, _ = load_elliptic()
    feats = feature_columns(df)
    train, _ = time_split(df)

    model = train_random_forest(train, feats)
    importances = dict(zip(feats, model.feature_importances_))

    rows = monitoring_report(model, train, df, feats, list(TEST_STEPS), importances=importances)
    report = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    report.to_csv(RESULTS_DIR / "monitoring_report.csv", index=False)

    cols = ["time_step", "pr_auc", "n_illicit", "median_psi", "weighted_psi", "prediction_psi"]
    print(report[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
