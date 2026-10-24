"""
Evaluation metrics for the illicit class.

Raw accuracy is misleading here: illicit transactions are a small minority
(~2% of all transactions, ~10% of labelled ones), so a model that always
predicts "licit" scores high accuracy while catching no fraud at all.
Use illicit-class precision/recall/F1 instead.
"""
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
)


def evaluate(y_true, y_pred):
    """Precision/recall/F1 for the illicit class (label 1), plus accuracy for comparison."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "illicit_precision": precision,
        "illicit_recall": recall,
        "illicit_f1": f1,
    }


def pr_auc(y_true, y_score):
    """Precision-recall AUC for the illicit class, summarizing all thresholds at once.

    y_score is the predicted probability of being illicit (not a 0/1 prediction).
    """
    return average_precision_score(y_true, y_score, pos_label=1)


def threshold_table(y_true, y_score, thresholds=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Precision/recall at a handful of thresholds, to see the tradeoff directly."""
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0
        )
        rows.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})
    return rows


def bootstrap_pr_auc_diff(y_true, score_a, score_b, n_boot=1000, random_state=0):
    """Paired bootstrap on a fixed pair of models' test-set scores: how much
    does score_b's PR-AUC exceed score_a's, and how much does that difference
    swing just from which test rows happen to get resampled?

    Paired = the same resampled row indices are used for both scores each
    draw, so the comparison reflects the two models' predictions on the same
    simulated test set each time, not independent sampling noise.

    If a resample happens to contain no positive rows, pr_auc() (sklearn's
    average_precision_score) emits a UserWarning and returns 0.0 for that
    draw rather than crashing or returning NaN - not an issue in practice
    here since the real test set has thousands of positives, but noted in
    case this is reused on a much smaller/sparser set.
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = pr_auc(y_true[idx], score_b[idx]) - pr_auc(y_true[idx], score_a[idx])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
    }


def pr_auc_over_time(model, full_df, feature_names, time_steps):
    """PR-AUC per time step, to check whether the model's real performance actually
    degrades at the same points drift metrics flag - not just whether drift exists.

    Only labelled rows count (no ground truth for unknowns), so some steps may
    have too few illicit examples for a stable PR-AUC - n_illicit is reported
    alongside so a thin step isn't mistaken for a real swing.
    """
    rows = []
    for step in time_steps:
        step_df = full_df[(full_df["time_step"] == step) & (full_df["label"] != -1)]
        y_true = step_df["label"]
        y_score = model.predict_proba(step_df[feature_names])[:, 1]
        rows.append({
            "time_step": step,
            "pr_auc": pr_auc(y_true, y_score),
            "n_illicit": int((y_true == 1).sum()),
            "n_labelled": len(step_df),
        })
    return rows
