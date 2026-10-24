"""
Tests for src/metrics.py, using small hand-built arrays so every expected
result can be verified by hand.
"""
import numpy as np
import pytest

from metrics import evaluate, pr_auc, bootstrap_pr_auc_diff


# ---------- pr_auc ----------

def test_pr_auc_perfect_scores_is_one():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]  # perfectly separates the classes
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_pr_auc_constant_scores_equals_positive_rate():
    # all rows get the same score -> ranking is arbitrary/tied, so average
    # precision collapses to the positive rate (2/5 here)
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert pr_auc(y_true, y_score) == pytest.approx(2 / 5)


# ---------- evaluate (illicit-class precision/recall/F1) ----------

def test_evaluate_illicit_metrics_hand_computed():
    # 5 rows: TP=1 (idx2), FP=1 (idx3), FN=1 (idx4), rest TN
    y_true = [0, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 0]
    result = evaluate(y_true, y_pred)
    # illicit (class 1): TP=1, FP=1, FN=1 -> precision=0.5, recall=0.5, f1=0.5
    assert result["illicit_precision"] == pytest.approx(0.5)
    assert result["illicit_recall"] == pytest.approx(0.5)
    assert result["illicit_f1"] == pytest.approx(0.5)
    assert result["accuracy"] == pytest.approx(3 / 5)


def test_evaluate_is_not_computing_for_class_0():
    # class 0 (TN=2,FP=1 -> precision=2/3, recall=2/3) differs from class 1
    # (precision=recall=0.5 above) - if evaluate() accidentally used
    # pos_label=0, these numbers would come back instead.
    y_true = [0, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 0]
    result = evaluate(y_true, y_pred)
    assert result["illicit_precision"] != pytest.approx(2 / 3)
    assert result["illicit_recall"] != pytest.approx(2 / 3)


def test_evaluate_catches_pos_label_flipped_to_0():
    # asymmetric case: class-1 precision/recall and class-0 precision/recall
    # are different numbers here (not just mirrored), so a pos_label=0 break
    # can't accidentally produce the same values this test checks for.
    # TP=1 (idx0), FN=2 (idx1,idx2), TN=1 (idx3), FP=0
    y_true = [1, 1, 1, 0]
    y_pred = [1, 0, 0, 0]
    result = evaluate(y_true, y_pred)
    assert result["illicit_precision"] == pytest.approx(1.0)      # 1 / (1+0)
    assert result["illicit_recall"] == pytest.approx(1 / 3)       # 1 / (1+2)


# ---------- bootstrap_pr_auc_diff ----------

def make_bootstrap_inputs():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=50)
    score_a = rng.random(50)
    return y_true, score_a


def test_bootstrap_pr_auc_diff_identical_scores_gives_zero():
    y_true, score = make_bootstrap_inputs()
    result = bootstrap_pr_auc_diff(y_true, score, score, n_boot=200, random_state=0)
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["ci_95"] == pytest.approx((0.0, 0.0))


def test_bootstrap_pr_auc_diff_same_random_state_is_deterministic():
    y_true, score_a = make_bootstrap_inputs()
    rng = np.random.default_rng(7)
    score_b = rng.random(50)
    result1 = bootstrap_pr_auc_diff(y_true, score_a, score_b, n_boot=200, random_state=0)
    result2 = bootstrap_pr_auc_diff(y_true, score_a, score_b, n_boot=200, random_state=0)
    assert result1 == result2


def test_bootstrap_pr_auc_diff_clearly_better_score_b_is_positive_ci():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 5)
    # score_a: no separation (constant); score_b: perfectly separates
    score_a = np.full(50, 0.5)
    score_b = np.where(y_true == 1, 0.9, 0.1)
    result = bootstrap_pr_auc_diff(y_true, score_a, score_b, n_boot=200, random_state=0)
    assert result["mean_diff"] > 0
    lo, hi = result["ci_95"]
    assert lo > 0, f"expected CI fully above 0, got ({lo}, {hi})"


def test_bootstrap_pr_auc_diff_resample_with_no_positives_current_behavior():
    """Documents current behavior, not a requirement: with very few positives,
    a bootstrap resample can by chance contain zero positive rows. pr_auc()
    (sklearn's average_precision_score) does not crash or return NaN in that
    case - it emits a UserWarning and returns 0.0 for that draw. So a resample
    with no positives silently contributes a diff of 0.0 (if both scores
    give 0.0) rather than being flagged as uninformative. This test pins that
    behavior down; it is not asserting it is correct.
    """
    y_true = np.array([0] * 19 + [1])  # 1 positive out of 20 - easy to miss on resample
    score_a = np.linspace(0, 1, 20)
    score_b = np.linspace(0, 1, 20)
    with pytest.warns(UserWarning, match="No positive"):
        result = bootstrap_pr_auc_diff(y_true, score_a, score_b, n_boot=50, random_state=0)
    # no crash, no NaN
    assert not np.isnan(result["mean_diff"])
    assert not any(np.isnan(x) for x in result["ci_95"])
