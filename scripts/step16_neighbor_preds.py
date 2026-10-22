"""
Step 16: does a leakage-safe collective-classification feature (a node's
graph neighbors' stage-1 predicted illicit probability) help beyond the
165 anonymized features? Companion to step 15's degree-feature check.

Two scenarios, because the obvious implementation leaks:
  A. stage-1 trained on all 165 features, compared against 165+neighbor_preds.
  B. stage-1 trained on the 94 LOCAL features only, compared against
     94 local+neighbor_preds. This is the one that actually isolates
     whether neighbor predictions add anything beyond the 94 local features
     themselves - if stage-1 saw the 72 aggregated features (f95-f165),
     the neighbor predictions would smuggle that aggregate information back
     in, making a "94 local only" comparison meaningless.

Train-window nodes get out-of-fold stage-1 predictions (5-fold); every other
node (unlabelled train-window rows, all test-window rows) gets scored by a
stage-1 model fit on the full labelled train set - see oof_predictions() and
train_random_forest() in src/model.py for why this avoids a row leaking its
own label into its own neighbor-derived feature.

Run: .venv/Scripts/python.exe scripts/step16_neighbor_preds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from data import load_elliptic, time_split, feature_columns, add_neighbor_pred_features
from model import train_random_forest, oof_predictions
from metrics import pr_auc, bootstrap_pr_auc_diff

df, edges = load_elliptic()
features_165 = feature_columns(df)
local94 = [f"f{i}" for i in range(1, 95)]
nb_feats = ["neighbor_mean_pred", "neighbor_max_pred"]

train, test = time_split(df, labelled_only=True)


def build_neighbor_features(stage1_features):
    oof = oof_predictions(train, stage1_features, n_folds=5, random_state=0)
    stage1_model = train_random_forest(train, feature_names=stage1_features, random_state=0)
    d = df.copy()
    d["stage1_pred"] = np.nan
    d.loc[train.index, "stage1_pred"] = oof
    other_idx = d.index.difference(train.index)
    d.loc[other_idx, "stage1_pred"] = stage1_model.predict_proba(
        d.loc[other_idx, stage1_features])[:, 1]
    d = add_neighbor_pred_features(d, edges, "stage1_pred")
    return d


def averaged_scores(train_df, test_df, feature_names, n_seeds=5):
    """Average predicted probability across n_seeds independently-trained
    random forests. A single-seed fit entangles "does this feature help"
    with "did this particular seed's random feature subsampling happen to
    like it" - averaging over seeds before bootstrapping isolates test-set
    sampling variance (what the bootstrap actually measures) from RF-fit
    variance (which a single-seed comparison can't distinguish from a real
    effect).
    """
    scores = np.zeros(len(test_df))
    for seed in range(n_seeds):
        model = train_random_forest(train_df, feature_names=feature_names, random_state=seed)
        scores += model.predict_proba(test_df[feature_names])[:, 1]
    return scores / n_seeds


def permutation_importance_for(model, X_test, y_test, feats_to_check, n_repeats=10, random_state=0):
    """Hand-rolled, scoped to only the columns we care about - sklearn's
    permutation_importance shuffles every column in X_test by default, which
    wastes most of its cost re-scoring 165/94 base features we already trust.
    """
    rng = np.random.default_rng(random_state)
    baseline = pr_auc(y_test, model.predict_proba(X_test)[:, 1])
    results = {}
    for feat in feats_to_check:
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            X_shuffled = X_test.copy()
            X_shuffled[feat] = rng.permutation(X_shuffled[feat].values)
            score = pr_auc(y_test, model.predict_proba(X_shuffled)[:, 1])
            drops[r] = baseline - score
        results[feat] = {"mean": float(drops.mean()), "std": float(drops.std())}
    return baseline, results


N_SEEDS = 5

print("=== Scenario A: stage-1 on 165 feats (165 vs 165+neighbor_preds) ===")
df_a = build_neighbor_features(features_165)
train_a, test_a = time_split(df_a, labelled_only=True)
y_test_a = test_a["label"]
score_a1 = averaged_scores(train_a, test_a, features_165, n_seeds=N_SEEDS)
score_a2 = averaged_scores(train_a, test_a, features_165 + nb_feats, n_seeds=N_SEEDS)
print(f"165 feats PR-AUC (avg of {N_SEEDS} seeds): {pr_auc(y_test_a, score_a1):.4f}")
print(f"165+neighbor_preds PR-AUC (avg of {N_SEEDS} seeds): {pr_auc(y_test_a, score_a2):.4f}")
boot_a = bootstrap_pr_auc_diff(y_test_a, score_a1, score_a2, n_boot=1000, random_state=0)
print(f'bootstrap mean diff (on seed-averaged scores): {boot_a["mean_diff"]:.4f}  95% CI: {boot_a["ci_95"]}')

# permutation importance uses one representative fit (seed 0) from the +neighbor_preds setup
model_a2 = train_random_forest(train_a, feature_names=features_165 + nb_feats, random_state=0)
baseline_a, perm_a = permutation_importance_for(
    model_a2, test_a[features_165 + nb_feats], y_test_a, nb_feats, n_repeats=10, random_state=0)
print(f"permutation importance (seed-0 model, test set, baseline PR-AUC={baseline_a:.4f}):")
for feat, r in perm_a.items():
    print(f'  {feat}: mean drop={r["mean"]:.5f}  std={r["std"]:.5f}')

print()
print("=== Scenario B: stage-1 on 94 local feats ONLY (isolates the real question) ===")
df_b = build_neighbor_features(local94)
train_b, test_b = time_split(df_b, labelled_only=True)
y_test_b = test_b["label"]
score_b1 = averaged_scores(train_b, test_b, local94, n_seeds=N_SEEDS)
score_b2 = averaged_scores(train_b, test_b, local94 + nb_feats, n_seeds=N_SEEDS)
print(f"94 local PR-AUC (avg of {N_SEEDS} seeds): {pr_auc(y_test_b, score_b1):.4f}")
print(f"94 local+neighbor_preds (local stage-1) PR-AUC (avg of {N_SEEDS} seeds): {pr_auc(y_test_b, score_b2):.4f}")
boot_b = bootstrap_pr_auc_diff(y_test_b, score_b1, score_b2, n_boot=1000, random_state=0)
print(f'bootstrap mean diff (on seed-averaged scores): {boot_b["mean_diff"]:.4f}  95% CI: {boot_b["ci_95"]}')

model_b2 = train_random_forest(train_b, feature_names=local94 + nb_feats, random_state=0)
baseline_b, perm_b = permutation_importance_for(
    model_b2, test_b[local94 + nb_feats], y_test_b, nb_feats, n_repeats=10, random_state=0)
print(f"permutation importance (seed-0 model, test set, baseline PR-AUC={baseline_b:.4f}):")
for feat, r in perm_b.items():
    print(f'  {feat}: mean drop={r["mean"]:.5f}  std={r["std"]:.5f}')
