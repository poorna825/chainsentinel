"""
Drift monitoring: is new data still statistically similar to what the model
was trained on? A model trained in the past can quietly get worse over time
even without retraining, if the incoming data stops resembling training data.
"""
import numpy as np

from metrics import pr_auc_over_time


def psi(train_values, new_values, buckets=10, epsilon=1e-4):
    """Population Stability Index for one feature.

    Buckets training values into deciles, then compares what fraction of
    train vs. new data falls in each bucket. Rule-of-thumb reading:
    < 0.1 no significant shift, 0.1-0.25 moderate, > 0.25 major shift.

    epsilon floors empty-bucket proportions so log() doesn't blow up. Too
    small a floor (e.g. 1e-6) lets a handful of empty buckets dominate the
    whole sum with an almost-constant contribution regardless of the real
    shift's size - found this by checking a real case (f136: 8/10 buckets
    empty contributed 80% of PSI, same ~1.15 each, drowning out the one
    bucket that actually carried the signal). 1e-4 keeps empty buckets
    informative without letting them swamp everything else.

    Many Elliptic features have a heavy point mass at one repeated value
    (e.g. a filler for transactions with limited history), which collapses
    several quantile levels onto the same edge - 126/165 features hit this.
    Duplicate edges are deduped here so every bucket has positive width and
    a real train population, instead of silently producing zero-width
    buckets that are empty by construction rather than by any real shift.
    """
    train_values = np.asarray(train_values)
    new_values = np.asarray(new_values)

    # bucket edges defined by training data's own quantiles; dedupe so a
    # heavy point mass doesn't produce zero-width, structurally-empty buckets
    edges = np.unique(np.quantile(train_values, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 2:
        return 0.0  # train feature is constant - nothing to compare buckets on
    edges[0], edges[-1] = -np.inf, np.inf  # catch values outside train's observed range

    train_counts, _ = np.histogram(train_values, bins=edges)
    new_counts, _ = np.histogram(new_values, bins=edges)

    train_props = np.maximum(train_counts / len(train_values), epsilon)
    new_props = np.maximum(new_counts / len(new_values), epsilon)

    return float(np.sum((new_props - train_props) * np.log(new_props / train_props)))


def weighted_psi(scores, importances):
    """Importance-weighted average PSI: a shift in a feature the model relies
    on heavily should move this more than an equal-sized shift in a feature
    the model barely uses - unlike median/frac, which treat all features the
    same regardless of whether the model's predictions actually depend on them.
    """
    feats = list(scores)
    psi_vals = np.array([scores[f] for f in feats])
    weights = np.array([importances[f] for f in feats])
    return float(np.sum(psi_vals * weights) / np.sum(weights))


def psi_over_time(train_df, full_df, feature_names, time_steps,
                   moderate_threshold=0.1, major_threshold=0.25, importances=None):
    """PSI per time step, each compared against the same fixed train baseline.

    Returns one row per time step with complementary views, since no single
    number tells the whole story:
    - median_psi: the typical feature's drift (robust to a few features that
      always drift by construction, e.g. time-trending aggregated features)
    - max_psi / max_psi_feature: the single worst feature that step
    - frac_moderate / frac_major: fraction of features past the standard
      moderate (0.1) / major (0.25) rule-of-thumb cutoffs - catches drift
      concentrated in a meaningful subset of features that the median alone
      would hide (e.g. 15/165 features shifting hard while the rest stay calm)
    - weighted_psi (only if importances is given): drift weighted by how much
      the model actually relies on each feature, so irrelevant-feature drift
      doesn't dilute the signal and important-feature drift isn't averaged away
    """
    rows = []
    for step in time_steps:
        step_df = full_df[full_df["time_step"] == step]
        scores = {feat: psi(train_df[feat], step_df[feat]) for feat in feature_names}
        values = np.array(list(scores.values()))
        worst_feat = max(scores, key=scores.get)
        row = {
            "time_step": step,
            "median_psi": float(np.median(values)),
            "max_psi": scores[worst_feat],
            "max_psi_feature": worst_feat,
            "frac_moderate": float(np.mean(values > moderate_threshold)),
            "frac_major": float(np.mean(values > major_threshold)),
        }
        if importances is not None:
            row["weighted_psi"] = weighted_psi(scores, importances)
        rows.append(row)
    return rows


def prediction_drift_over_time(model, train_df, full_df, feature_names, time_steps):
    """PSI of the model's predicted illicit-probability score per time step,
    against the score distribution on training data.

    Unlike feature-level drift (165 numbers to reconcile, diluted by features
    the model barely uses), this collapses everything into one number the
    model actually produces - and unlike PR-AUC, it needs no ground-truth
    labels, so it's something that could be monitored continuously in a real
    deployment where fraud labels often arrive late, if at all.
    """
    train_scores = model.predict_proba(train_df[feature_names])[:, 1]
    rows = []
    for step in time_steps:
        step_df = full_df[full_df["time_step"] == step]
        step_scores = model.predict_proba(step_df[feature_names])[:, 1]
        rows.append({"time_step": step, "prediction_psi": psi(train_scores, step_scores)})
    return rows


def rolling_psi_over_time(model, full_df, feature_names, time_steps, importances, window=3):
    """Like psi_over_time/prediction_drift_over_time, but each step t is compared
    against the previous `window` steps (t-window..t-1) instead of the fixed
    training set. A fixed baseline sits above the "major" cutoff at every step
    because aggregated features trend with time, so it can't tell gradual
    trend from a sudden break; a rolling baseline absorbs the trend and should
    only move when the data changes faster than the window can follow.

    Baselines use all rows of those steps (labelled and unknown), same as the
    fixed-baseline metrics. Returns one row per time step with median_psi,
    weighted_psi and prediction_psi.
    """
    step_dfs = {s: full_df[full_df["time_step"] == s] for s in
                range(min(time_steps) - window, max(time_steps) + 1)}
    step_scores = {s: model.predict_proba(d[feature_names])[:, 1] for s, d in step_dfs.items()}

    rows = []
    for step in time_steps:
        base_steps = range(step - window, step)
        base_df = np.concatenate([step_dfs[s][feature_names].values for s in base_steps])
        base_idx = {f: i for i, f in enumerate(feature_names)}
        new_df = step_dfs[step]
        scores = {f: psi(base_df[:, base_idx[f]], new_df[f]) for f in feature_names}
        base_scores = np.concatenate([step_scores[s] for s in base_steps])
        rows.append({
            "time_step": step,
            "median_psi": float(np.median(list(scores.values()))),
            "weighted_psi": weighted_psi(scores, importances),
            "prediction_psi": psi(base_scores, step_scores[step]),
        })
    return rows


def monitoring_report(model, train_df, full_df, feature_names, time_steps, importances=None):
    """One row per time step combining every drift signal plus real performance.

    Not an automated stale/not-stale verdict - we checked in steps 10-12 and
    none of these signals reliably predicts performance on their own at this
    sample size. This is the honest alternative: lay out everything a human
    reviewing the model would want side by side (feature drift, prediction
    drift, and - when labels exist - actual PR-AUC with its n_illicit, so a
    thin sample isn't mistaken for a confirmed swing).
    """
    drift_rows = {r["time_step"]: r for r in
                  psi_over_time(train_df, full_df, feature_names, time_steps, importances=importances)}
    pred_rows = {r["time_step"]: r for r in
                 prediction_drift_over_time(model, train_df, full_df, feature_names, time_steps)}
    perf_rows = {r["time_step"]: r for r in pr_auc_over_time(model, full_df, feature_names, time_steps)}

    rows = []
    for step in time_steps:
        row = {**drift_rows[step], **pred_rows[step], **perf_rows[step]}
        rows.append(row)
    return rows
