# Graph features: results (steps 15-16)

Two attempts to add real transaction-graph structure on top of the 165 anonymized Elliptic
features. Both came back negative - a useful result in its own right, not just a dead end, since
it tells us where this model's predictive power actually comes from.

## Step 15: directed degree (`src/data.py::add_degree_features`)

`in_deg`/`out_deg`/`total_deg`, computed from the directed edgelist (`txId1 -> txId2`). Compared
4 feature sets over 5 random forest seeds, pooled test set (steps 35-49):

| setup | mean PR-AUC | std |
|---|---|---|
| 165 feats | 0.798 | 0.001 |
| 165+deg | 0.797 | 0.002 |
| 94 local | 0.788 | 0.001 |
| 94 local+deg | 0.788 | 0.001 |

No gain in either condition - with or without the aggregated features (`f95`-`f165`) present. The
degree features ranked near the bottom of feature importance in both setups (e.g. rank 141-151 of
168). **Conclusion: raw degree is uninformative for this task**, not merely redundant with
something else - a node's simple connection count doesn't distinguish illicit from licit
transactions here.

## Step 16: leakage-safe neighbor predictions (`scripts/step16_neighbor_preds.py`)

Tried next: collective classification. A stage-1 random forest produces a predicted illicit
probability per transaction (out-of-fold for train rows, so a row never sees its own label
reflected back at it); `neighbor_mean_pred`/`neighbor_max_pred`
(`src/data.py::add_neighbor_pred_features`) aggregate a node's *neighbors'* stage-1 predictions as
features for a second-stage model. (A third candidate, `has_neighbors`, was dropped - every
transaction in this dataset has at least one edge, so it's a constant column with nothing to
split on.)

A same-time-step check first ruled out a simpler "neighbor illicit rate" feature: all 234,355
edges connect transactions within the same time step, so a test-window node's neighbors are
always other test-window nodes with no legitimately "known" label at prediction time - using
their true labels would leak the future.

Two scenarios, because the first version of this experiment leaked:

- **Scenario A** - stage-1 trained on all 165 features, compared against `165+neighbor_preds`.
- **Scenario B** - stage-1 trained on the 94 *local* features only, compared against
  `94 local+neighbor_preds`. This is the one that actually isolates whether neighbor predictions
  add anything beyond the local features alone - if stage-1 sees the 72 aggregated features,
  the neighbor predictions smuggle that aggregate information back in, making a "local only"
  comparison meaningless. (An earlier run made exactly this mistake: it used a 165-feature
  stage-1 for the "94 local" comparison too, and appeared to show a real gain, 0.788 -> 0.791.
  That gain disappeared entirely once stage-1 was correctly restricted to 94 features - it was a
  leakage artifact, not a real effect.)

**A note on how not to measure this**: an earlier pass fit one model per setup (seed 0 only) and
bootstrapped directly on its scores. That gave Scenario A a bootstrap CI entirely below zero
(looked like a real negative effect) - but simply dropping the (constant, now-removed)
`has_neighbors` column changed that CI from straddling zero to excluding it, with nothing else
changed. A single seed's random forest fit has its own randomness (which features its per-split
subsampling happens to draw), and a bootstrap over the test set alone can't tell that fit-level
noise apart from a genuine effect. **Fix: average predicted scores over 5 seeds per setup before
bootstrapping**, so the bootstrap CI reflects test-set sampling variance on a stabilized score
rather than one arbitrary fit:

| scenario | setup | PR-AUC (avg of 5 seeds) | bootstrap mean diff | 95% CI |
|---|---|---|---|---|
| A | 165 feats | 0.7994 | | |
| A | 165+neighbor_preds | 0.7983 | -0.0011 | (-0.0032, 0.0009) |
| B | 94 local | 0.7901 | | |
| B | 94 local+neighbor_preds | 0.7880 | -0.0020 | (-0.0053, 0.0011) |

Both CIs now straddle zero. The apparent "real negative effect" in Scenario A was not robust
across seeds - it was single-fit noise, not a genuine effect of adding these features.

Permutation importance on the test set (mean PR-AUC drop from shuffling just that column, 10
repeats, one representative seed-0 model per scenario) is consistent with this - both
neighbor-pred features show a drop that's **approximately 0, within noise** (std of the same
order as the mean), not a meaningful positive contribution:

| scenario | feature | mean drop | std |
|---|---|---|---|
| A | neighbor_mean_pred | -0.00132 | 0.00041 |
| A | neighbor_max_pred | -0.00112 | 0.00038 |
| B | neighbor_mean_pred | 0.00041 | 0.00083 |
| B | neighbor_max_pred | -0.00070 | 0.00070 |

**Conclusion: no gain from neighbor predictions, in either scenario, once leakage is properly
controlled.** Point estimates are slightly negative in both (more so in Scenario A), but neither
is statistically distinguishable from zero, and Scenario A's apparent negative effect specifically
is not robust across seeds - it should not be read as a real cost of adding these features either,
just as noise.

## Takeaway

Across two different graph-feature designs (simple connectivity, and a more sophisticated
leakage-safe collective-classification scheme), neither added real predictive power beyond the
165 anonymized features already in the Elliptic dataset. This project's model gets essentially
all of its signal from the local and pre-aggregated features already provided - at least for the
graph features tried so far.

# Rolling-baseline drift

The fixed-baseline PSI in `src/report.py` (every step vs train steps 1-34) kept weighted and
prediction PSI far above the 0.25 "major" cutoff at every step 35-49 (median PSI ranged 0.17-0.47),
and did not flag the step-43 collapse (PR-AUC 0.89 at step 42
-> 0.04 at step 43, the known dark-market shutdown). `src/rolling_report.py` tests a rolling
baseline instead, with the rule pre-registered in its docstring before running: baseline for step
t = steps t-3..t-1, alarm if a metric exceeds mean + 2*std over steps 35-42, success = alarm at 43
and none at 35-42.

| metric | threshold | alarms at | result |
|---|---|---|---|
| median_psi | 0.059 | none | FAIL |
| weighted_psi | 0.619 | none | FAIL |
| prediction_psi | 0.159 | 43 | PASS |

**Overall: pass, but marginal.** `prediction_psi` at step 43 is 0.165 vs a 0.159 threshold; step 40
reached 0.156. One metric, one event, a few thousandths of margin. Feature-level PSI shows nothing
at step 43 - the shutdown changed what the model's scores mean more than it changed the inputs.
The alarm also does not persist: the rolling baseline absorbs the new regime, so steps 44-49 are
quiet even though PR-AUC stays at 0.01-0.27 (except 0.51 at step 46, which has only 2 illicit examples). It catches the break once; it is not a "still stale"
indicator.

**Limitations**
1. The threshold is fit on steps 35-42 and checked on those same steps, so "no false alarms at
   35-42" is in-sample and weak.
2. Baselines for steps 35-37 include train steps 32-34, where the model's scores are in-sample
   (overconfident), which may inflate `prediction_psi` there.

Plot: `results/rolling_psi.png`. Data: `results/rolling_report.csv`.
