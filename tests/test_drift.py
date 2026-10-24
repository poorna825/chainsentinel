"""
Tests for src/drift.py, using small hand-built arrays/DataFrames so every
expected result can be verified by hand.
"""
import numpy as np
import pandas as pd
import pytest

from drift import psi, psi_over_time


# ---------- psi ----------

def test_psi_identical_distributions_is_zero():
    train = np.arange(1, 101)
    new = np.arange(1, 101)  # same values -> every bucket proportion matches exactly
    assert psi(train, new) == pytest.approx(0.0)


def test_psi_hand_computable_shift():
    # buckets=2: train quantiles [1,2,3,4] at [0,.5,1] -> edges [1, 2.5, 4],
    # deduped and capped -> [-inf, 2.5, inf]. train 1,2 fall below 2.5 (2
    # values), 3,4 fall above (2 values) -> train_props = [0.5, 0.5].
    train = [1, 2, 3, 4]
    # new: 1,1,1 fall below 2.5 (3 values), 4 falls above (1 value)
    new = [1, 1, 1, 4]
    expected = (0.75 - 0.5) * np.log(0.75 / 0.5) + (0.25 - 0.5) * np.log(0.25 / 0.5)
    assert psi(train, new, buckets=2) == pytest.approx(expected)


def test_psi_out_of_range_values_stay_finite():
    train = np.arange(1, 101)  # 1..100
    new = np.full(20, 10_000.0)  # wildly outside train's observed range
    result = psi(train, new)
    assert np.isfinite(result)
    assert not np.isnan(result)


def test_psi_empty_train_bin_stays_finite():
    # buckets=3 on this train array produces a middle bucket with ZERO train
    # count (edges [-inf, 1.667, 2, inf] -> train counts [2, 0, 4]) - this
    # exercises the epsilon floor on the train side, not just the new side.
    train = [1, 1, 2, 2, 2, 3]
    new = [1, 1.8, 1.9, 3, 3, 3]  # puts values into that empty-train bucket
    result = psi(train, new, buckets=3)
    assert np.isfinite(result)
    assert not np.isnan(result)


# ---------- psi_over_time ----------

def make_time_df(f1_values, f2_values, f3_values, time_step):
    n = len(f1_values)
    return pd.DataFrame({
        "time_step": [time_step] * n,
        "f1": f1_values,
        "f2": f2_values,
        "f3": f3_values,
    })


def test_psi_over_time_one_row_per_step():
    train_df = make_time_df(np.linspace(0, 1, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30), 1)
    full_df = pd.concat([
        make_time_df(np.linspace(0, 1, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30), 10),
        make_time_df(np.linspace(0, 1, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30), 20),
    ])
    rows = psi_over_time(train_df, full_df, ["f1", "f2", "f3"], [10, 20])
    assert len(rows) == 2
    assert [r["time_step"] for r in rows] == [10, 20]


def test_psi_over_time_median_robust_to_extreme_feature():
    # f1 is shifted hard (train 0..1, test 100..101); f2/f3 are unshifted
    # (identical to train) -> psi ~0. With 3 features, sorted psi values are
    # [~0, ~0, huge]: the median is ~0 (matches the two calm features), but
    # the mean would be pulled way up toward the huge value.
    train_df = make_time_df(np.linspace(0, 1, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30), 1)
    full_df = make_time_df(np.linspace(100, 101, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30), 10)

    rows = psi_over_time(train_df, full_df, ["f1", "f2", "f3"], [10])
    row = rows[0]

    assert row["median_psi"] == pytest.approx(0.0, abs=1e-6)
    assert row["max_psi_feature"] == "f1"
    assert row["max_psi"] > 1.0  # f1's shift is large, nowhere near 0
    # the mean would NOT be robust - sanity check the setup actually produces
    # a case where mean and median diverge
    mean_psi = np.mean([psi(train_df["f1"], full_df["f1"]),
                         psi(train_df["f2"], full_df["f2"]),
                         psi(train_df["f3"], full_df["f3"])])
    assert mean_psi > row["median_psi"] + 0.1


def test_psi_over_time_baseline_is_fixed_across_steps():
    # same train_df used for two different test steps that happen to carry
    # the identical shift relative to train -> both steps' PSI numbers must
    # come out equal, which they can only do if train_df (the baseline)
    # isn't being recomputed or resliced per step.
    train_df = make_time_df(np.linspace(0, 1, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30), 1)
    full_df = pd.concat([
        make_time_df(np.linspace(100, 101, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30), 10),
        make_time_df(np.linspace(100, 101, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30), 20),
    ])
    rows = psi_over_time(train_df, full_df, ["f1", "f2", "f3"], [10, 20])
    assert rows[0]["median_psi"] == pytest.approx(rows[1]["median_psi"])
    assert rows[0]["max_psi"] == pytest.approx(rows[1]["max_psi"])
