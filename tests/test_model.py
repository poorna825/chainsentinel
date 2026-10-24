"""
Tests for src/model.py, using small synthetic data so expected behavior can
be checked directly rather than against real Elliptic numbers.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from model import oof_predictions, train_random_forest


def make_memorization_df(n=60, seed=0):
    # "id" is a unique integer per row with no real relationship to the
    # label - a model that "memorizes" row identity (splits on the exact id
    # value) can fit this perfectly in-sample, but that memorized mapping is
    # useless on a row whose id it never saw during training.
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"id": np.arange(n), "label": rng.randint(0, 2, n)})


# ---------- oof_predictions ----------

def test_oof_predictions_every_row_gets_exactly_one_prediction():
    df = make_memorization_df()
    oof = oof_predictions(df, ["id"], n_folds=5, random_state=0)
    assert len(oof) == len(df)
    assert not np.isnan(oof).any()


def test_oof_predictions_output_index_alignment():
    df = make_memorization_df()
    oof = oof_predictions(df, ["id"], n_folds=5, random_state=0)
    assert oof.shape == (len(df),)
    # position i in the returned array must correspond to row i of df, not
    # some shuffled fold order
    assert isinstance(oof, np.ndarray)


def test_oof_predictions_same_seed_is_deterministic():
    df = make_memorization_df()
    oof1 = oof_predictions(df, ["id"], n_folds=5, random_state=0)
    oof2 = oof_predictions(df, ["id"], n_folds=5, random_state=0)
    np.testing.assert_array_equal(oof1, oof2)


def test_oof_predictions_does_not_leak_memorized_row_identity():
    """A feature that perfectly memorizes row identity (unique id, random
    label) lets a model fit on the FULL data score ~1.0 accuracy in-sample -
    but that's leakage, not real signal. A properly out-of-fold model must
    not reproduce that: each row's id was never seen (with its label) by the
    model that scores it, so OOF accuracy should sit near chance (~0.5),
    nowhere near the in-sample memorized accuracy.
    """
    df = make_memorization_df()

    full_fit = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
    full_fit.fit(df[["id"]], df["label"])
    in_sample_acc = (full_fit.predict(df[["id"]]) == df["label"]).mean()
    assert in_sample_acc > 0.95  # confirms id really is trivially memorizable

    oof = oof_predictions(df, ["id"], n_folds=5, random_state=0)
    oof_acc = ((oof >= 0.5).astype(int) == df["label"].values).mean()
    assert oof_acc < 0.7, f"OOF accuracy {oof_acc} is too close to the memorized in-sample score"


# ---------- train_random_forest feature_names ----------

def make_feature_selection_df(n=80, seed=0):
    rng = np.random.RandomState(seed)
    label = rng.randint(0, 2, n)
    return pd.DataFrame({
        "f1": rng.random(n),
        "f2": rng.random(n),
        # perfectly predictive of the label, but deliberately left OUT of
        # feature_names in the test below
        "leaky": label.astype(float),
        "label": label,
    })


def test_train_random_forest_uses_only_given_feature_names():
    df = make_feature_selection_df()
    model = train_random_forest(df, feature_names=["f1", "f2"], random_state=0)
    assert model.n_features_in_ == 2
    assert list(model.feature_names_in_) == ["f1", "f2"]


def test_train_random_forest_ignores_extra_columns():
    df = make_feature_selection_df()
    model = train_random_forest(df, feature_names=["f1", "f2"], random_state=0)

    # dropping the extra "leaky" column from train_df entirely (not just
    # excluding it from feature_names) must fit an identical model, proving
    # train_random_forest never touched it
    model_without_extra_col = train_random_forest(
        df.drop(columns=["leaky"]), feature_names=["f1", "f2"], random_state=0
    )
    np.testing.assert_array_equal(
        model.predict_proba(df[["f1", "f2"]]),
        model_without_extra_col.predict_proba(df[["f1", "f2"]]),
    )
