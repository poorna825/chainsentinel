"""
Tests for src/data.py, using small hand-built DataFrames (not the real
dataset or the fake-data generator) so every expected result can be verified
by hand and tests run instantly.
"""
import numpy as np
import pandas as pd
import pytest

from data import (
    TRAIN_STEPS,
    TEST_STEPS,
    time_split,
    feature_columns,
    add_degree_features,
    add_neighbor_pred_features,
)


# ---------- time_split ----------

def make_split_df():
    # steps straddling the 34/35 boundary, plus some unlabelled (-1) rows
    return pd.DataFrame({
        "txId":      [1,  2,  3,  4,  5,  6,  7],
        "time_step": [1, 34, 34, 35, 35, 49,  1],
        "label":     [0,  1, -1,  0,  1,  0, -1],
        "f1":        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    })


def test_time_split_boundary():
    df = make_split_df()
    train, test = time_split(df)
    assert (train["time_step"] < 35).all()
    assert (test["time_step"] >= 35).all()
    assert set(TRAIN_STEPS) == set(range(1, 35))
    assert set(TEST_STEPS) == set(range(35, 50))


def test_time_split_no_txid_overlap():
    df = make_split_df()
    train, test = time_split(df)
    assert set(train["txId"]).isdisjoint(set(test["txId"]))


def test_train_test_steps_partition_full_range_exactly():
    """TRAIN_STEPS and TEST_STEPS must cover 1..49 with no gap and no overlap.

    A gap (e.g. TRAIN_STEPS stopping at 33 instead of 34) silently drops that
    step's rows from BOTH splits - a different failure than overlap, and the
    overlap-only test above would not catch it.
    """
    all_steps = set(range(1, 50))
    assert set(TRAIN_STEPS) | set(TEST_STEPS) == all_steps
    assert set(TRAIN_STEPS).isdisjoint(set(TEST_STEPS))


def test_time_split_every_labelled_row_in_exactly_one_split():
    df = make_split_df()
    train, test = time_split(df)
    labelled = df[df["label"] != -1]
    assert set(train["txId"]) | set(test["txId"]) == set(labelled["txId"])
    assert len(train) + len(test) == len(labelled)


def test_time_split_excludes_unlabelled_by_default():
    df = make_split_df()
    train, test = time_split(df)
    assert (train["label"] != -1).all()
    assert (test["label"] != -1).all()
    # txId 3 (step 34, label -1) and txId 7 (step 1, label -1) must be gone
    assert 3 not in set(train["txId"])
    assert 7 not in set(train["txId"])


def test_time_split_labelled_only_false_keeps_unlabelled():
    df = make_split_df()
    train, test = time_split(df, labelled_only=False)
    assert -1 in set(train["label"])


def test_time_split_does_not_mutate_input():
    df = make_split_df()
    original = df.copy()
    time_split(df)
    pd.testing.assert_frame_equal(df, original)


# ---------- feature_columns ----------

def test_feature_columns_order_and_exclusion():
    df = pd.DataFrame(columns=["txId", "time_step", "label", "f1", "f2", "f3"])
    assert feature_columns(df) == ["f1", "f2", "f3"]


def test_feature_columns_count_with_different_column_order():
    # txId/time_step/label interleaved with features - must still exclude
    # them and keep only "f"-prefixed columns, in the df's own column order
    df = pd.DataFrame(columns=["f2", "txId", "f1", "label", "time_step", "f3"])
    assert feature_columns(df) == ["f2", "f1", "f3"]
    assert len(feature_columns(df)) == 3


# ---------- add_degree_features ----------

def make_degree_df():
    return pd.DataFrame({"txId": [1, 2, 3, 4], "time_step": [1, 1, 1, 1], "label": [0, 0, 0, 0]})


def make_degree_edges():
    # node 1 has a self-loop; node 4 has no edges at all
    return pd.DataFrame({
        "txId1": [1, 1, 2, 1],
        "txId2": [2, 3, 3, 1],
    })


def test_add_degree_features_directed_counts():
    df = make_degree_df()
    edges = make_degree_edges()
    out = add_degree_features(df, edges)
    expected = {1: (3, 1, 4), 2: (1, 1, 2), 3: (0, 2, 2), 4: (0, 0, 0)}  # out, in, total
    for tx_id, (exp_out, exp_in, exp_total) in expected.items():
        row = out[out["txId"] == tx_id].iloc[0]
        assert row["out_deg"] == exp_out, f"txId {tx_id} out_deg"
        assert row["in_deg"] == exp_in, f"txId {tx_id} in_deg"
        assert row["total_deg"] == exp_total, f"txId {tx_id} total_deg"


def test_add_degree_features_self_loop_counts_both_directions():
    # documents actual behavior: a self-loop (txId1 == txId2) is counted in
    # both out_deg and in_deg for that node, not excluded
    df = make_degree_df()
    edges = make_degree_edges()
    out = add_degree_features(df, edges)
    row = out[out["txId"] == 1].iloc[0]
    assert row["out_deg"] >= 1 and row["in_deg"] >= 1


def test_add_degree_features_does_not_mutate_input():
    df = make_degree_df()
    original = df.copy()
    add_degree_features(df, make_degree_edges())
    pd.testing.assert_frame_equal(df, original)


# ---------- add_neighbor_pred_features ----------

def make_neighbor_df():
    # node 4 is isolated (no edges at all)
    return pd.DataFrame({
        "txId": [1, 2, 3, 4],
        "stage1_pred": [0.1, 0.2, 0.3, 0.4],
    })


def make_neighbor_edges():
    # (1,1) is a self-loop; real edges are 1-2 and 2-3
    return pd.DataFrame({
        "txId1": [1, 2, 1],
        "txId2": [2, 3, 1],
    })


def test_add_neighbor_pred_features_undirected_and_hand_computed():
    df = make_neighbor_df()
    edges = make_neighbor_edges()
    out = add_neighbor_pred_features(df, edges, "stage1_pred")

    def row(tx_id):
        return out[out["txId"] == tx_id].iloc[0]

    # node1's only real neighbor is node2 (pred 0.2) - self-loop excluded
    assert row(1)["neighbor_mean_pred"] == pytest.approx(0.2)
    assert row(1)["neighbor_max_pred"] == pytest.approx(0.2)
    # node2's neighbors are node1 (0.1) and node3 (0.3), via BOTH edge directions
    assert row(2)["neighbor_mean_pred"] == pytest.approx(0.2)
    assert row(2)["neighbor_max_pred"] == pytest.approx(0.3)
    # node3's only neighbor is node2 (0.2)
    assert row(3)["neighbor_mean_pred"] == pytest.approx(0.2)
    assert row(3)["neighbor_max_pred"] == pytest.approx(0.2)


def test_add_neighbor_pred_features_isolated_node_is_zero_not_nan():
    df = make_neighbor_df()
    out = add_neighbor_pred_features(df, make_neighbor_edges(), "stage1_pred")
    row = out[out["txId"] == 4].iloc[0]
    assert row["neighbor_mean_pred"] == 0
    assert row["neighbor_max_pred"] == 0
    assert not pd.isna(row["neighbor_mean_pred"])
    assert not pd.isna(row["neighbor_max_pred"])


def test_add_neighbor_pred_features_self_loop_excluded():
    # if the self-loop on node1 were NOT excluded, node1's mean would include
    # its own pred (0.1) alongside node2's (0.2), giving 0.15 instead of 0.2
    df = make_neighbor_df()
    out = add_neighbor_pred_features(df, make_neighbor_edges(), "stage1_pred")
    row = out[out["txId"] == 1].iloc[0]
    assert row["neighbor_mean_pred"] == pytest.approx(0.2)


def test_add_neighbor_pred_features_does_not_mutate_input():
    df = make_neighbor_df()
    original = df.copy()
    add_neighbor_pred_features(df, make_neighbor_edges(), "stage1_pred")
    pd.testing.assert_frame_equal(df, original)
