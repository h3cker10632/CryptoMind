"""Unit tests for the pure purged/embargoed walk-forward harness.

These are lab- and pandas-independent: fold geometry and pooling are plain Python,
so they pin down exactly the leakage-prevention and consistency semantics the ML
promotion gate depends on.
"""
from app.learn.validation import walk_forward_folds, aggregate_oos


def test_folds_are_expanding_and_contiguous_without_embargo():
    folds = walk_forward_folds(100, n_folds=4, embargo=0)
    assert folds == [(0, 20, 20, 40), (0, 40, 40, 60),
                     (0, 60, 60, 80), (0, 80, 80, 100)]
    # every train block starts at 0 and grows; test always follows train
    for tr0, tr1, te0, te1 in folds:
        assert tr0 == 0 and tr1 < te1 and te0 >= tr1


def test_embargo_inserts_a_purge_gap_between_train_and_test():
    folds = walk_forward_folds(100, n_folds=4, embargo=5)
    for tr0, tr1, te0, te1 in folds:
        # test must start exactly `embargo` rows after train ends (no overlap)
        assert te0 == tr1 + 5
        assert te0 < te1  # non-empty test


def test_last_fold_test_runs_to_the_end():
    folds = walk_forward_folds(100, n_folds=4, embargo=0)
    assert folds[-1][3] == 100


def test_tiny_dataset_degrades_gracefully():
    assert walk_forward_folds(3, n_folds=4, embargo=0) == []
    assert walk_forward_folds(0, n_folds=4) == []
    assert walk_forward_folds(1, n_folds=4) == []


def test_large_embargo_drops_folds_with_empty_test():
    # an embargo wide enough to eat a test block should drop that fold, not crash
    folds = walk_forward_folds(30, n_folds=4, embargo=100)
    assert folds == []


def test_aggregate_uses_median_return_and_worst_drawdown():
    per_fold = [
        {"total_return": 0.10, "max_drawdown": -0.05},
        {"total_return": -0.20, "max_drawdown": -0.30},
        {"total_return": 0.05, "max_drawdown": -0.02},
    ]
    agg = aggregate_oos(per_fold)
    assert agg["total_return"] == 0.05           # median, not the mean
    assert agg["max_drawdown"] == -0.30          # worst (most negative) fold
    assert abs(agg["oos_frac_folds_positive"] - 2 / 3) < 1e-9
    assert agg["oos_n_folds"] == 3
    assert agg["_source"] == "purged_walkforward"


def test_aggregate_all_positive_folds():
    agg = aggregate_oos([
        {"total_return": 0.2, "max_drawdown": -0.01},
        {"total_return": 0.3, "max_drawdown": -0.04},
    ])
    assert agg["oos_frac_folds_positive"] == 1.0


def test_aggregate_returns_none_when_no_usable_returns():
    assert aggregate_oos([]) is None
    assert aggregate_oos([{"total_return": None, "max_drawdown": None}]) is None


def test_aggregate_tolerates_missing_drawdowns():
    agg = aggregate_oos([
        {"total_return": 0.1, "max_drawdown": None},
        {"total_return": 0.2, "max_drawdown": None},
    ])
    assert agg["max_drawdown"] is None
    assert agg["total_return"] == 0.15000000000000002 or abs(agg["total_return"] - 0.15) < 1e-9
