"""BCa (bias-corrected & accelerated) bootstrap CI tests."""
import os, sys, random, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backtest import stats as st


def test_returns_point_lo_hi_and_brackets_mean():
    vals = [random.gauss(0.01, 0.02) for _ in range(200)]
    point, lo, hi = st.bootstrap_ci(vals, seed=1)
    assert lo is not None and hi is not None
    assert abs(point - statistics.fmean(vals)) < 1e-5    # point is rounded to 6dp
    assert lo <= point <= hi


def test_too_few_values_returns_nones():
    assert st.bootstrap_ci([]) == (None, None, None)
    assert st.bootstrap_ci([0.1]) == (None, None, None)


def test_reproducible_with_seed():
    vals = [random.gauss(0, 1) for _ in range(100)]
    a = st.bootstrap_ci(vals, seed=42)
    b = st.bootstrap_ci(vals, seed=42)
    assert a == b


def test_bca_differs_from_percentile_on_skewed_data():
    # heavily right-skewed sample: many small values, a few large ones.
    # BCa should shift the interval relative to the naive percentile method.
    random.seed(7)
    vals = [random.expovariate(1.0) for _ in range(300)]     # skew ~2
    _, bca_lo, bca_hi = st.bootstrap_ci(vals, seed=3, method="bca")
    _, pct_lo, pct_hi = st.bootstrap_ci(vals, seed=3, method="percentile")
    # same bootstrap draws (same seed), so any difference is the BCa adjustment
    assert (abs(bca_lo - pct_lo) > 1e-6) or (abs(bca_hi - pct_hi) > 1e-6)


def test_symmetric_data_bca_close_to_percentile():
    # on symmetric data the BCa correction should be small (a~0, z0~0)
    random.seed(11)
    vals = [random.gauss(0.0, 1.0) for _ in range(500)]
    _, bca_lo, bca_hi = st.bootstrap_ci(vals, seed=5, method="bca")
    _, pct_lo, pct_hi = st.bootstrap_ci(vals, seed=5, method="percentile")
    assert abs(bca_lo - pct_lo) < 0.15
    assert abs(bca_hi - pct_hi) < 0.15


def test_identical_values_falls_back_gracefully():
    # zero variability -> jackknife den 0 -> percentile fallback, no crash
    point, lo, hi = st.bootstrap_ci([0.05] * 50, seed=1)
    assert point == 0.05 and lo == 0.05 and hi == 0.05


def test_wider_ci_for_smaller_sample():
    random.seed(2)
    big = [random.gauss(0.01, 0.05) for _ in range(500)]
    small = [random.gauss(0.01, 0.05) for _ in range(20)]
    _, blo, bhi = st.bootstrap_ci(big, seed=9)
    _, slo, shi = st.bootstrap_ci(small, seed=9)
    assert (shi - slo) > (bhi - blo)      # smaller sample -> wider interval
