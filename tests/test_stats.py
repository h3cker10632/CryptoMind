"""Statistical-validity toolkit tests (DSR / PSR / PBO / intervals)."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backtest import stats


def test_norm_ppf_cdf_roundtrip():
    for p in (0.1, 0.5, 0.9, 0.975):
        assert abs(stats._norm_cdf(stats._norm_ppf(p)) - p) < 1e-3


def test_psr_higher_for_better_returns():
    # positive drift with modest noise vs zero-drift noise
    good = [0.01 + 0.002 * ((-1) ** i) for i in range(100)]
    noisy = [0.002 * ((-1) ** i) for i in range(100)]
    assert stats.probabilistic_sharpe_ratio(good) > \
        stats.probabilistic_sharpe_ratio(noisy)


def test_dsr_penalizes_more_trials():
    rets = [random.gauss(0.001, 0.01) for _ in range(200)]
    few = stats.deflated_sharpe_ratio(rets, n_trials=1)
    many = stats.deflated_sharpe_ratio(rets, n_trials=5000)
    assert many <= few    # more trials -> harder bar -> lower/equal DSR


def test_expected_max_sharpe_grows_with_trials():
    assert stats.expected_max_sharpe(1000, 0.5) > stats.expected_max_sharpe(10, 0.5)


def test_wilson_interval_brackets_point():
    p, lo, hi = stats.wilson_interval(7, 10)
    assert lo <= p <= hi and 0 <= lo and hi <= 1


def test_pbo_returns_probability_or_none():
    # two strategies, one clearly better in-sample AND out — low overfit prob
    good = [0.01 + random.gauss(0, 0.002) for _ in range(200)]
    bad = [random.gauss(0, 0.01) for _ in range(200)]
    pbo = stats.probability_of_backtest_overfitting([good, bad], n_splits=6)
    assert pbo is None or (0.0 <= pbo <= 1.0)
