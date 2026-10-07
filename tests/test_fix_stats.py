"""The IC t-statistic on overlapping h-day labels must hold its size: a feature
with no predictive power passes |t| >= 2 about 5% of the time, not ~10%."""
import numpy as np
import pytest


def _null_ic(seed, T=1200, N=30, h=7):
    """Daily cross-sectional rank IC of a persistent feature (per-coin random
    walk) against independent next-h-day returns: overlapping labels, no signal."""
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(size=(T, N)), axis=0)
    c = np.vstack([np.zeros((1, N)), np.cumsum(rng.normal(0, 0.03, (T + h, N)), axis=0)])
    Y = c[1 + h:T + 1 + h] - c[1:T + 1]
    a = X.argsort(1).argsort(1).astype(float)
    b = Y.argsort(1).argsort(1).astype(float)
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


@pytest.mark.parametrize("h", [7, 30])
def test_overlapping_ic_t_has_nominal_size_under_the_null(h, monkeypatch):
    from app.engine import screen as S
    from app.ml import evaluate as Ev
    monkeypatch.setattr(S, "ic_series", lambda X, Y, mask=None: X[:, 0])   # feed ICs as-is
    t_screen, t_eval = [], []
    for seed in range(600):
        ic = _null_ic(seed, h=h)
        t_screen.append(S._verdict(ic, h, 2.0)["t"])
        one = np.ones((len(ic), 1), bool)
        t_eval.append(Ev.ranking(ic[:, None], one * 0.0, one, one * 0.0, h=h)["ic_t"])
    for ts in (t_screen, t_eval):
        rate = float(np.mean(np.abs(np.array(ts, dtype=float)) >= 2))
        assert rate <= 0.07, rate            # NW with h lags on the daily series: ~0.09-0.12


def test_overlap_tstat_gives_no_verdict_without_enough_independent_data():
    import numpy as np

    from app.engine.evidence import overlap_tstat
    x = np.random.default_rng(0).normal(0.5, 1, 300)
    assert overlap_tstat(x, 30) != overlap_tstat(x, 30)      # 10 non-overlapping obs: NaN
    assert overlap_tstat(x, 7) > 2                          # ~43 per sub-series: judged
