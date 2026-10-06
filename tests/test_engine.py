"""Portfolio engine: reproduces the daily lab exactly, never looks ahead, live
weights equal the backtest's row for the same day, registry counts trials."""
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from app.engine import panel as P, strategies as S, backtest as B, features as F

COINS = ("BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "AVAX-USD", "DOGE-USD", "XRP-USD")


def _daily(n, seed, drift, t0):
    r = random.Random(seed)
    px, out = 100.0, []
    for i in range(n):
        px *= math.exp(drift + r.gauss(0, 0.03))
        out.append([t0 + i * 86400, px * 0.98, px * 1.02, px, px, 1000 + r.random() * 100])
    return out


def _universe(n=500):
    t0 = int(time.time() // 86400 - n - 2) * 86400
    return {p: _daily(n, i, 0.002 * (i - 3) / 3, t0) for i, p in enumerate(COINS)}


@pytest.mark.parametrize("sel", ["trend", "momentum"])
@pytest.mark.parametrize("siz", ["equal", "inverse_vol", "vol_target"])
def test_engine_reproduces_daily_lab(sel, siz):
    from app.backtest import daily_lab as dl
    U = _universe()
    days, pan = dl.daily_panel(U)
    fbd = dl._features_by_day(days, pan, 100)
    start_day = days[150]
    old, _ = dl.simulate(fbd, pan, sel, siz, None, start_day=start_day, sma_days=100, top_k=3)
    p = P.from_candles(U)
    s0 = int(np.searchsorted(p.days, start_day))
    W = S.trend_portfolio(p, sel, siz, sma=100, top_k=3, start=s0, anchor=start_day)
    new, _ = B.simulate(W, p.returns(), 0.006, start=s0)
    assert len(old) == len(new)
    assert np.max(np.abs(np.array(old) - new)) < 1e-12


def test_rolling_features_match_naive():
    x = np.random.default_rng(0).normal(size=(80, 3))
    x[10, 1] = np.nan
    m = F.rolling_mean(x, 5)
    for t in range(80):
        for j in range(3):
            w = x[max(0, t - 4):t + 1, j]
            want = w.mean() if t >= 4 and not np.isnan(w).any() else np.nan
            assert (np.isnan(want) and np.isnan(m[t, j])) or abs(m[t, j] - want) < 1e-12
    s = F.rolling_std(x, 5)
    assert abs(s[40, 0] - x[36:41, 0].std()) < 1e-12


def test_weights_never_depend_on_the_future():
    U = _universe()
    p = P.from_candles(U)
    W = S.trend_portfolio(p, "momentum", "vol_target", sma=100, top_k=3)
    cut = 300
    p2 = P.Panel(p.days, p.coins, p.close.copy(), p.volume.copy())
    p2.close[cut + 1:] *= np.random.default_rng(1).uniform(0.2, 5.0, p2.close[cut + 1:].shape)
    W2 = S.trend_portfolio(p2, "momentum", "vol_target", sma=100, top_k=3)
    assert np.array_equal(W[:cut + 1], W2[:cut + 1])


def test_live_weights_equal_backtest_row_for_the_same_day():
    U = _universe()
    full = P.from_candles(U)
    W = S.trend_portfolio(full, "momentum", "vol_target", sma=100, top_k=3)
    for t in (250, 300, 371, 499):
        today = P.Panel(full.days[:t + 1], full.coins, full.close[:t + 1], full.volume[:t + 1])
        live = S.latest(today, lookback_days=60, selection="momentum", sizing="vol_target",
                        sma=100, top_k=3)
        want = {c: float(w) for c, w in zip(full.coins, W[t]) if w > 0}
        assert live.keys() == want.keys()
        assert all(abs(live[c] - want[c]) < 1e-12 for c in want)


def test_registry_counts_distinct_trials(tmp_path, monkeypatch):
    from app.engine import registry as R
    monkeypatch.setattr(R, "DIR", str(tmp_path))
    r = np.random.default_rng(0).normal(0.001, 0.02, 400)
    R.log("core", "a", {"k": 1}, "v1", r, {})
    R.log("core", "a", {"k": 1}, "v1", r, {})          # identical re-run: same trial
    R.log("core", "a", {"k": 1}, "v2", r, {})          # same variant, newer data
    R.log("core", "b", {"k": 2}, "v1", r * 1.1, {})
    R.log("other", "c", {}, "v1", r, {})
    assert R.n_trials("core") == 2
    assert R.trial_sharpe_std("core") is not None


def test_report_has_halves_and_deflated_sharpe():
    r = np.random.default_rng(0).normal(0.001, 0.02, 730)
    rep = B.report(r, invested=np.ones(730), n_trials=50)
    assert {"full", "first_half", "second_half", "cagr_pct", "deflated_sharpe"} <= set(rep)
    assert 0 <= rep["deflated_sharpe"] <= 1


def test_tranches_average_the_pick_days_and_stay_live_consistent():
    U = _universe()
    p = P.from_candles(U)
    kw = dict(selection="momentum", sizing="equal", sma=100, top_k=3)
    Wt = S.trend_portfolio(p, tranches=7, **kw)
    Ws = [S.trend_portfolio(p, anchor=a, **kw) for a in range(7)]
    assert np.allclose(Wt, sum(Ws) / 7)
    t = 400
    today = P.Panel(p.days[:t + 1], p.coins, p.close[:t + 1], p.volume[:t + 1])
    live = S.latest(today, lookback_days=60, tranches=7, **kw)
    assert all(abs(live.get(c, 0.0) - w) < 1e-12 for c, w in zip(p.coins, Wt[t]))


def test_trend_buffer_switches_at_the_band_and_is_causal():
    elig = np.ones((6, 1), bool)
    gap = np.array([[0.01], [0.03], [0.0], [-0.01], [-0.03], [0.01]])
    held = S._trend_buffered(elig, gap, 0.02)[:, 0].tolist()
    # in only above +2%, out only below -2%
    assert held == [False, True, True, True, False, False]
    assert S._trend_buffered(elig, gap, 0.0)[:, 0].tolist() == \
        [True, True, False, False, False, True]


def test_universe_mask_is_point_in_time():
    from app.engine import universe as U
    U_ = _universe(400)
    p = P.from_candles(U_)
    m = U.liquid_mask(p, n=3, min_history=50, min_dollar_volume=0)
    assert m[:49].sum() == 0                          # nobody has 50 days yet
    assert (m[60:].sum(axis=1) == 3).all()
    p2 = P.Panel(p.days, p.coins, p.close.copy(), p.volume.copy())
    p2.volume[201:] *= 1000                           # the future changes...
    m2 = U.liquid_mask(p2, n=3, min_history=50, min_dollar_volume=0)
    assert np.array_equal(m[:201], m2[:201])          # ...the past universe doesn't


def test_drift_simulation_charges_rebalancing_and_respects_band():
    T = 30
    W = np.full((T, 2), 0.5)
    R = np.zeros((T, 2))
    R[1::2, 0], R[2::2, 0] = 0.10, -0.0909           # coin 0 oscillates, drifting the book
    tight, _, t_tight = B.simulate_drift(W, R, 0.01, band_rel=0.0)
    loose, _, t_loose = B.simulate_drift(W, R, 0.01, band_rel=0.5)
    assert t_tight[1:].sum() > 0                      # drift back to target is traded...
    assert t_loose[1:].sum() == 0                     # ...unless inside the band
    naive, _ = B.simulate(W, R, 0.01)
    assert naive.sum() > tight.sum()                  # target-only costs under-count
