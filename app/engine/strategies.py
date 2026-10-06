"""Strategies as pure functions: Panel -> (T, N) target weights (fractions of
the sleeve's capital). Live trading evaluates the SAME function on a panel
ending today and uses the last row (`latest`).

`trend_portfolio` reproduces the daily lab's core rules exactly (verified in
tests/test_engine.py): selection trend / momentum / score, sizing equal /
inverse_vol / vol_target, weekly re-pick for momentum and score.

Re-picks happen on FIXED calendar days (day index % rebalance_days ==
anchor), not "7 days after the backtest started": live trading has no start
day, so this is what makes a live pick identical to the backtest's pick on the
same date and reconstructible from data alone (no saved pick state).
"""
from __future__ import annotations
import math

import numpy as np

from . import features as F


REBALANCE_ANCHOR = 0          # day 0 (1970-01-01) was a Thursday -> weekly picks on Thursdays


def trend_portfolio(panel, selection="trend", sizing="equal", sma=100, top_k=5,
                    vol_target=0.5, score=None, rebalance_days=7, start=0, min_rows=31,
                    anchor=REBALANCE_ANCHOR, tranches=1, universe=None, hysteresis=0.0,
                    vol=None):
    """Target weights. `tranches` > 1 splits the capital into equal slices
    re-picked on different days of the cycle (tranche k on anchor + k*step):
    on real data the weekday a single portfolio re-picks on moved momentum's
    Sharpe between 1.03 and 1.37 — pure timing luck that tranching averages
    away (and it spreads turnover out)."""
    if tranches > 1 and selection != "trend":
        step = max(1, rebalance_days // tranches)
        Ws = [_trend_portfolio(panel, selection, sizing, sma, top_k, vol_target, score,
                               rebalance_days, start, min_rows, anchor + k * step, universe,
                               hysteresis, vol)
              for k in range(tranches)]
        return sum(Ws) / tranches
    return _trend_portfolio(panel, selection, sizing, sma, top_k, vol_target, score,
                            rebalance_days, start, min_rows, anchor, universe, hysteresis, vol)


def _trend_buffered(elig, gap, h):
    """Trend state with a buffer: switch IN only above average x (1 + h), OUT
    only below average x (1 - h). h = 0 is the plain "above the average" rule.
    The state at row t depends only on rows <= t (causal)."""
    if h <= 0:
        return elig & (gap > 0)
    T, N = gap.shape
    held = np.zeros((T, N), bool)
    state = np.zeros(N, bool)
    g = np.nan_to_num(gap, nan=-np.inf)
    for t in range(T):
        state = np.where(state, g[t] > -h, g[t] > h) & elig[t]
        held[t] = state
    return held


def _trend_portfolio(panel, selection, sizing, sma, top_k, vol_target, score,
                     rebalance_days, start, min_rows, anchor, universe=None, hysteresis=0.0,
                     vol=None):
    close = panel.close
    T, N = close.shape
    W = np.zeros((T, N))
    if T == 0 or N == 0:
        return W
    present = ~np.isnan(close)
    hist = F.history_count(close)
    elig = present & (hist >= sma)
    if universe is not None:                  # point-in-time investable set
        elig &= universe
    gap = F.sma_gap(close, sma)
    held_trend = _trend_buffered(elig, gap, hysteresis)
    known = present & (hist >= min_rows)               # the lab's "has features today"
    r30 = F.ret_n(close, 30)
    lr = F.log_returns(close)
    vol30 = F.rolling_std(lr, 30)
    pick = None
    for t in range(start, T):
        if not elig[t].any():                 # nothing eligible: no pick today
            continue
        if selection == "trend":
            held = np.flatnonzero(held_trend[t])
            n_slots = int(elig[t].sum())
        else:
            due = (int(panel.days[t]) - anchor) % rebalance_days == 0
            if pick is not None and not due:
                held = np.array([j for j in pick if known[t, j]], dtype=int)
            else:
                cand = np.flatnonzero(held_trend[t])
                s = (r30 if selection == "momentum" else score)[t, cand]
                ok = ~np.isnan(s)
                cand, s = cand[ok], s[ok]
                held = cand[np.argsort(-s, kind="stable")][:top_k]
                pick = held
            n_slots = top_k
        if not len(held):
            continue
        if sizing == "inverse_vol":
            v = vol30[t, held]
            ok = ~np.isnan(v)
            iv = 1.0 / np.maximum(v[ok], 1e-4)
            W[t, held[ok]] = (len(held) / n_slots) * iv / iv.sum()
            continue
        w = np.full(len(held), 1.0 / n_slots)
        if sizing == "vol_forecast" and vol is not None:
            # per-coin: scale each slot down to `vol_target` using the
            # FORECAST volatility (risk_models.har_vol_forecast); never up
            f = vol[t, held]
            w = w * np.where(np.isfinite(f) & (f > 0), np.minimum(1.0, vol_target / f), 1.0)
        if sizing == "vol_target":
            pv = _port_vol(lr, t, held, w)
            if pv and pv > 0:
                w = w * min(vol_target / pv, 1.0 / w.sum())
        W[t, held] = w
    return W


def _port_vol(lr, t, held, w, lookback=60):
    """Annualized volatility of weights `w` on coins `held` from their last
    `lookback` daily log returns (coins with < 20 valid returns are left out,
    as in the daily lab)."""
    lo = max(0, t - lookback + 1)
    win = lr[lo:t + 1, held]
    keep = (~np.isnan(win)).sum(axis=0) >= 20
    if not keep.any():
        return None
    sub = win[:, keep]
    rows = ~np.isnan(sub).any(axis=1)
    sub = sub[rows]
    if len(sub) < 2:
        return None
    cov = np.atleast_2d(np.cov(sub.T))
    wk = w[keep]
    return math.sqrt(max(float(wk @ cov @ wk), 0.0) * 365)


def buy_hold(panel, start=0, min_rows=31):
    """Equal weight in every coin with >= `min_rows` of history and a bar today
    and tomorrow (rebalanced daily)."""
    close = panel.close
    W = np.zeros_like(close)
    known = ~np.isnan(close) & (F.history_count(close) >= min_rows)
    for t in range(start, close.shape[0] - 1):
        live = known[t] & ~np.isnan(close[t + 1])
        if live.any():
            W[t, live] = 1.0 / live.sum()
    return W


def latest(panel, strategy=trend_portfolio, lookback_days=None, **kw):
    """{coin: weight} for TODAY from the same function the backtest ran.
    Path-dependent strategies (weekly picks) are replayed over the last
    `lookback_days` so today's pick state matches the simulation."""
    start = 0 if lookback_days is None else max(0, panel.T - lookback_days)
    W = strategy(panel, start=start, **kw)
    if not panel.T:
        return {}
    return {c: float(w) for c, w in zip(panel.coins, W[-1]) if w > 0}
