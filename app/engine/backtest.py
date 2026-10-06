"""Vectorized portfolio simulation and honest metrics.

Convention (same as the daily lab): weights W[t] are decided at day t's close
and earn day t+1's return; turnover |W[t] - W[t-1]| pays `cost_side` per unit
traded at t. A coin with no bar tomorrow earns 0 for that day.
"""
from __future__ import annotations
import math

import numpy as np


def simulate(W, R, cost_side=0.006, start=0):
    """Daily net returns for t = start .. T-2 (each earning t -> t+1)."""
    Wz = np.nan_to_num(W)
    Rz = np.nan_to_num(R)
    T = Wz.shape[0]
    if T - 1 <= start:
        return np.zeros(0), np.zeros(0)
    w = Wz[start:T - 1]
    prev = np.vstack([np.zeros((1, Wz.shape[1])), w[:-1]])
    turnover = np.abs(w - prev).sum(axis=1)
    gross = (w * Rz[start + 1:T]).sum(axis=1)
    return gross - cost_side * turnover, w.sum(axis=1)


def simulate_drift(W, R, cost_side=0.006, start=0, band_rel=0.0, band_abs=0.0):
    """Realistic daily simulation: holdings DRIFT with prices between trades,
    and every trade (including re-balancing drift back to target) pays
    `cost_side`. A coin is only traded when its holding is outside a no-trade
    band around target — max(band_abs, band_rel * target) — or when it enters
    / exits the portfolio. band 0 = trade back to target every day.

    `simulate` (above) charges only changes in TARGET weights and assumes the
    book sits exactly at target, so it under-counts costs; this one doesn't.
    `cost_side` may be a scalar or a (T, N) per-coin, per-day cost (see
    risk_models.cost_per_side).
    Returns (daily net returns, invested fraction, turnover per day)."""
    Wz = np.nan_to_num(W)
    C = np.broadcast_to(np.asarray(cost_side, dtype=float), Wz.shape)         if np.ndim(cost_side) else None
    Rz = np.nan_to_num(R)
    T, N = Wz.shape
    h = np.zeros(N)                         # holdings as fraction of equity
    rets, inv, turns = [], [], []
    for t in range(start, T - 1):
        tgt = Wz[t]
        diff = tgt - h
        band = np.maximum(band_abs, band_rel * tgt)
        trade = (np.abs(diff) > band) | ((tgt == 0) & (h != 0)) | ((tgt > 0) & (h == 0))
        delta = np.where(trade, diff, 0.0)
        cost = (C[t] * np.abs(delta)).sum() if C is not None else             cost_side * np.abs(delta).sum()
        h = h + delta
        cash = 1.0 - h.sum() - cost
        grow = h * (1 + Rz[t + 1])
        port = grow.sum() + cash
        r = port - 1.0
        h = grow / port if port > 0 else np.zeros(N)
        rets.append(r)
        inv.append(float((h * port).sum() / port) if port > 0 else 0.0)
        turns.append(float(np.abs(delta).sum()))
    return np.array(rets), np.array(inv), np.array(turns)


def stats(rets, periods=365):
    """Return %, annualized Sharpe, max drawdown % (same as the daily lab)."""
    rets = np.asarray(rets, dtype=float)
    if not len(rets):
        return {"return_pct": None, "sharpe": None, "max_drawdown_pct": None, "days": 0}
    eq = np.cumprod(1 + rets)
    dd = eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1
    sd = rets.std()
    return {"return_pct": round((eq[-1] - 1) * 100, 1),
            "sharpe": round(float(rets.mean() / sd * math.sqrt(periods)), 2) if sd > 0 else 0.0,
            "max_drawdown_pct": round(float(min(dd.min(), 0.0)) * 100, 1),
            "days": int(len(rets))}


DEFAULT_TRIAL_SR_STD_ANNUAL = 0.5      # spread of Sharpes across tried variants


def deflated_sharpe(rets, n_trials=1, trial_sr_std=None, periods=365):
    """Probability the strategy's true Sharpe beats the best Sharpe expected
    from `n_trials` skill-less variants (Bailey & Lopez de Prado). Everything
    in PER-PERIOD units: the shared stats helper defaults the trial spread to
    0.5 per period (~9.5/yr on daily data), which deflates any daily strategy
    to zero, and treats one trial as two — so it is handled here."""
    from ..backtest.stats import probabilistic_sharpe_ratio, expected_max_sharpe
    r = list(map(float, rets))
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(r, 0.0)
    if not trial_sr_std or trial_sr_std <= 0:
        trial_sr_std = DEFAULT_TRIAL_SR_STD_ANNUAL / math.sqrt(periods)
    return probabilistic_sharpe_ratio(r, expected_max_sharpe(n_trials, trial_sr_std))


def report(rets, invested=None, n_trials=1, trial_sr_std=None):
    """Full / halves stats, CAGR, average invested, and the deflated Sharpe
    ratio (probability the Sharpe beats the best of `n_trials` random tries)."""
    rets = np.asarray(rets, dtype=float)
    h = len(rets) // 2
    out = {"full": stats(rets), "first_half": stats(rets[:h]), "second_half": stats(rets[h:])}
    if len(rets):
        yrs = len(rets) / 365
        out["cagr_pct"] = round(((np.prod(1 + rets)) ** (1 / yrs) - 1) * 100, 1) if yrs > 0 else None
        out["deflated_sharpe"] = round(float(deflated_sharpe(rets, n_trials, trial_sr_std)), 3)
        out["n_trials"] = int(n_trials)
    if invested is not None and len(invested):
        out["avg_invested_pct"] = round(float(np.mean(invested)) * 100, 1)
    return out
