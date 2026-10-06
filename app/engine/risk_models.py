"""ML / statistical models for RISK, COSTS and REGIME — the things that are
actually forecastable — all causal (row t uses rows <= t) and fitted
walk-forward (a model used on day t was fitted only on labels that had fully
happened before t).

  har_vol_forecast   next-week volatility per coin: HAR regression (daily,
                     weekly, monthly Parkinson range variance), refit monthly
  spread_estimate    Corwin-Schultz bid-ask spread from daily highs / lows
  cost_per_side      fee + half-spread + square-root market impact
  regime_probability P(next 30 days are up) for the market, gradient-boosted
                     trees on BTC trend / vol / drawdown / funding features
"""
from __future__ import annotations
import math

import numpy as np

from . import features as F

LN2x4 = 4 * math.log(2)


# ------------------------------------------------------------------ volatility
def parkinson_var(panel):
    """Daily variance proxy from the high-low range (Parkinson, 1980): far
    less noisy than a squared close-to-close return. Without highs/lows, the
    squared daily log return."""
    if panel.high is None or panel.low is None:
        lr = F.log_returns(panel.close)
        return lr * lr
    hl = np.log(panel.high / panel.low)
    v = hl * hl / LN2x4
    return np.where(np.isfinite(v), v, np.nan)


def realized_forward_var(v, h):
    """Mean of v over the NEXT h days (t+1 .. t+h): the forecasting target."""
    out = np.full_like(v, np.nan)
    m = F.rolling_mean(v, h)                        # mean over t-h+1 .. t
    out[:-h] = m[h:]
    return out


def har_vol_forecast(panel, horizon=7, min_train=365, refit_days=30):
    """(T, N) annualized vol forecast for the next `horizon` days.

    Per coin: log(next-h mean variance) ~ a + b1 log(v_d) + b2 log(v_w) +
    b3 log(v_m) (HAR in logs: robust to crypto's fat tails), pooled over
    coins, refit every `refit_days` on rows whose target had fully happened.
    Before the first fit: the 30-day mean (naive) forecast."""
    v = parkinson_var(panel)
    vd = v
    vw = F.rolling_mean(v, 7)
    vm = F.rolling_mean(v, 30)
    tgt = realized_forward_var(v, horizon)
    eps = 1e-10
    X = np.stack([np.log(vd + eps), np.log(vw + eps), np.log(vm + eps)], axis=-1)
    y = np.log(tgt + eps)
    T, N = v.shape
    pred = np.where(np.isnan(vm), np.nan, vm)       # naive fallback
    beta = None
    for t in range(T):
        if t >= min_train and (beta is None or t % refit_days == 0):
            rows = slice(0, max(0, t - horizon))      # labels ended before t
            Xa = X[rows].reshape(-1, 3)
            ya = y[rows].reshape(-1)
            ok = np.isfinite(Xa).all(axis=1) & np.isfinite(ya)
            if ok.sum() > 200:
                A = np.column_stack([np.ones(ok.sum()), Xa[ok]])
                beta, *_ = np.linalg.lstsq(A, ya[ok], rcond=None)
        if beta is not None:
            xt = X[t]
            ok = np.isfinite(xt).all(axis=1)
            pred[t, ok] = np.exp(beta[0] + xt[ok] @ beta[1:])
    return np.sqrt(pred * 365)


def vol_forecast_quality(panel, forecast, horizon=7, start=0):
    """Out-of-sample error of a vol forecast vs the realized next-h vol, next
    to the naive 30-day-realized forecast (lower is better)."""
    v = parkinson_var(panel)
    real = np.sqrt(realized_forward_var(v, horizon) * 365)
    naive = np.sqrt(F.rolling_mean(v, 30) * 365)
    def err(f):
        e = (np.log(f[start:]) - np.log(real[start:])) ** 2
        return float(np.nanmean(np.where(np.isfinite(e), e, np.nan)))
    return {"model_log_mse": round(err(forecast), 4), "naive_log_mse": round(err(naive), 4)}


# ------------------------------------------------------------------ costs
def spread_estimate(panel, window=30):
    """Corwin & Schultz (2012) bid-ask spread from two-day high/low ranges,
    floored at 0, smoothed by a `window`-day rolling mean."""
    H, L = panel.high, panel.low
    beta = np.log(H / L) ** 2
    beta = beta + F.shift(beta, 1)
    h2 = np.fmax(H, F.shift(H, 1))
    l2 = np.fmin(L, F.shift(L, 1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = np.where(np.isfinite(s), np.maximum(s, 0.0), np.nan)
    return F.rolling_mean(np.nan_to_num(s), window)


def cost_per_side(panel, fee=0.005, trade_usd=20_000, impact_k=0.1, window=30):
    """(T, N) estimated cost per side: fee + half the estimated spread +
    impact_k * sqrt(trade size / average daily dollar volume)."""
    spread = np.nan_to_num(spread_estimate(panel, window))
    adv = F.rolling_mean(np.where(np.isnan(panel.close), np.nan,
                                  panel.close * panel.volume), window)
    impact = impact_k * np.sqrt(trade_usd / np.maximum(np.nan_to_num(adv), 1.0))
    return fee + spread / 2 + np.minimum(impact, 0.05)


# ------------------------------------------------------------------ regime
def regime_features(panel, funding_daily=None, market="BTC-USD"):
    """(T, K) market features for day t (causal). `funding_daily`: optional
    (T,) average daily funding (NaN where unknown — trees handle it)."""
    j = panel.coins.index(market)
    c = panel.close[:, [j]]
    lr = F.log_returns(c)
    feats = [F.ret_n(c, 7), F.ret_n(c, 30), F.ret_n(c, 90), F.ret_n(c, 180),
             F.rolling_std(lr, 30), F.rolling_std(lr, 90),
             F.sma_gap(c, 50), F.sma_gap(c, 125), F.sma_gap(c, 200),
             c / _rolling_max(c, 365) - 1]
    X = np.hstack(feats)
    if funding_daily is not None:
        f = np.asarray(funding_daily, dtype=float).reshape(-1, 1)
        X = np.hstack([X, f, F.rolling_mean(f, 7), F.rolling_mean(f, 30)])
    return X


def _rolling_max(x, n):
    out = np.full_like(x, np.nan)
    for t in range(len(x)):
        w = x[max(0, t - n + 1):t + 1]
        if np.isfinite(w).any():
            out[t] = np.nanmax(w)
    return out


def regime_probability(X, fwd_return, horizon=30, min_train=730, refit_days=30, seed=0):
    """(T,) walk-forward P(next `horizon`-day market return > 0). A fit on day
    t only uses rows whose label window ended before t. NaN before the first
    fit."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return np.full(len(X), np.nan)
    T = len(X)
    p = np.full(T, np.nan)
    y = (fwd_return > 0).astype(float)
    model = None
    for t in range(T):
        if t >= min_train and (model is None or t % refit_days == 0):
            n = max(0, t - horizon)
            ok = np.isfinite(fwd_return[:n])
            if ok.sum() > 300 and len(np.unique(y[:n][ok])) == 2:
                model = HistGradientBoostingClassifier(
                    max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=60,
                    l2_regularization=1.0, random_state=seed)
                model.fit(X[:n][ok], y[:n][ok])
        if model is not None:
            p[t] = model.predict_proba(X[t:t + 1])[0, 1]
    return p


def forward_return(close, horizon):
    """Market forward return over the next `horizon` days (the regime label)."""
    out = np.full(len(close), np.nan)
    out[:-horizon] = close[horizon:] / close[:-horizon] - 1
    return out
