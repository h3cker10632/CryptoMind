"""Signal screen — does a feature predict returns at all, before anyone builds
a strategy around it?

Cheap rejection is the efficient part of research: most candidate signals
don't predict anything, and finding that out from one honest statistic beats
building, backtesting and forward-tracking a strategy for each.

  cross_sectional  per day, the rank correlation between the feature and the
                   next-h-day return ACROSS coins in the point-in-time liquid
                   universe (information coefficient, IC). Pooling coins uses
                   all ~400 store coins' history even when only BTC/ETH are
                   traded, and the market's direction drops out entirely.
  time_series      one asset (e.g. BTC) against a market-wide series such as
                   Fear & Greed: correlation with the next-h-day return.

Both use Newey-West standard errors with `h` lags (overlapping h-day labels
are not independent), report each half separately, and a feature PASSES only
with |t| >= `min_t` and the same sign in both halves. Across a batch, p-values
are Holm-adjusted for the number of features tried.
"""
from __future__ import annotations
import math

import numpy as np

from . import features as F
from .evidence import nw_tstat


def forward_return(close, h):
    """(T, N) return from t's close to t+h's close (NaN when unknown)."""
    out = np.full_like(close, np.nan)
    out[:-h] = close[h:] / close[:-h] - 1
    return out


def _rank_rows(x):
    return F.cross_rank(x)


def ic_series(X, Y, mask=None, min_coins=5):
    """(T,) daily Spearman correlation between X[t] and Y[t] across coins."""
    X = np.where(mask, X, np.nan) if mask is not None else X
    ok = ~np.isnan(X) & ~np.isnan(Y)
    Xr = _rank_rows(np.where(ok, X, np.nan))
    Yr = _rank_rows(np.where(ok, Y, np.nan))
    out = np.full(X.shape[0], np.nan)
    for t in range(X.shape[0]):
        m = ok[t]
        if m.sum() < min_coins:
            continue
        a, b = Xr[t, m] - Xr[t, m].mean(), Yr[t, m] - Yr[t, m].mean()
        d = math.sqrt(float(a @ a) * float(b @ b))
        if d > 0:
            out[t] = float(a @ b) / d
    return out


def _p_two_sided(t):
    return math.erfc(abs(t) / math.sqrt(2)) if t == t else 1.0


def _verdict(series, h, min_t):
    s = series[np.isfinite(series)]
    n = len(s)
    half = n // 2
    t_all = nw_tstat(s, lags=h)
    t1, t2 = nw_tstat(s[:half], lags=h), nw_tstat(s[half:], lags=h)
    m1 = float(s[:half].mean()) if half else float("nan")
    m2 = float(s[half:].mean()) if n - half else float("nan")
    passes = (t_all == t_all and abs(t_all) >= min_t and m1 == m1 and m2 == m2
              and np.sign(m1) == np.sign(m2) != 0)
    return {"days": int(n), "mean": round(float(s.mean()), 4) if n else None,
            "t": round(t_all, 2) if t_all == t_all else None,
            "p": _p_two_sided(t_all), "halves_mean": [round(m1, 4), round(m2, 4)],
            "halves_t": [round(t1, 2) if t1 == t1 else None, round(t2, 2) if t2 == t2 else None],
            "passes": bool(passes)}


def cross_sectional(X, close, h=7, mask=None, min_t=2.0, start=0):
    """Screen one (T, N) feature against next-h-day returns across coins."""
    Y = forward_return(close, h)
    ic = ic_series(X, Y, mask)
    ic[:start] = np.nan
    return _verdict(ic, h, min_t)


def time_series(x, close, h=7, min_t=2.0, start=0):
    """Screen one (T,) series against one asset's next-h-day return. Each
    day's product of standardized feature and return is the 'IC' series."""
    x = np.asarray(x, dtype=float)
    y = forward_return(np.asarray(close, dtype=float)[:, None], h)[:, 0]
    ok = np.isfinite(x) & np.isfinite(y)
    ok[:start] = False
    s = np.full(len(x), np.nan)
    if ok.sum() > 10:
        xs = (x[ok] - x[ok].mean()) / (x[ok].std() or 1.0)
        ys = (y[ok] - y[ok].mean()) / (y[ok].std() or 1.0)
        s[ok] = xs * ys
    out = _verdict(s, h, min_t)
    out["corr"] = out.pop("mean")
    return out


def holm(results):
    """Holm-adjust the 'p' of each result in {name: result} in place, and
    clear 'passes' for those no longer significant at 5%."""
    names = sorted(results, key=lambda k: results[k]["p"])
    m = len(names)
    running = 0.0
    for i, k in enumerate(names):
        adj = min(1.0, (m - i) * results[k]["p"])
        running = max(running, adj)
        results[k]["p_holm"] = round(running, 4)
        if running > 0.05:
            results[k]["passes"] = False
        results[k]["p"] = round(results[k]["p"], 4)
    return results


def builtin_features(panel):
    """Causal per-coin features worth screening, (T, N) each."""
    c = panel.close
    lr = F.log_returns(c)
    out = {"ret_7d": F.ret_n(c, 7), "ret_30d": F.ret_n(c, 30), "ret_90d": F.ret_n(c, 90),
           "vol_30d": F.rolling_std(lr, 30), "sma_gap_50": F.sma_gap(c, 50),
           "sma_gap_200": F.sma_gap(c, 200)}
    v = np.where(np.isnan(c), np.nan, panel.volume)
    out["volume_ratio_7_30"] = F.rolling_mean(v, 7) / F.rolling_mean(v, 30)
    out["dollar_volume_30d"] = F.rolling_mean(np.where(np.isnan(c), np.nan, c * v), 30)
    peak = np.fmax.accumulate(np.where(np.isnan(c), -np.inf, c), axis=0)
    out["drawdown_from_peak"] = np.where(np.isnan(c), np.nan, c / peak - 1)
    return out
