"""Causal features on (T, N) arrays: row t uses only rows <= t. A window with
any missing bar yields NaN (that coin is simply not eligible that day)."""
from __future__ import annotations
import numpy as np


def shift(x, n):
    out = np.full_like(x, np.nan)
    if n < len(x):
        out[n:] = x[:-n] if n else x
    return out


def history_count(close):
    """Bars seen so far per coin (including today)."""
    return np.cumsum(~np.isnan(close), axis=0)


def rolling_sum(x, n):
    """Sum over the last n rows; NaN unless all n are present."""
    v = ~np.isnan(x)
    cs = np.cumsum(np.where(v, x, 0.0), axis=0)
    cv = np.cumsum(v, axis=0)
    s = cs.copy()
    c = cv.copy()
    s[n:] -= cs[:-n]
    c[n:] -= cv[:-n]
    out = np.where(c == n, s, np.nan)
    out[:n - 1] = np.nan
    return out


def rolling_mean(x, n):
    return rolling_sum(x, n) / n


def rolling_std(x, n):
    """Population std over the last n rows (NaN unless all present)."""
    m = rolling_mean(x, n)
    m2 = rolling_mean(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def ret_n(close, n):
    return close / shift(close, n) - 1


def log_returns(close):
    return np.log(close / shift(close, 1))


def sma_gap(close, n):
    return close / rolling_mean(close, n) - 1


def dollar_volume(close, volume):
    return close * volume


def cross_rank(x):
    """Per-row percentile rank in (0, 1) among non-NaN entries (NaN stays NaN)."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        ok = ~np.isnan(row)
        k = ok.sum()
        if k:
            order = np.argsort(np.argsort(row[ok]))
            out[t, ok] = (order + 0.5) / k
    return out
