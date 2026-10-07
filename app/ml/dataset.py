"""Features and labels for the pooled model — every (day, coin) of the
point-in-time liquid universe is one sample, so the model learns from ~all
store coins' history (incl. delisted ones) even when only BTC/ETH trade.

All features are CAUSAL (row t uses rows <= t). Labels follow the
recommendations that make noisy returns learnable:

  * market-relative: the coin's next-h-day log return minus the universe
    average that day — the model is not asked to call the market;
  * volatility-scaled: divided by the coin's own 30-day daily vol x sqrt(h),
    so calm and wild periods weigh alike;
  * overlap-weighted: consecutive h-day labels share h-1 days, and coins on
    the same day share the market, so each DAY carries total weight 1/h.

Per-coin features are z-scored across coins each day (robust to regime
level shifts); market-wide features (BTC trend, breadth, external series)
are added unscaled.
"""
from __future__ import annotations
import numpy as np

from ..engine import features as F

COIN_FEATURES = ("r7", "r30", "r90", "vol30", "sma_gap50", "sma_gap100", "sma_gap200",
                 "volume_ratio", "dd60", "r30_rank", "vol30_rank")
MARKET_FEATURES = ("btc_r30", "btc_sma_gap100", "breadth100")


def rolling_max(x, n):
    """(T, N) max over the last n rows (NaN until n rows, NaN-aware)."""
    from numpy.lib.stride_tricks import sliding_window_view
    T = x.shape[0]
    out = np.full_like(x, np.nan)
    if T >= n:
        w = sliding_window_view(np.where(np.isnan(x), -np.inf, x), n, axis=0)
        m = w.max(axis=-1)
        out[n - 1:] = np.where(np.isfinite(m), m, np.nan)
    return out


def _zscore_rows(x, mask):
    """Per-day z-score across the universe; a feature with no spread across
    coins that day carries no ranking information -> 0 (not NaN, which would
    drop every sample)."""
    x = np.where(mask, x, np.nan)
    if not x.size:
        return x
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 1e-12, (x - mu) / sd, np.where(np.isnan(x), np.nan, 0.0))
    return np.clip(z, -5, 5)


def coin_features(panel):
    c = panel.close
    lr = F.log_returns(c)
    v = np.where(np.isnan(c), np.nan, panel.volume)
    vol30 = F.rolling_std(lr, 30)
    r30 = F.ret_n(c, 30)
    f = {"r7": F.ret_n(c, 7), "r30": r30, "r90": F.ret_n(c, 90), "vol30": vol30,
         "sma_gap50": F.sma_gap(c, 50), "sma_gap100": F.sma_gap(c, 100),
         "sma_gap200": F.sma_gap(c, 200),
         "volume_ratio": F.rolling_mean(v, 7) / F.rolling_mean(v, 30),
         "dd60": c / rolling_max(c, 60) - 1,
         "r30_rank": F.cross_rank(r30), "vol30_rank": F.cross_rank(vol30)}
    return f


def market_features(panel, mask):
    T = panel.T
    out = {}
    if "BTC-USD" in panel.coins:
        b = panel.close[:, [panel.coins.index("BTC-USD")]]
        out["btc_r30"] = F.ret_n(b, 30)[:, 0]
        out["btc_sma_gap100"] = F.sma_gap(b, 100)[:, 0]
    else:
        out["btc_r30"] = np.full(T, np.nan)
        out["btc_sma_gap100"] = np.full(T, np.nan)
    above = np.where(mask, F.sma_gap(panel.close, 100) > 0, False)
    n = mask.sum(axis=1)
    out["breadth100"] = np.where(n > 0, above.sum(axis=1) / np.maximum(n, 1), np.nan)
    return out


def build(panel, mask, h=7, extra_market=None):
    """(X (T, N, F), y (T, N) label, w (T, N) weight, names, vol30).
    `mask` (T, N): the point-in-time universe. `extra_market`: {name: (T,)}
    market-wide series (e.g. series.daily_array outputs). Rows outside the
    mask, or without a known label / complete features, have weight 0."""
    cf = coin_features(panel)
    mf = market_features(panel, mask)
    for k, arr in (extra_market or {}).items():
        mf[k] = np.asarray(arr, dtype=float)
    names = list(COIN_FEATURES) + list(mf)
    T, N = panel.T, panel.N
    X = np.full((T, N, len(names)), np.nan)
    for i, k in enumerate(COIN_FEATURES):
        X[:, :, i] = _zscore_rows(cf[k], mask)
    for i, k in enumerate(mf, start=len(COIN_FEATURES)):
        X[:, :, i] = np.broadcast_to(np.asarray(mf[k], dtype=float)[:, None], (T, N))
    lr = F.log_returns(panel.close)
    fwd = np.full((T, N), np.nan)
    cs = np.nancumsum(np.nan_to_num(lr), axis=0)
    fwd[:-h] = cs[h:] - cs[:-h]
    full = np.zeros((T, N), bool)                 # every one of the h days has a bar
    present = ~np.isnan(panel.close)
    cnt = np.cumsum(present, axis=0)
    full[:-h] = (cnt[h:] - cnt[:-h]) == h
    fwd = np.where(full, fwd, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # days with no universe
        rel = fwd - np.nanmean(np.where(mask, fwd, np.nan), axis=1, keepdims=True)
    vol = cf["vol30"]
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.clip(rel / (vol * np.sqrt(h)), -5, 5)
    ok = mask & np.isfinite(y) & np.isfinite(X).all(axis=2)
    n_t = ok.sum(axis=1, keepdims=True)
    w = np.where(ok, 1.0 / (h * np.maximum(n_t, 1)), 0.0)
    return X, np.where(ok, y, np.nan), w, names, fwd


def meta_labels(panel, held, h=7):
    """Binary label for the trend meta-model: did a coin the trend rule holds
    at t gain over the next h days? NaN where not held / unknown."""
    lr = F.log_returns(panel.close)
    cs = np.nancumsum(np.nan_to_num(lr), axis=0)
    fwd = np.full(panel.close.shape, np.nan)
    fwd[:-h] = cs[h:] - cs[:-h]
    present = ~np.isnan(panel.close)
    cnt = np.cumsum(present, axis=0)
    full = np.zeros_like(present)
    full[:-h] = (cnt[h:] - cnt[:-h]) == h
    return np.where(held & full, (fwd > 0).astype(float), np.nan)


def series_z(x, n=365, min_obs=60):
    """Causal rolling z-score of a market-wide series (T,): level series like
    stablecoin supply are non-stationary, their deviation from the trailing
    year is not."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        w = x[max(0, t - n + 1):t + 1]
        w = w[np.isfinite(w)]
        if len(w) >= min_obs and np.isfinite(x[t]):
            sd = w.std()
            out[t] = (x[t] - w.mean()) / sd if sd > 0 else 0.0
    return out
