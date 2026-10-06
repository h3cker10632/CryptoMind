"""Fast per-bar feature precompute for the composite backtester.

`composite._features_at(candles, i)` rebuilt the full OHLCV arrays from a fresh
`candles[:i+1]` prefix copy on EVERY bar — O(N^2) — even though every feature it
computes reads only a TRAILING window (<=61 bars). This module computes the same
features for all bars in ONE pass over the price columns:

  * the numeric work is a Numba-JIT kernel (`_kernel`) when numba is installed;
  * otherwise it falls back to an identical pure-Python kernel.

Either way the output is verified bit-identical (to fp epsilon) against the
original `_features_at`, so trade decisions, promoted champions and reported
metrics are unchanged — this is a pure speedup. See tests/test_fast_features.py.
"""
from __future__ import annotations
import math

try:
    import numpy as _np
except Exception:                       # numpy optional
    _np = None

# ---- optional Numba JIT -------------------------------------------------
_USE_NUMBA = False
try:
    from numba import njit as _njit      # type: ignore
    _USE_NUMBA = True
except Exception:                        # numba optional -> pure-Python kernel
    def _njit(*a, **k):                  # no-op decorator fallback
        def wrap(fn):
            return fn
        return wrap if not (a and callable(a[0])) else a[0]


def _kernel_impl(highs, lows, closes, vols, n):
    """Compute feature arrays for bars [60, n). Written in a Numba-friendly
    style (typed scalars, index loops, no Python objects) so the SAME source
    runs JIT-compiled when numba is present and interpreted otherwise.

    Returns parallel arrays; index i is valid for i in [60, n)."""
    price = [0.0] * n
    rsi = [0.0] * n
    atr = [0.0] * n
    sma20 = [0.0] * n
    sma50 = [0.0] * n
    ema12 = [0.0] * n
    ema26 = [0.0] * n
    macd = [0.0] * n
    macd_delta = [0.0] * n
    volatility = [0.0] * n
    hi20 = [0.0] * n
    lo20 = [0.0] * n
    vol_ratio = [0.0] * n
    mom_1h = [0.0] * n
    mom_4h = [0.0] * n

    k12 = 2.0 / 13.0
    k26 = 2.0 / 27.0

    for i in range(60, n):
        price[i] = closes[i]

        # RSI(14): 14 diffs over closes[i-13 .. i]
        ag = 0.0
        al = 0.0
        for j in range(i - 13, i + 1):
            d = closes[j] - closes[j - 1]
            if d > 0:
                ag += d
            else:
                al += -d
        ag /= 14.0
        al /= 14.0
        rsi[i] = 100.0 if al == 0.0 else 100.0 - 100.0 / (1.0 + ag / al)

        # ATR(14): TR over bars i-13 .. i, prev close j-1
        s_tr = 0.0
        for j in range(i - 13, i + 1):
            hlo = highs[j] - lows[j]
            hpc = highs[j] - closes[j - 1]
            if hpc < 0:
                hpc = -hpc
            lpc = lows[j] - closes[j - 1]
            if lpc < 0:
                lpc = -lpc
            tr = hlo
            if hpc > tr:
                tr = hpc
            if lpc > tr:
                tr = lpc
            s_tr += tr
        atr[i] = s_tr / 14.0

        # windowed EMA(12)/EMA(26): reseed at start of the trailing window
        e12 = closes[i - 11]
        for j in range(i - 10, i + 1):
            e12 = closes[j] * k12 + e12 * (1.0 - k12)
        e26 = closes[i - 25]
        for j in range(i - 24, i + 1):
            e26 = closes[j] * k26 + e26 * (1.0 - k26)
        ema12[i] = e12
        ema26[i] = e26
        macd[i] = e12 - e26

        # macd_prev over closes[:i] (window ends at i-1)
        e12p = closes[i - 12]
        for j in range(i - 11, i):
            e12p = closes[j] * k12 + e12p * (1.0 - k12)
        e26p = closes[i - 26]
        for j in range(i - 25, i):
            e26p = closes[j] * k26 + e26p * (1.0 - k26)
        macd_delta[i] = macd[i] - (e12p - e26p)

        # SMA20 / SMA50
        s20 = 0.0
        for j in range(i - 19, i + 1):
            s20 += closes[j]
        sma20[i] = s20 / 20.0
        s50 = 0.0
        for j in range(i - 49, i + 1):
            s50 += closes[j]
        sma50[i] = s50 / 50.0

        # volatility: sample stdev of 60 log returns over bars i-59 .. i
        m = 0.0
        lr = [0.0] * 60
        for w in range(60):
            j = i - 59 + w
            v = math.log(closes[j] / closes[j - 1])
            lr[w] = v
            m += v
        m /= 60.0
        var = 0.0
        for w in range(60):
            dv = lr[w] - m
            var += dv * dv
        volatility[i] = math.sqrt(var / 59.0)      # sample stdev (n-1)

        # 20-bar high/low and volume ratio
        h = highs[i - 19]
        lo = lows[i - 19]
        sv = 0.0
        for j in range(i - 19, i + 1):
            if highs[j] > h:
                h = highs[j]
            if lows[j] < lo:
                lo = lows[j]
            sv += vols[j]
        hi20[i] = h
        lo20[i] = lo
        vol_ratio[i] = vols[i] / (sv / 20.0) if sv != 0.0 else 1.0

        mom_1h[i] = price[i] / closes[i - 12] - 1.0
        mom_4h[i] = price[i] / closes[i - 48] - 1.0

    return (price, rsi, atr, sma20, sma50, ema12, ema26, macd, macd_delta,
            volatility, hi20, lo20, vol_ratio, mom_1h, mom_4h)


# JIT-compiled variant (compiled lazily on first call; falls back on any error)
_kernel_jit = _njit(cache=True, fastmath=False)(_kernel_impl) if _USE_NUMBA else None


def precompute(candles):
    """Return a list `feats` where feats[i] is the feature dict for bar i
    (None for warmup bars i < 60), identical to _features_at(candles, i).

    Uses the Numba kernel when available (with numpy arrays), else the pure-
    Python kernel. Returns None if the history is too short to be meaningful,
    signalling the caller to fall back to per-bar computation.
    """
    n = len(candles)
    if n < 61:
        return None

    highs = [c[2] for c in candles]
    lows = [c[1] for c in candles]
    closes = [c[4] for c in candles]
    vols = [c[5] for c in candles]

    cols = None
    if _USE_NUMBA and _np is not None and _kernel_jit is not None:
        try:
            cols = _kernel_jit(_np.asarray(highs), _np.asarray(lows),
                               _np.asarray(closes), _np.asarray(vols), n)
        except Exception:
            cols = None
    if cols is None:
        cols = _kernel_impl(highs, lows, closes, vols, n)

    (price, rsi, atr, sma20, sma50, ema12, ema26, macd, macd_delta,
     volatility, hi20, lo20, vol_ratio, mom_1h, mom_4h) = cols

    feats = [None] * n
    for i in range(60, n):
        feats[i] = {
            "price": float(price[i]), "rsi": float(rsi[i]), "atr": float(atr[i]),
            "sma20": float(sma20[i]), "sma50": float(sma50[i]),
            "ema12": float(ema12[i]), "ema26": float(ema26[i]),
            "macd": float(macd[i]), "macd_delta": float(macd_delta[i]),
            "volatility": float(volatility[i]),
            "hi20": float(hi20[i]), "lo20": float(lo20[i]),
            "vol_ratio": float(vol_ratio[i]),
            "mom_1h": float(mom_1h[i]), "mom_4h": float(mom_4h[i]),
            "imbalance": 0.0, "spread_bps": 0.0,
        }
        feats[i].update(_slow_feats(highs, lows, closes, i))
    return feats


def _slow_feats(highs, lows, closes, i):
    """Slow-horizon keys (ema24/ema96/mom_72/hi48/lo48) for bar i, computed
    with the SAME windowed arithmetic as features_from_ohlcv (it reseeds the
    EMA at the window start, so passing just the trailing window is exact)."""
    from ..data.features import _ema
    n = i + 1
    return {
        "ema24": _ema(closes[i - 23:n], 24) if n >= 24 else None,
        "ema96": _ema(closes[i - 95:n], 96) if n >= 96 else None,
        "mom_72": closes[i] / closes[i - 72] - 1 if n >= 73 else None,
        "hi48": max(highs[i - 48:i]) if n >= 49 else None,
        "lo48": min(lows[i - 48:i]) if n >= 49 else None,
    }
