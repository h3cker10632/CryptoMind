"""Candle-derived market-regime classification (no lookahead).

The GA's strategies are single-timeframe trend-following (EMA cross + breakout +
RSI). The measured failure of the flat market-structure genes was that forcing
ONE genome to fit a ~100-day window spanning mixed regimes bleeds edge. Before
building per-regime evolution we must first MEASURE whether the edge is actually
regime-dependent — i.e. whether these genomes make money in trending stretches
and lose it in choppy/ranging stretches.

Regime label for bar i uses ONLY bars up to and including i (strictly causal):

  Kaufman Efficiency Ratio (ER) over the trailing n bars:
      ER[i] = |close[i] - close[i-n]| / sum_{j=i-n+1..i} |close[j] - close[j-1]|
  ER -> 1  : price moved in a straight line  (TREND)
  ER -> 0  : lots of motion, no net progress (RANGE / chop)

A single threshold splits TREND vs RANGE. This is deliberately simple and cheap;
it is a measurement instrument first, and only becomes a router if the data
justifies it.
"""

TREND = "trend"
RANGE = "range"

try:
    import numpy as _np
except Exception:                       # numpy optional — pure-Python fallback
    _np = None


def efficiency_ratio(closes, n):
    """Trailing Kaufman efficiency ratio per bar (0..1); 0.0 for warmup bars.

    Vectorized with NumPy (cumulative-sum rolling window) when available — the
    GA calls this once per genome, and it was one of the pure-Python hot spots
    in the fitness loop. The NumPy path is numerically identical to the running-
    window arithmetic below to floating-point epsilon (verified in tests), so
    regime labels, champions and promotions are unchanged. Falls back to the
    original per-bar loop when NumPy is missing or history is too short.
    """
    m = len(closes)
    if n < 1:
        return [0.0] * m
    if _np is not None and m > n:
        c = _np.asarray(closes, dtype=float)
        ad = _np.empty(m)
        ad[0] = 0.0
        ad[1:] = _np.abs(_np.diff(c))
        cs = _np.concatenate(([0.0], _np.cumsum(ad)))     # cs[k] = sum(ad[:k])
        out = _np.zeros(m)
        idx = _np.arange(n, m)
        win = cs[idx + 1] - cs[idx - n + 1]               # sum(ad[i-n+1 .. i])
        num = _np.abs(c[idx] - c[idx - n])
        with _np.errstate(divide="ignore", invalid="ignore"):
            r = num / win
        r[win <= 0] = 0.0
        out[idx] = r
        return out.tolist()
    # pure-Python fallback (original running-window implementation)
    out = [0.0] * m
    abs_delta = [0.0] + [abs(closes[j] - closes[j - 1]) for j in range(1, m)]
    win = 0.0
    for i in range(m):
        win += abs_delta[i]
        if i - n >= 0:
            win -= abs_delta[i - n]
        if i >= n and win > 0:
            out[i] = abs(closes[i] - closes[i - n]) / win
    return out


def classify_regimes(candles, n=48, thresh=0.35):
    """Return a per-bar list of TREND / RANGE labels (causal).

    n ~ two days of hourly bars; thresh 0.35 is a neutral default (ER above it =
    directional enough to call a trend). Warmup bars (< n) are labelled RANGE
    (conservative: no confirmed trend yet).
    """
    closes = [c[4] for c in candles]
    er = efficiency_ratio(closes, n)
    return [TREND if (i >= n and er[i] >= thresh) else RANGE
            for i in range(len(closes))], er
