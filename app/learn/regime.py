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


def efficiency_ratio(closes, n):
    """Trailing Kaufman efficiency ratio per bar (0..1); 0.0 for warmup bars."""
    out = [0.0] * len(closes)
    if n < 1:
        return out
    # rolling sum of |delta| via a running window
    abs_delta = [0.0] + [abs(closes[j] - closes[j - 1]) for j in range(1, len(closes))]
    win = 0.0
    for i in range(len(closes)):
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
