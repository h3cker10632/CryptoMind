"""Shared OHLCV feature core — the single source of truth for the technical
features derived purely from a candle history.

Before this module the same formulas (RSI-14, ATR-14, MACD + delta, SMA20/50,
EMA12/26, 60-bar log-return volatility, 20-bar high/low, volume ratio, momentum)
were hand-duplicated in THREE places:

  * app/data/market.py         `Market.features()`      (live path)
  * app/backtest/composite.py  `_features_at()`         (backtest path)
  * app/learn/evolution.py     `_precompute_indicators` (GA, per-genome periods)

The live and backtest copies MUST agree or train/live parity silently breaks —
a genome validated on one feature definition would trade on another. This helper
computes that shared core once so both callers delegate to identical arithmetic.
(The GA copy uses per-genome EMA/breakout periods and a NumPy vectorization, so
it stays separate by design — it is a different, parameterized computation.)

`features_from_ohlcv` takes the trailing OHLCV lists and returns exactly the
OHLCV-derivable keys. Live-only additions (atr_swing, multi-timeframe, order-book
imbalance/spread) are layered on TOP by the caller, not here.
"""
from __future__ import annotations
import math

MIN_BARS = 60          # warmup: features need at least this much history


def stdev(xs, ddof=1):
    """Standard deviation (sample by default; ddof=0 for population) with
    two-pass `math.fsum` sums. Within 1 ulp of `statistics.stdev`, which does
    exact rational arithmetic and was ~40% of a cold replay's run time (23x
    slower per call). tests/test_speed_parity.py checks both agree."""
    n = len(xs)
    m = math.fsum(xs) / n
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - ddof))


def _sma(xs, n):
    return sum(xs[-n:]) / n


def _ema(xs, n):
    """EMA over the trailing n samples, reseeded at the window start (matches
    the historical hand-written loops in market.py / composite.py exactly)."""
    k = 2 / (n + 1)
    e = xs[-n]
    for x in xs[-n + 1:]:
        e = x * k + e * (1 - k)
    return e


def features_from_ohlcv(closes, highs, lows, vols):
    """Compute the shared technical-feature core from trailing OHLCV lists.

    Returns None if there is less than MIN_BARS of history (the caller then
    treats the bar as warmup). The output dict's keys and arithmetic are the
    canonical definitions used by BOTH the live market feed and the backtester.
    `imbalance` / `spread_bps` are returned as 0.0 placeholders (no order book
    in history); the live caller overwrites them from its book.
    """
    if len(closes) < MIN_BARS:
        return None
    price = closes[-1]

    # RSI(14)
    gains, losses = [], []
    for a, b in zip(closes[-15:-1], closes[-14:]):
        d = b - a
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    avg_g, avg_l = sum(gains) / 14, sum(losses) / 14
    rsi = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)

    # ATR(14) on the native bar
    trs = []
    for i in range(-14, 0):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    atr = sum(trs) / 14

    # MACD (12/26) + one-bar delta
    macd = _ema(closes, 12) - _ema(closes, 26)
    macd_prev = _ema(closes[:-1], 12) - _ema(closes[:-1], 26)

    # 60-bar log-return volatility (sample stdev)
    rets = [math.log(b / a) for a, b in zip(closes[-61:-1], closes[-60:])]
    vol = stdev(rets) if len(rets) > 2 else 0.0

    hi20, lo20 = max(highs[-20:]), min(lows[-20:])
    vol_ratio = vols[-1] / (sum(vols[-20:]) / 20) if sum(vols[-20:]) else 1.0

    # ---- slow-horizon features (multi-hour / multi-day holds) ----
    # On 1h bars these are: EMA 24h vs 96h trend, 72h momentum, and the prior
    # 48h high/low channel (excluding the current bar, so a close above it is a
    # genuine breakout). These are the families that showed an after-cost edge
    # on the hourly backtest; absent (None) on short histories.
    n = len(closes)
    ema24 = _ema(closes, 24) if n >= 24 else None
    ema96 = _ema(closes, 96) if n >= 96 else None
    mom_72 = price / closes[-73] - 1 if n >= 73 else None
    hi48 = max(highs[-49:-1]) if n >= 49 else None
    lo48 = min(lows[-49:-1]) if n >= 49 else None

    return {
        "ema24": ema24, "ema96": ema96, "mom_72": mom_72,
        "hi48": hi48, "lo48": lo48,
        "price": price, "rsi": rsi, "atr": atr,
        "sma20": _sma(closes, 20), "sma50": _sma(closes, 50),
        "ema12": _ema(closes, 12), "ema26": _ema(closes, 26),
        "macd": macd, "macd_delta": macd - macd_prev,
        "volatility": vol, "hi20": hi20, "lo20": lo20, "vol_ratio": vol_ratio,
        "mom_1h": price / closes[-13] - 1 if len(closes) >= 13 else 0,
        "mom_4h": price / closes[-49] - 1 if len(closes) >= 49 else 0,
        "imbalance": 0.0, "spread_bps": 0.0,
    }
