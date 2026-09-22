"""Shared OHLCV feature core: the live feed (market.features) and the backtester
(_features_at) must compute identical technical features — train/live parity is
enforced by both delegating to features_from_ohlcv, not by hand-synced copies."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data.features import features_from_ohlcv, MIN_BARS
from app.backtest.composite import _features_at
from app.data.market import market


# the technical keys that MUST match live vs backtest (book fields legitimately
# differ: live has an order book, backtest uses 0.0 placeholders)
SHARED = ("price", "rsi", "atr", "sma20", "sma50", "ema12", "ema26", "macd",
          "macd_delta", "volatility", "hi20", "lo20", "vol_ratio",
          "mom_1h", "mom_4h")


def _candles(n, seed=0):
    rng = random.Random(seed)
    px = 100.0
    t = 0
    out = []
    for _ in range(n):
        t += 300
        op = px
        px *= (1 + rng.gauss(0.0003, 0.011))
        cl = px
        hi = max(op, cl) * (1 + abs(rng.gauss(0, 0.003)))
        lo = min(op, cl) * (1 - abs(rng.gauss(0, 0.003)))
        out.append([t, lo, hi, op, cl, rng.uniform(100, 200)])
    return out


def test_helper_returns_none_below_min_bars():
    c = _candles(MIN_BARS - 1)
    assert features_from_ohlcv([x[4] for x in c], [x[2] for x in c],
                               [x[1] for x in c], [x[5] for x in c]) is None


def test_live_and_backtest_agree_on_shared_keys():
    candles = _candles(300, seed=5)
    market.candles["PARITYTEST-USD"] = candles
    market.books["PARITYTEST-USD"] = {"imbalance": 0.42, "spread_bps": 3.1}
    live = market.features("PARITYTEST-USD")
    bt = _features_at(candles, len(candles) - 1)
    for k in SHARED:
        assert abs(live[k] - bt[k]) <= 1e-12, f"{k}: {live[k]} != {bt[k]}"


def test_live_layers_extras_backtest_does_not():
    candles = _candles(300, seed=6)
    market.candles["PARITYTEST2-USD"] = candles
    market.books["PARITYTEST2-USD"] = {"imbalance": -0.2, "spread_bps": 5.0}
    live = market.features("PARITYTEST2-USD")
    bt = _features_at(candles, len(candles) - 1)
    # live-only extras exist on the live feed
    for k in ("atr_swing", "mtf_align", "mtf_trend_1h", "mtf_rsi_1h"):
        assert k in live
        assert k not in bt
    # book fields: live from the book, backtest placeholder 0.0
    assert live["imbalance"] == -0.2 and live["spread_bps"] == 5.0
    assert bt["imbalance"] == 0.0 and bt["spread_bps"] == 0.0


def test_helper_is_pure_function_of_inputs():
    candles = _candles(200, seed=8)
    closes = [c[4] for c in candles]; highs = [c[2] for c in candles]
    lows = [c[1] for c in candles]; vols = [c[5] for c in candles]
    a = features_from_ohlcv(closes, highs, lows, vols)
    b = features_from_ohlcv(closes, highs, lows, vols)
    assert a == b
