"""Tests for the swing-ATR sizing horizon fix.

The bug: stops/targets were sized off the native 5m ATR, so a trade's target
could never honestly clear a ~3% round-trip cost — every trade was a doomed
scalp. The fix sizes off a HIGHER-TIMEFRAME ATR (default 1h = 12x5m bars),
making it a genuine swing strategy. This is a horizon change, not looser fees.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.market import MarketData


def _trending_bars(n=200, start=100.0, step=0.5, rng=0.3):
    """Build [ts, low, high, open, close, volume] bars on a steady uptrend so
    that a higher-timeframe bar spans a visibly larger range than one 5m bar."""
    bars = []
    c = start
    for i in range(n):
        o = c
        c = o + step
        hi = max(o, c) + rng
        lo = min(o, c) - rng
        bars.append([i * 300, lo, hi, o, c, 1000.0])
    return bars


def test_swing_atr_wider_than_native_on_trend():
    highs = [b[2] for b in _trending_bars()]
    lows = [b[1] for b in _trending_bars()]
    closes = [b[4] for b in _trending_bars()]
    native = MarketData._swing_atr(highs, lows, closes, period=14)  # default 12x
    # force native by making the helper read swing_atr_bars=1
    from app.tunables import update as tupdate
    tupdate({"swing_atr_bars": 1})
    native_1 = MarketData._swing_atr(highs, lows, closes, period=14)
    tupdate({"swing_atr_bars": 12})
    swing_12 = MarketData._swing_atr(highs, lows, closes, period=14)
    # a 1h bar accumulates ~12x the drift of a 5m bar → materially wider ATR
    assert swing_12 > native_1 * 2, (swing_12, native_1)


def test_swing_atr_bars_one_equals_native():
    bars = _trending_bars()
    highs = [b[2] for b in bars]; lows = [b[1] for b in bars]
    closes = [b[4] for b in bars]
    from app.tunables import update as tupdate
    tupdate({"swing_atr_bars": 1})
    try:
        got = MarketData._swing_atr(highs, lows, closes, period=14)
        # recompute native ATR(14) directly
        trs = []
        for i in range(-14, 0):
            trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
        native = sum(trs) / 14
        assert abs(got - native) < 1e-9
    finally:
        tupdate({"swing_atr_bars": 12})


def test_features_exposes_swing_atr(monkeypatch):
    from app.data.market import market
    bars = _trending_bars(n=200)
    market.candles["BTC-USD"] = bars
    market.books["BTC-USD"] = {}
    f = market.features("BTC-USD")
    assert f is not None
    assert "atr_swing" in f and "atr" in f
    # on this uptrend the swing ATR must be wider than the native 5m ATR
    assert f["atr_swing"] > f["atr"]


def test_swing_horizon_lets_target_clear_cost_gate():
    """With the swing ATR, a normal trade's take-profit clears the round-trip
    cost gate; with the tiny native 5m ATR it would be rejected (notional=0)."""
    from app.risk.manager import risk
    from app.tunables import tv
    price = 100.0
    # native 5m ATR ~ 0.3 (one bar range); swing 1h ATR ~ 3.6 (12x drift+range)
    native_atr = 0.3
    swing_atr = 3.6
    rs = {"effective_risk_scale": 1.0}
    # native: 3x ATR target = 0.9 => 0.9% move; round-trip 2*(0.5%+10bps)=1.2%,
    # *cost_multiple 2.5 => needs 3% target distance -> REJECTED
    n_native, _, _ = risk.size(100_000, price, native_atr, 0.8, rs, product="BTC-USD")
    assert n_native == 0, "native 5m ATR target should fail the cost gate"
    # swing: 3x ATR target = 10.8 => 10.8% move -> clears the gate
    n_swing, stop, take = risk.size(100_000, price, swing_atr, 0.8, rs, product="BTC-USD")
    assert n_swing > 0, "swing ATR target should clear the cost gate"
    assert take > price and stop < price
