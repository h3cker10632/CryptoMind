"""Tier-2 multi-timeframe feature tests."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.market import MarketData
from app.learn.online_model import build_x, N_IN, FEAT_NAMES


def _synth(n, step):
    """n candles [ts, low, high, open, close, vol] with a constant per-bar drift."""
    cs = []
    px = 100.0
    for i in range(n):
        o = px
        px = px * (1 + step)
        lo, hi = min(o, px) * 0.999, max(o, px) * 1.001
        cs.append([i * 300, lo, hi, o, px, 10.0])
    return cs


def test_mtf_align_bullish_uptrend():
    m = MarketData()
    m.candles["BTC-USD"] = _synth(300, +0.002)   # steady uptrend
    f = m.features("BTC-USD")
    assert f is not None
    assert f["mtf_align"] > 0.5                   # timeframes agree bullish
    assert f["mtf_trend_1h"] == 1.0


def test_mtf_align_bearish_downtrend():
    m = MarketData()
    m.candles["BTC-USD"] = _synth(300, -0.002)
    f = m.features("BTC-USD")
    assert f["mtf_align"] < -0.5
    assert f["mtf_trend_1h"] == -1.0


def test_build_x_includes_mtf_and_matches_dims():
    m = MarketData()
    m.candles["BTC-USD"] = _synth(300, +0.002)
    f = m.features("BTC-USD")
    x = build_x(f, 0.0, 0.0, None)
    assert len(x) == N_IN == len(FEAT_NAMES)
    assert FEAT_NAMES[-1] == "mtf_align"
    assert -1.0 <= x[-1] <= 1.0
    assert x[-1] > 0                              # uptrend -> positive alignment


def test_mtf_present_even_with_short_history():
    m = MarketData()
    m.candles["BTC-USD"] = _synth(70, +0.001)    # enough for features, thin HTF
    f = m.features("BTC-USD")
    assert f is not None
    for k in ("mtf_align", "mtf_trend_15m", "mtf_trend_1h", "mtf_trend_4h"):
        assert k in f
        assert -1.0 <= f[k] <= 1.0
