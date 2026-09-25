"""Chart-pattern engine: detectors fire with the right sign, the report shape is
stable, and the ML input vector stays width-consistent (N_IN) whether or not a
pattern report is present."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.signals import patterns as P
from app.signals.patterns import analyze, feature_vector, PATTERN_FEATURES


def _ohlc(closes, opens=None):
    highs = [c * 1.003 for c in closes]
    lows = [c * 0.997 for c in closes]
    return highs, lows, closes, opens


# ---------------- report shape ----------------

def test_report_shape_and_ranges():
    closes = [100 + i * 0.5 for i in range(80)]
    h, l, c, _ = _ohlc(closes)
    r = analyze(h, l, c)
    assert set(r) == {"lean", "features", "detected", "structure"}
    assert -1.0 <= r["lean"] <= 1.0
    assert list(r["features"].keys()) == PATTERN_FEATURES
    assert all(-1.0 <= v <= 1.0 for v in r["features"].values())


def test_short_history_is_neutral():
    r = analyze([1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert r["lean"] == 0.0
    assert all(v == 0.0 for v in r["features"].values())
    assert r["structure"] == "unknown"


def test_never_raises_on_garbage():
    r = analyze([], [], [])
    assert r["lean"] == 0.0
    r2 = analyze([1] * 50, [1] * 50, [1] * 40)   # mismatched lengths
    assert r2["lean"] == 0.0


def test_feature_vector_length_and_default():
    assert len(feature_vector(None)) == len(PATTERN_FEATURES)
    assert feature_vector(None) == [0.0] * len(PATTERN_FEATURES)
    closes = [100 + i * 0.5 for i in range(80)]
    h, l, c, _ = _ohlc(closes)
    assert len(feature_vector(analyze(h, l, c))) == len(PATTERN_FEATURES)


# ---------------- market structure ----------------

def test_uptrend_structure_positive():
    closes = [100 + i * 0.6 + 3 * math.sin(i / 3) for i in range(80)]
    h, l, c, _ = _ohlc(closes)
    r = analyze(h, l, c)
    assert r["features"]["pat_structure"] > 0
    assert r["structure"] == "uptrend"


def test_downtrend_structure_negative():
    closes = [160 - i * 0.6 + 3 * math.sin(i / 3) for i in range(80)]
    h, l, c, _ = _ohlc(closes)
    r = analyze(h, l, c)
    assert r["features"]["pat_structure"] < 0
    assert r["structure"] == "downtrend"


# ---------------- reversal patterns ----------------

def test_double_bottom_is_bullish():
    db = []
    for i in range(20):
        db.append(120 - i)             # decline to 100
    for i in range(15):
        db.append(100 + i * 0.8)       # bounce
    for i in range(15):
        db.append(112 - i * 0.8)       # retest the low
    for i in range(20):
        db.append(100 + i * 1.2)       # rally / confirm
    h, l, c, _ = _ohlc(db)
    r = analyze(h, l, c)
    assert r["features"]["pat_reversal"] > 0
    assert any("bottom" in d["name"].lower() for d in r["detected"])


def test_double_top_is_bearish():
    dt = []
    for i in range(20):
        dt.append(100 + i)             # rally to 120
    for i in range(15):
        dt.append(120 - i * 0.8)       # pullback
    for i in range(15):
        dt.append(108 + i * 0.8)       # retest the high
    for i in range(20):
        dt.append(120 - i * 1.2)       # decline / confirm
    h, l, c, _ = _ohlc(dt)
    r = analyze(h, l, c)
    assert r["features"]["pat_reversal"] < 0
    assert any("top" in d["name"].lower() for d in r["detected"])


# ---------------- candlesticks (direct, deterministic) ----------------

def test_bullish_engulfing():
    closes = [100.0] * 60
    opens = [100.0] * 60
    highs = [100.5] * 60
    lows = [99.5] * 60
    # prev bar down, current bar up and engulfing it
    opens[-2], closes[-2] = 101.0, 99.0
    opens[-1], closes[-1] = 98.5, 101.5
    highs[-2], lows[-2] = 101.2, 98.8
    highs[-1], lows[-1] = 101.7, 98.3
    r = analyze(highs, lows, closes, opens=opens)
    assert r["features"]["pat_candle"] > 0
    assert any("engulf" in d["name"].lower() for d in r["detected"])


def test_bearish_engulfing():
    closes = [100.0] * 60
    opens = [100.0] * 60
    highs = [100.5] * 60
    lows = [99.5] * 60
    opens[-2], closes[-2] = 99.0, 101.0     # prev up
    opens[-1], closes[-1] = 101.5, 98.5     # curr down, engulfs
    highs[-2], lows[-2] = 101.2, 98.8
    highs[-1], lows[-1] = 101.7, 98.3
    r = analyze(highs, lows, closes, opens=opens)
    assert r["features"]["pat_candle"] < 0
    assert any("engulf" in d["name"].lower() for d in r["detected"])


# ---------------- divergence (direct) ----------------

def test_bullish_divergence_direct():
    # price makes a lower low while RSI makes a higher low
    pl = [(10, 100.0), (30, 98.0)]      # (idx, price): lower low
    ph = []
    rsi = [50.0] * 40
    rsi[10], rsi[30] = 25.0, 35.0       # higher RSI low
    score, det = P._detect_divergence([0] * 40, rsi, pl, ph)
    assert score > 0
    assert any("bullish" in d["direction"] for d in det)


def test_bearish_divergence_direct():
    ph = [(10, 100.0), (30, 102.0)]     # higher high
    pl = []
    rsi = [50.0] * 40
    rsi[10], rsi[30] = 75.0, 65.0       # lower RSI high
    score, det = P._detect_divergence([0] * 40, rsi, pl, ph)
    assert score < 0
    assert any("bearish" in d["direction"] for d in det)


# ---------------- integration with engine + ML vector ----------------

def test_pattern_sleeve_registered():
    from app.signals.engine import STRATEGIES, strat_pattern
    assert "pattern" in STRATEGIES
    # reads the lean off the features dict; 0.0 when absent
    assert strat_pattern({}, 0.0, {}) == 0.0
    assert strat_pattern({"patterns": {"lean": 0.8}}, 0.0, {}) == 0.8
    # clamps
    assert strat_pattern({"patterns": {"lean": 5.0}}, 0.0, {}) == 1.0


def test_ml_vector_width_matches_n_in():
    from app.learn.online_model import build_x, N_IN, FEAT_NAMES
    assert N_IN == len(FEAT_NAMES) == 24
    f = {"rsi": 55, "macd": 0.1, "macd_delta": 0.02, "mom_1h": 0.001,
         "mom_4h": 0.002, "vol_ratio": 1.1, "imbalance": 0.1, "spread_bps": 3,
         "price": 100, "sma20": 99, "sma50": 98, "volatility": 0.003, "atr": 0.5}
    # no pattern report -> zeros appended, still N_IN wide
    assert len(build_x(f, 0.2, 0.1, None)) == N_IN
    # with a pattern report -> still N_IN wide, pattern slots populated
    f2 = dict(f, patterns={"features": {k: 0.5 for k in PATTERN_FEATURES}})
    x = build_x(f2, 0.2, 0.1, None)
    assert len(x) == N_IN
    assert x[-len(PATTERN_FEATURES):] == [0.5] * len(PATTERN_FEATURES)
