"""Fast feature precompute must be numerically identical to the original
per-bar _features_at (this is what makes the speedup safe)."""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backtest import composite
from app.backtest import fast_features as ff


def _make_candles(n, seed=0):
    rng = random.Random(seed)
    px = 100.0
    t = 0
    out = []
    for _ in range(n):
        t += 3600
        op = px
        px *= (1 + rng.gauss(0.0003, 0.012))
        cl = px
        hi = max(op, cl) * (1 + abs(rng.gauss(0, 0.004)))
        lo = min(op, cl) * (1 - abs(rng.gauss(0, 0.004)))
        out.append([t, lo, hi, op, cl, rng.uniform(80, 300)])
    return out


def test_precompute_matches_features_at_bitwise():
    candles = _make_candles(400, seed=3)
    feats = ff.precompute(candles)
    assert feats is not None
    checked = 0
    for i in range(60, len(candles)):
        want = composite._features_at(candles, i)
        got = feats[i]
        assert want is not None and got is not None
        assert set(want) == set(got)
        for k in want:
            a, b = want[k], got[k]
            # allow only floating-point epsilon differences
            assert abs(a - b) <= 1e-9 + 1e-9 * abs(a), f"bar {i} key {k}: {a} != {b}"
        checked += 1
    assert checked > 300


def test_pure_python_kernel_matches_features_at():
    # force the pure-Python path (independent of whether numba is installed)
    candles = _make_candles(250, seed=7)
    highs = [c[2] for c in candles]
    lows = [c[1] for c in candles]
    closes = [c[4] for c in candles]
    vols = [c[5] for c in candles]
    cols = ff._kernel_impl(highs, lows, closes, vols, len(candles))
    names = ("price", "rsi", "atr", "sma20", "sma50", "ema12", "ema26", "macd",
             "macd_delta", "volatility", "hi20", "lo20", "vol_ratio",
             "mom_1h", "mom_4h")
    for i in range(60, len(candles)):
        want = composite._features_at(candles, i)
        for name, arr in zip(names, cols):
            assert abs(want[name] - arr[i]) <= 1e-9 + 1e-9 * abs(want[name]), \
                f"bar {i} {name}"


def test_short_history_returns_none():
    assert ff.precompute(_make_candles(40)) is None


def test_run_composite_unchanged_by_fast_path():
    """The end-to-end backtest result must be identical whether or not the fast
    feature path is used."""
    candles = _make_candles(500, seed=11)
    fast = composite.run_composite(candles)
    slow = composite.run_composite(candles, _use_fast_features=False)
    for k in ("total_return", "sharpe_annualized", "sortino_annualized",
              "max_drawdown", "n_trades", "win_rate", "profit_factor",
              "calmar", "exposure", "final_equity"):
        assert fast[k] == slow[k], f"{k}: {fast[k]} != {slow[k]}"
    assert fast["trades"] == slow["trades"]
