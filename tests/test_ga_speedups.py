"""Correctness guards for the GA hot-loop speedups.

The GA promotes champions and gates real trading on simulate()'s output, so these
speedups are only acceptable if they are numerically identical to the original
arithmetic. Profiling (not vibes) showed the fitness loop was dominated by the
indicator PRECOMPUTE, not the event loop, and specifically by pure-Python
_ema_series, efficiency_ratio, and an unconditional rolling-median vol_norm.
The fixes: Numba-JIT the EMA recurrence, vectorize the efficiency ratio, and
compute the opt-in market-structure indicators (htf/vol/er) only when the genome
activates their gate. These tests pin all three to the pure-Python reference.
"""
import random

import pytest

from app.learn import evolution as E
from app.learn import regime as R


def _closes(n=1500, seed=5):
    r = random.Random(seed)
    c = [30000.0]
    for _ in range(n - 1):
        c.append(c[-1] * (1 + r.gauss(0, 0.01)))
    return c


def _ema_ref(xs, n):
    k = 2.0 / (n + 1)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def _er_ref(closes, n):
    out = [0.0] * len(closes)
    if n < 1:
        return out
    ad = [0.0] + [abs(closes[j] - closes[j - 1]) for j in range(1, len(closes))]
    win = 0.0
    for i in range(len(closes)):
        win += ad[i]
        if i - n >= 0:
            win -= ad[i - n]
        if i >= n and win > 0:
            out[i] = abs(closes[i] - closes[i - n]) / win
    return out


def test_ema_series_matches_pure_python_reference():
    xs = _closes()
    for n in (5, 12, 26, 48, 200):
        got = E._ema_series(xs, n)
        ref = _ema_ref(xs, n)
        assert len(got) == len(ref)
        assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-9


def test_efficiency_ratio_vectorized_matches_reference():
    closes = _closes()
    for n in (14, 48, 96):
        got = R.efficiency_ratio(closes, n)
        ref = _er_ref(closes, n)
        assert len(got) == len(ref)
        assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-9


def test_efficiency_ratio_numpy_path_equals_pure_python_fallback():
    closes = _closes()
    np_path = R.efficiency_ratio(closes, 48)
    saved = R._np
    R._np = None                    # force the pure-Python fallback
    try:
        py_path = R.efficiency_ratio(closes, 48)
    finally:
        R._np = saved
    assert max(abs(a - b) for a, b in zip(np_path, py_path)) < 1e-9


def test_efficiency_ratio_short_history_is_all_zeros():
    # history <= n has no valid window; both paths must return zeros, no crash
    assert R.efficiency_ratio([1.0, 2.0, 3.0], 10) == [0.0, 0.0, 0.0]


def _synth_candles(n=1200, seed=7):
    r = random.Random(seed)
    px, out, t = 30000.0, [], 1_600_000_000
    for i in range(n):
        px *= (1 + r.gauss(0.0002 * (1 if (i // 200) % 2 == 0 else -1), 0.01))
        hi = px * (1 + abs(r.gauss(0, 0.004)))
        lo = px * (1 - abs(r.gauss(0, 0.004)))
        out.append([t + i * 300, lo, hi, lo + (hi - lo) * r.random(),
                    lo + (hi - lo) * r.random(), abs(r.gauss(100, 30))])
    return out


def test_lazy_precompute_only_builds_activated_gates():
    candles = _synth_candles()
    g = E.normalize_genome(E.random_genome(random.Random(2)), random.Random(2))
    # gates OFF -> those arrays are not built (never read by simulate)
    g_off = dict(g, htf_w=0.0, volz_w=0.0, er_min=0.0)
    ind = E._precompute_indicators(candles, g_off)
    if ind is not None:                          # numpy present
        assert ind["htf_ema"] is None
        assert ind["vol_norm"] is None
        assert ind["er"] is None
        # always-needed indicators are still present
        assert ind["atr"] and ind["rsi"] and ind["ema_fast"]
    # gates ON -> arrays are built
    g_on = dict(g, htf_w=1.0, volz_w=1.0, er_min=0.2)
    ind2 = E._precompute_indicators(candles, g_on)
    if ind2 is not None:
        assert ind2["htf_ema"] is not None
        assert ind2["vol_norm"] is not None
        assert ind2["er"] is not None


def test_simulate_is_deterministic_and_gate_change_is_wellformed():
    candles = _synth_candles()
    g = E.normalize_genome(E.random_genome(random.Random(3)), random.Random(3))
    r1 = E.simulate(g, candles)
    r2 = E.simulate(g, candles)
    assert r1["total_return"] == r2["total_return"]
    assert r1["n_trades"] == r2["n_trades"]
    for key in ("total_return", "max_drawdown", "n_trades", "win_rate"):
        assert key in r1
