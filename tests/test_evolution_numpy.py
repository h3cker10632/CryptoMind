"""NumPy-vectorized backtester must stay numerically identical to the
pure-Python fallback (same trades, same returns) — the optimization is a
speedup, never a behavior change."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn import evolution as ev


def _synth(n=900, seed=7):
    r = random.Random(seed)
    out, px = [], 100.0
    for i in range(n):
        drift = 0.0015 if (i // 60) % 2 == 0 else -0.0010
        px *= 1 + drift + r.gauss(0, 0.005)
        px = max(1.0, px)
        hi = px * (1 + abs(r.gauss(0, 0.003)))
        lo = px * (1 - abs(r.gauss(0, 0.003)))
        out.append([i * 300, lo, hi, px, px, 1000.0])
    return out


_GENOMES = [
    {"ema_fast": 12, "ema_slow": 26, "breakout_n": 20, "stop_atr": 2.0,
     "take_atr": 3.0, "rsi_buy": 35, "rsi_sell": 70, "mom_w": 1.0, "short_w": 1.0},
    {"ema_fast": 8, "ema_slow": 30, "breakout_n": 15, "stop_atr": 1.5,
     "take_atr": 4.0, "rsi_buy": 30, "rsi_sell": 68, "mom_w": 0.0, "short_w": 0.0},
    {"ema_fast": 5, "ema_slow": 40, "breakout_n": 25, "stop_atr": 2.5,
     "take_atr": 5.0, "rsi_buy": 40, "rsi_sell": 72, "mom_w": 1.0, "short_w": 0.0},
]


def _run_both(genome, candles, sizing="risk"):
    """Return (numpy_result, purepython_result) for the same inputs."""
    orig = ev._np
    try:
        ev._np = orig                       # vectorized path (numpy present)
        r_np = ev.simulate(genome, candles, sizing=sizing)
        ev._np = None                       # forced pure-Python fallback
        r_py = ev.simulate(genome, candles, sizing=sizing)
    finally:
        ev._np = orig
    return r_np, r_py


def test_numpy_available():
    assert ev._np is not None, "numpy should be installed (see requirements.txt)"


def test_numpy_matches_fallback_all_genomes():
    candles = _synth()
    for g in _GENOMES:
        r_np, r_py = _run_both(g, candles)
        assert r_np["n_trades"] == r_py["n_trades"]
        assert abs(r_np["total_return"] - r_py["total_return"]) < 1e-9
        assert abs(r_np["max_drawdown"] - r_py["max_drawdown"]) < 1e-9
        assert abs(r_np["win_rate"] - r_py["win_rate"]) < 1e-9
        assert len(r_np["trades"]) == len(r_py["trades"])
        for a, b in zip(r_np["trades"], r_py["trades"]):
            assert abs(a - b) < 1e-9


def test_numpy_matches_fallback_fullcash_sizing():
    candles = _synth(seed=11)
    r_np, r_py = _run_both(_GENOMES[0], candles, sizing="fullcash")
    assert r_np["n_trades"] == r_py["n_trades"]
    assert abs(r_np["total_return"] - r_py["total_return"]) < 1e-9


def test_walk_forward_identical_across_paths():
    candles = _synth(n=800, seed=3)
    g = _GENOMES[1]
    orig = ev._np
    try:
        ev._np = orig
        wf_np = ev.walk_forward_eval(g, candles, n_windows=5, embargo=70)
        ev._np = None
        wf_py = ev.walk_forward_eval(g, candles, n_windows=5, embargo=70)
    finally:
        ev._np = orig
    assert wf_np["total_trades"] == wf_py["total_trades"]
    assert abs(wf_np["mean_return"] - wf_py["mean_return"]) < 1e-9
    assert abs(wf_np["frac_positive"] - wf_py["frac_positive"]) < 1e-9


def test_short_genome_precompute_parity():
    # a small breakout window + shorts exercises the sliding-window edges
    candles = _synth(seed=21)
    g = {"ema_fast": 6, "ema_slow": 20, "breakout_n": 8, "stop_atr": 1.8,
         "take_atr": 3.5, "rsi_buy": 33, "rsi_sell": 66, "mom_w": 1.0, "short_w": 1.0}
    r_np, r_py = _run_both(g, candles)
    assert r_np["n_trades"] == r_py["n_trades"]
    assert abs(r_np["total_return"] - r_py["total_return"]) < 1e-9
