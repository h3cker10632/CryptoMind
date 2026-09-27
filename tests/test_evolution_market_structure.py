"""Market-structure genome enrichment (htf trend agreement + volatility regime).

These candle-derived, opt-in features must:
  1. flow through random_genome / mutate / crossover automatically (GENE_SPACE);
  2. leave a genome with the gates DISABLED bit-identical to a legacy genome
     that never had these genes (strict superset — evolution can turn them off);
  3. only ever FILTER entries when enabled (never invent trades / lookahead);
  4. keep the NumPy and pure-Python paths identical for the htf gate (the vol
     gate is a NumPy-only refinement and is disabled on the fallback path).

The candles here are synthetic ONLY to exercise the estimator's arithmetic —
they are never presented as real market data or fed to the live learner.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn import evolution as ev

NEW = {"htf_n", "htf_w", "volz_w", "volz_max", "er_n", "er_min"}


def _synth(n=800, seed=7):
    r = random.Random(seed)
    out, px = [], 100.0
    for i in range(n):
        drift = 0.0015 if (i // 50) % 2 == 0 else -0.0012
        px *= 1 + drift + r.gauss(0, 0.02)
        px = max(1.0, px)
        hi = px * (1 + abs(r.gauss(0, 0.012)))
        lo = px * (1 - abs(r.gauss(0, 0.012)))
        out.append([i * 3600, lo, hi, px, px, 1000.0])
    return out


def _summ(res):
    return (round(res["total_return"], 8), res["n_trades"])


def test_new_genes_flow_through_operators():
    rnd = random.Random(1)
    g = ev.random_genome(rnd)
    assert NEW <= set(g), "new genes missing from random_genome"
    assert isinstance(g["htf_n"], int) and 24 <= g["htf_n"] <= 120
    assert NEW <= set(ev.mutate(g, rnd)), "mutate dropped new genes"
    assert NEW <= set(ev.crossover(g, ev.random_genome(rnd), rnd)), \
        "crossover dropped new genes"


def test_gates_off_is_identical_to_legacy_genome():
    """Disabled gates == a genome that never had these genes (superset)."""
    candles = _synth()
    for seed in (3, 11, 42, 99):
        rnd = random.Random(seed)
        g = ev.random_genome(rnd)
        legacy = {k: v for k, v in g.items() if k not in NEW}
        off = dict(g, htf_w=0.0, volz_w=0.0)
        assert _summ(ev.simulate(legacy, candles)) == _summ(ev.simulate(off, candles))


def test_enabled_gates_only_filter_entries():
    candles = _synth()
    for seed in (3, 11, 42, 99):
        rnd = random.Random(seed)
        g = ev.random_genome(rnd)
        off = dict(g, htf_w=0.0, volz_w=0.0)
        on = dict(g, htf_w=1.0, volz_w=1.0, volz_max=1.1)
        base = ev.simulate(off, candles)["n_trades"]
        gated = ev.simulate(on, candles)["n_trades"]
        assert gated <= base, "market-structure gates must never add trades"


def test_regime_filter_pinned_off_by_default_and_activates_on_flag():
    """er_min is pinned to 0 (inert) unless ga_regime_filter is enabled."""
    from app import tunables as T
    rnd = random.Random(2)
    T._overrides = {}                                  # default: filter off
    assert all(ev.random_genome(rnd)["er_min"] == 0.0 for _ in range(20))
    T._overrides = {"ga_regime_filter": 1}             # enabled
    assert any(ev.random_genome(rnd)["er_min"] > 0 for _ in range(20))
    T._overrides = {}


def test_regime_filter_only_reduces_trades():
    candles = _synth()
    g = ev.random_genome(random.Random(4))
    off = dict(g, er_min=0.0)
    on = dict(g, er_min=0.25)                           # require a real trend
    assert ev.simulate(on, candles)["n_trades"] <= ev.simulate(off, candles)["n_trades"]


def test_er_gate_identical_numpy_and_fallback():
    """The efficiency-ratio filter is computed on both paths -> must match."""
    candles = _synth()
    g = dict(ev.random_genome(random.Random(6)), er_min=0.2, htf_w=0.0, volz_w=0.0)
    orig = ev._np
    try:
        ev._np = orig
        r_np = ev.simulate(g, candles)
        ev._np = None
        r_py = ev.simulate(g, candles)
    finally:
        ev._np = orig
    assert _summ(r_np) == _summ(r_py)


def test_trade_log_matches_trade_count():
    """The optional entry-index logger records exactly one (idx, side) per trade."""
    candles = _synth()
    g = ev.random_genome(random.Random(8))
    log = []
    res = ev.simulate(g, candles, trade_log=log)
    assert len(log) == res["n_trades"]
    assert all(isinstance(i, int) and s in (1, -1) for i, s in log)


def test_htf_gate_identical_numpy_and_fallback():
    """The htf trend gate is computed on both paths -> must match exactly."""
    candles = _synth()
    g = dict(ev.random_genome(random.Random(5)), htf_w=1.0, volz_w=0.0)
    orig = ev._np
    try:
        ev._np = orig
        r_np = ev.simulate(g, candles)
        ev._np = None
        r_py = ev.simulate(g, candles)
    finally:
        ev._np = orig
    assert _summ(r_np) == _summ(r_py)
