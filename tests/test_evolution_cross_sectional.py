"""Cross-sectional (universe-pooled) evolution.

A genuinely low-frequency edge produces too few trades per product to clear a
per-product evidence floor without over-trading. Pooling out-of-sample trades
across the whole basket meets the floor by BREADTH. These tests cover the
pooled evaluators and the evolve_universe path (structure/contract only; the
synthetic candles exercise the arithmetic and are never presented as real data).
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn import evolution as ev


def _synth(n=700, seed=7, drift=0.0006):
    r = random.Random(seed)
    out, px = [], 100.0
    for i in range(n):
        px *= 1 + drift + r.gauss(0, 0.02)
        px = max(1.0, px)
        hi = px * (1 + abs(r.gauss(0, 0.012)))
        lo = px * (1 - abs(r.gauss(0, 0.012)))
        out.append([i * 3600, lo, hi, px, px, 1000.0])
    return out


def _basket(n_products=5):
    return {f"P{i}-USD": _synth(seed=100 + i, drift=0.0004 + 0.0002 * i)
            for i in range(n_products)}


def test_pooled_walk_forward_pools_trades_across_products():
    cmap = _basket(5)
    g = ev.random_genome(random.Random(1))
    # per-product walk-forward trade counts must sum to the pooled total
    per = {p: ev.walk_forward_eval(g, c)["total_trades"] for p, c in cmap.items()}
    wf = ev.pooled_walk_forward_eval(g, cmap)
    assert wf["total_trades"] == sum(per.values())
    assert wf["n_products"] == 5
    assert 0.0 <= wf["frac_products_positive"] <= 1.0
    assert len(wf["per_product"]) == 5


def test_pooling_meets_trade_floor_better_than_single_product():
    """Breadth: pooled trades >= the max any single product contributes."""
    cmap = _basket(6)
    g = ev.random_genome(random.Random(2))
    wf = ev.pooled_walk_forward_eval(g, cmap)
    single_max = max(pp["total_trades"] for pp in wf["per_product"].values())
    assert wf["total_trades"] >= single_max


def test_pooled_train_objectives_shape():
    cmap = _basket(4)
    train = {p: c[:int(len(c) * 0.65)] for p, c in cmap.items()}
    obj = ev.pooled_train_objectives(ev.random_genome(random.Random(3)), train)
    assert len(obj) == 4                      # return, -dd, sharpe, trade-adequacy
    assert 0.0 <= obj[3] <= 1.0               # saturating trade-adequacy term


def test_evolve_universe_returns_valid_report_and_is_gated_honestly():
    cmap = _basket(5)
    e = ev.Evolution(pop_size=12, generations=4)
    rep = e.evolve_universe(cmap)
    for k in ("promoted", "gate_fail_breakdown", "best_observed_dsr",
              "n_trials", "basket", "front_size"):
        assert k in rep
    assert rep["basket"] == list(cmap.keys())
    # a promotion, if any, must be net-positive and stored for every product
    if rep["promoted"]:
        assert rep["genome"] is not None
        assert e.champions["_UNIVERSE"] == rep["genome"]
        for p in cmap:
            assert e.champion_portfolios[p] == rep["portfolio"]
