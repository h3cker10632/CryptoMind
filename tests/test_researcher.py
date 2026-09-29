"""Tests for the Strategy Researcher (app/learn/researcher.py).

Focus: the honest primitives — DSL safety/validation, train/live parity of the
vote, the walk-forward evaluation, the deflated-Sharpe gate that prices multiple
testing, and that a discovered candidate becomes a live ensemble arm without
crashing the signal engine.
"""
import math
import random

import pytest

from app.learn import researcher as R


def _synthetic_candles(n=600, seed=1, trend=0.0, noise=0.01):
    """Synthetic OHLCV [ts, lo, hi, op, cl, vol] purely to exercise the harness
    (labeled synthetic — never fed to the live learner)."""
    rnd = random.Random(seed)
    px = 100.0
    out = []
    for i in range(n):
        drift = trend
        ret = drift + rnd.gauss(0, noise)
        op = px
        cl = max(0.1, px * (1 + ret))
        hi = max(op, cl) * (1 + abs(rnd.gauss(0, noise / 2)))
        lo = min(op, cl) * (1 - abs(rnd.gauss(0, noise / 2)))
        vol = 1000 + rnd.random() * 100
        out.append([1_600_000_000 + i * 3600, lo, hi, op, cl, vol])
        px = cl
    return out


def test_validate_candidate_whitelist():
    good = {"long": [{"feat": "rsi", "op": "<", "thr": 30}], "conf": 0.6}
    assert R.validate_candidate(good)
    # unknown feature rejected (LLM injection guard)
    assert not R.validate_candidate({"long": [{"feat": "os.system", "op": "<", "thr": 1}], "conf": 0.5})
    # bad op rejected
    assert not R.validate_candidate({"long": [{"feat": "rsi", "op": "==", "thr": 1}], "conf": 0.5})
    # non-numeric threshold rejected
    assert not R.validate_candidate({"long": [{"feat": "rsi", "op": "<", "thr": "x"}], "conf": 0.5})
    # no clauses at all rejected
    assert not R.validate_candidate({"conf": 0.5})
    # confidence out of range rejected
    assert not R.validate_candidate({"long": [{"feat": "rsi", "op": "<", "thr": 30}], "conf": 5})


def test_vote_parity_derived_vs_live():
    """vote() on a live feature dict must equal _vote_derived() on the derived
    dict — this is the train/live parity guarantee."""
    f = {"price": 105.0, "hi20": 110.0, "lo20": 100.0, "atr": 2.0,
         "rsi": 25.0, "mom_1h": 0.01, "mom_4h": 0.02, "macd_delta": 0.5,
         "vol_ratio": 1.2, "ema12": 106.0, "ema26": 104.0}
    cand = {"long": [{"feat": "rsi", "op": "<", "thr": 30}], "conf": 0.7}
    assert R.vote(cand, f) == pytest.approx(0.7)
    d = R.derive(f)
    assert R._vote_derived(cand, d) == pytest.approx(0.7)
    # a short candidate whose condition is false returns 0
    cand2 = {"short": [{"feat": "rsi", "op": ">", "thr": 70}], "conf": 0.7}
    assert R.vote(cand2, f) == 0.0


def test_derive_ranges():
    f = {"price": 105.0, "hi20": 110.0, "lo20": 100.0, "atr": 2.0,
         "rsi": 55.0, "mom_1h": 0.0, "mom_4h": 0.0, "macd_delta": 0.0,
         "vol_ratio": 1.0, "ema12": 106.0, "ema26": 104.0}
    d = R.derive(f)
    assert 0.0 <= d["pos20"] <= 1.0
    assert d["ema_cross"] == 1.0
    assert d["pos20"] == pytest.approx(0.5)


def test_feature_series_and_backtest_shapes():
    candles = _synthetic_candles(400)
    feats = R.feature_series(candles)
    assert len(feats) == len(candles)
    assert feats[0] is None                      # warmup
    assert any(x is not None for x in feats[-50:])
    cand = {"long": [{"feat": "ema_cross", "op": ">", "thr": 0.0}], "conf": 0.6}
    closes = [c[4] for c in candles]
    rets, trades = R._backtest_returns(cand, feats, closes)
    assert len(rets) == len(candles) - 1
    assert trades >= 0


def test_evaluate_walk_forward_metrics():
    candles = _synthetic_candles(600, trend=0.0005)
    feats = R.feature_series(candles)
    closes = [c[4] for c in candles]
    cand = {"long": [{"feat": "ema_cross", "op": ">", "thr": 0.0}], "conf": 0.6}
    m = R.evaluate(cand, feats, closes)
    assert m is not None
    for k in ("oos_sharpe", "frac_folds_positive", "trades", "oos_returns"):
        assert k in m
    assert 0.0 <= m["frac_folds_positive"] <= 1.0


def test_gate_prices_multiple_testing():
    """More trials => higher deflated-Sharpe benchmark => harder to pass. A pure
    noise candidate should NOT pass the gate."""
    candles = _synthetic_candles(700, seed=7, trend=0.0)      # no real edge
    feats = R.feature_series(candles)
    closes = [c[4] for c in candles]
    cand = {"long": [{"feat": "rsi", "op": "<", "thr": 45}], "conf": 0.6}
    m = R.evaluate(cand, feats, closes)
    if m:
        passed_few, dsr_few = R.gate(m, n_trials=2, trial_sr_std=0.3)
        passed_many, dsr_many = R.gate(m, n_trials=500, trial_sr_std=0.3)
        # pricing more trials can only make the deflated Sharpe no larger
        assert dsr_many <= dsr_few + 1e-9
        # a no-edge candidate must not clear the honest bar under heavy search
        assert not passed_many


def test_sample_systematic_valid_and_reproducible():
    a = R.sample_systematic(random.Random(42), 30)
    b = R.sample_systematic(random.Random(42), 30)
    assert len(a) == 30
    assert all(R.validate_candidate(c) for c in a)
    # same seed => identical candidates (reproducible search)
    assert [R.candidate_id(x) for x in a] == [R.candidate_id(y) for y in b]


def test_run_once_and_engine_arm(tmp_path):
    """End-to-end: run discovery on synthetic candles, and confirm the engine's
    `discovered` arm reads the promoted portfolio without crashing."""
    store = tmp_path / "disc.json"
    res = R.Researcher(store_path=str(store))
    candles = _synthetic_candles(700, seed=3, trend=0.0008)   # mild real trend
    summary = res.run_once("BTC-USD", candles=candles, n_systematic=40, use_llm=False,
                           seed=11)
    assert summary["ok"] is True
    assert summary["tested"] > 0
    assert summary["evaluated"] > 0
    # portfolio_for returns validated candidates (possibly empty if none passed)
    pop = res.portfolio_for("BTC-USD")
    for c in pop:
        assert R.validate_candidate(c)

    # the engine arm must be registered and never raise on a live feature dict
    from app.signals.engine import STRATEGIES, strat_discovered, _ctx
    assert "discovered" in STRATEGIES
    _ctx.product = "BTC-USD"
    f = {"price": 105.0, "hi20": 110.0, "lo20": 100.0, "atr": 2.0, "rsi": 25.0,
         "mom_1h": 0.01, "mom_4h": 0.02, "macd_delta": 0.5, "vol_ratio": 1.2,
         "ema12": 106.0, "ema26": 104.0}
    v = strat_discovered(f, 0.0, {"label": "trending"})
    assert -1.0 <= v <= 1.0


def test_status_and_clear(tmp_path):
    res = R.Researcher(store_path=str(tmp_path / "disc.json"))
    st = res.status()
    assert "gate" in st and st["gate"]["dsr_min"] == R.DSR_MIN
    assert "features" in st
    res.clear()
    assert res.status()["promoted_total"] == 0
