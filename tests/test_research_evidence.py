"""Research loop evidence: paired always-valid forward test, promotion rule,
candidate queue, after-tax / cost scenarios, the promotion-process replay and
the signal screen."""
import json

import numpy as np
import pytest


DAY = 86400


def _panel(T=1500, seed=0, drifts=(0.0015, 0.0008), vols=(0.03, 0.04), t0_day=17000):
    from app.engine.panel import Panel
    rng = np.random.default_rng(seed)
    N = len(drifts)
    r = rng.normal(np.array(drifts), np.array(vols), size=(T, N))
    close = 100 * np.exp(np.cumsum(r, axis=0))
    days = np.arange(t0_day, t0_day + T)
    high, low = close * 1.02, close * 0.98
    return Panel(days, ["BTC-USD", "ETH-USD"][:N], close, np.full_like(close, 1e9), "test",
                 high, low)


# ------------------------------------------------------------------ evidence
def test_newey_west_zero_lags_is_variance():
    from app.engine.evidence import newey_west_var
    x = np.random.default_rng(1).normal(size=500)
    assert abs(newey_west_var(x, 0) - x.var()) < 1e-12


def test_paired_test_detects_edge_and_ignores_leverage():
    from app.engine.evidence import paired_sequential
    rng = np.random.default_rng(2)
    champ = rng.normal(0.0005, 0.02, 400)
    better = champ + rng.normal(0.002, 0.003, 400)      # correlated, real edge
    worse = champ - rng.normal(0.002, 0.003, 400)
    assert paired_sequential(better, champ)["decision"] == "better"
    assert paired_sequential(worse, champ)["decision"] == "worse"
    levered = 2 * champ                                  # more risk is not skill
    assert paired_sequential(levered, champ)["decision"] == "undecided"


def test_paired_test_rarely_fires_on_no_difference_even_checked_daily():
    """Always-valid: checking every 5 days over ~2 years must not inflate the
    error rate far beyond alpha (two-sided -> ~2 x 0.05)."""
    from app.engine.evidence import paired_sequential
    rng = np.random.default_rng(3)
    fired = 0
    sims = 150
    for _ in range(sims):
        common = rng.normal(0.0005, 0.02, 700)
        a = common + rng.normal(0, 0.005, 700)
        b = common + rng.normal(0, 0.005, 700)
        for k in range(30, 701, 5):
            if paired_sequential(a[:k], b[:k])["decision"] != "undecided":
                fired += 1
                break
    assert fired / sims <= 0.12


def test_promotion_decision_gates_and_retirement():
    from app.engine.evidence import promotion_decision
    rng = np.random.default_rng(4)
    champ = rng.normal(0.0005, 0.02, 2000)
    good = champ + rng.normal(0.003, 0.002, 2000)
    bad = champ - rng.normal(0.003, 0.002, 2000)
    bt = {"champ": champ, "good": good, "bad": bad}
    fwd = {k: v[-120:] for k, v in bt.items()}
    v, best = promotion_decision(bt, fwd, "champ", n_trials=3, min_dsr=0.5,
                                 min_forward_days=30)
    assert best == "good" and v["good"]["eligible_for_promotion"]
    assert v["bad"]["retired"] and not v["bad"]["eligible_for_promotion"]
    # too little forward history: not eligible yet
    v2, best2 = promotion_decision(bt, {k: x[-10:] for k, x in bt.items()}, "champ", 3,
                                   min_dsr=0.5, min_forward_days=30)
    assert best2 is None and not v2["good"]["eligible_for_promotion"]
    # a retired candidate stays out even with good numbers
    _, best3 = promotion_decision(bt, fwd, "champ", 3, min_dsr=0.5, retired={"good"})
    assert best3 is None


# ------------------------------------------------------------------ queue + loop
@pytest.fixture
def research_dirs(tmp_path, monkeypatch):
    from app.engine import challengers as C, registry as Rg
    monkeypatch.setattr(C, "STATE", str(tmp_path / "challengers.json"))
    monkeypatch.setattr(C, "CHAMPION", str(tmp_path / "champion.json"))
    monkeypatch.setattr(C, "QUEUE", str(tmp_path / "queue.json"))
    monkeypatch.setattr(Rg, "DIR", str(tmp_path / "experiments"))
    majors = {k: v for k, v in C.CANDIDATES.items() if "assets" in v}
    monkeypatch.setattr(C, "CANDIDATES", majors)
    return C


def test_queue_validates_and_feeds_candidates(research_dirs):
    C = research_dirs
    ok, msg = C.register("trend_150", {"assets": ["BTC-USD", "ETH-USD"], "selection": "trend",
                                       "sizing": "equal", "sma": 150, "hysteresis": 0.02})
    assert ok, msg
    assert "trend_150" in C.candidates()
    bad = [{"assets": ["BTC-USD"], "selection": "rm -rf"},
           {"assets": ["BTC-USD"], "universe_top": 5, "selection": "trend"},
           {"assets": ["BTC-USD"], "selection": "trend", "sma": 9999},
           {"assets": ["BTC-USD"], "selection": "trend", "evil": 1}]
    for cfg in bad:
        assert not C.register("x", cfg)[0]
    assert not C.register("btc_eth_trend", C.CANDIDATES["btc_eth_trend"])[0]
    assert C.unregister("trend_150") and "trend_150" not in C.candidates()


def test_champion_survives_leaving_the_queue(research_dirs):
    C = research_dirs
    cfg = {"assets": ["BTC-USD", "ETH-USD"], "selection": "trend", "sma": 150}
    C.register("q1", cfg)
    C._save(C.CHAMPION, {"name": "q1", "config": cfg})
    C.unregister("q1")
    assert C.champion() == ("q1", cfg)


def test_research_run_reports_forward_test_and_costs(research_dirs):
    C = research_dirs
    p = _panel()
    rep = C.run(panel=p, backtest_from="2016-07-01", min_forward_days=30)
    champ = rep["candidates"][rep["champion"]]
    assert champ["forward_test"]["decision"] == "champion"
    other = next(v for k, v in rep["candidates"].items() if k != rep["champion"])
    assert other["forward_test"]["decision"] in ("undecided", "better", "worse")
    sc = rep["costs_taxes"]["scenarios"]
    assert {"exchange taker (current) pre-tax", "spot ETF pre-tax",
            "exchange taker after-tax", "hold same coins after-tax"} <= set(sc)
    json.dumps(rep, default=float)                       # JSON-able


# ------------------------------------------------------------------ costs / taxes
def test_taxed_simulation_without_tax_equals_drift_simulation():
    from app.engine import backtest as B, costs_tax as CT, strategies as S
    p = _panel(T=900)
    W = S.trend_portfolio(p, "trend", sma=50, hysteresis=0.01)
    r, _, _ = B.simulate_drift(W, p.returns(), 0.006, start=60, band_rel=0.2)
    tx = CT.simulate_taxed(W, p.close, p.days, 0.006, start=60, band_rel=0.2)
    np.testing.assert_allclose(tx["rets"], r, rtol=0, atol=1e-12)
    assert tx["taxes"] == 0


def test_taxes_cost_more_for_frequent_trading_than_holding():
    from app.engine import costs_tax as CT, strategies as S
    p = _panel(T=1500, drifts=(0.002, 0.0015))
    W = S.trend_portfolio(p, "trend", sma=20)
    Wh = np.zeros_like(W)
    Wh[60:] = 0.5
    trade = CT.simulate_taxed(W, p.close, p.days, 0.0, 60, 0.0, 0.3, 0.15)
    hold = CT.simulate_taxed(Wh, p.close, p.days, 0.0, 60, 1e9, 0.3, 0.15)
    assert trade["taxes"] > 0 and hold["taxes"] == 0      # holding defers until sold
    assert hold["liquidation"] < hold["final"]            # ...then pays at the end


def test_etf_execution_waits_for_next_weekday():
    from app.engine.costs_tax import etf_execution, _weekday
    days = np.arange(20000, 20014)
    fri = int(np.flatnonzero([_weekday(d) == 4 for d in days])[0])
    W = np.zeros((14, 1))
    W[fri:] = 1.0
    E = etf_execution(W, days)
    mon = fri + 3
    assert E[mon - 1, 0] == 0 and E[mon, 0] == 1.0


# ------------------------------------------------------------------ promotion replay
def test_promotion_backtest_promotes_a_clearly_better_candidate():
    from app.engine import promotion_backtest as PB
    p = _panel(T=1800, drifts=(0.0003, 0.004), vols=(0.03, 0.03))

    def wfn(cfg, panel, start=0):
        W = np.zeros((panel.T, panel.N))
        W[:, :] = cfg["w"]
        return W
    cands = {"base": {"w": [1.0, 0.0]}, "better": {"w": [0.0, 1.0]}}
    rep = PB.run(p, cands, "base", backtest_from="2016-07-01", eval_every=30, min_dsr=0.5,
                 weights_fn=wfn)
    assert rep["promotions"] and rep["promotions"][0]["to"] == "better"
    assert rep["process"]["full"]["sharpe"] > rep["never_switch"]["full"]["sharpe"]
    assert rep["verdict"].startswith("promotion process beat")


# ------------------------------------------------------------------ signal screen
def test_screen_finds_a_real_signal_and_rejects_noise():
    from app.engine import screen as S
    rng = np.random.default_rng(5)
    T, N, h = 1200, 30, 7
    signal = rng.normal(size=(T, N))
    r = rng.normal(0, 0.03, size=(T, N))
    r[1:] += 0.004 * signal[:-1]                          # today's signal -> tomorrow's return
    close = 100 * np.exp(np.cumsum(r, axis=0))
    good = S.cross_sectional(signal, close, h=1)
    noise = S.cross_sectional(rng.normal(size=(T, N)), close, h=1)
    assert good["passes"] and good["mean"] > 0
    assert not noise["passes"]
    res = S.holm({"good": good, "noise": noise})
    assert res["good"]["passes"] and res["good"]["p_holm"] <= 0.05
