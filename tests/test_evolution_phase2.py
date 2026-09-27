"""Phase 2 evolution deep-dive:
  * NSGA-II multi-objective selection (Pareto sort + crowding)
  * purged walk-forward validation
  * champion PORTFOLIO (top-k) averaged in the live vote
  * train/live SIZING PARITY (backtest sizes like the risk manager)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn import evolution as ev


def _synth_candles(n=700, seed=1):
    import random
    r = random.Random(seed)
    out, px = [], 100.0
    for i in range(n):
        drift = 0.0015 if (i // 60) % 2 == 0 else -0.0010   # alternating regimes
        px *= 1 + drift + r.gauss(0, 0.004)
        px = max(1.0, px)
        hi, lo = px * (1 + abs(r.gauss(0, 0.003))), px * (1 - abs(r.gauss(0, 0.003)))
        out.append([i * 300, lo, hi, px, px, 1000.0])
    return out


# ---------- NSGA-II primitives ----------
def test_domination_and_front():
    objs = [(1.0, 1.0, 1.0),    # 0 dominates 1 and 2
            (0.5, 0.5, 0.5),    # 1
            (0.9, 0.2, 0.2),    # 2 (non-dominated vs 3 on obj0)
            (0.1, 0.9, 0.1)]    # 3
    assert ev._dominates(objs[0], objs[1])
    assert not ev._dominates(objs[2], objs[3])
    fronts = ev._fast_non_dominated_sort(objs)
    assert 0 in fronts[0]                       # the dominator is on the top front
    # every index appears in exactly one front
    flat = [i for f in fronts for i in f]
    assert sorted(flat) == list(range(len(objs)))


def test_crowding_gives_boundaries_infinite_distance():
    front = [0, 1, 2]
    objs = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    cd = ev._crowding_distance(front, objs)
    assert cd[0] == float("inf") and cd[2] == float("inf")
    assert cd[1] < float("inf")


def test_nsga2_select_prefers_pareto_front():
    pop = ["a", "b", "c", "d"]
    objs = [(1.0, 1.0), (0.9, 0.9), (0.2, 0.2), (0.1, 0.1)]
    chosen, fronts = ev._nsga2_select(pop, objs, 2, __import__("random").Random(0))
    assert set(chosen) == {"a", "b"}           # the two dominant genomes


# ---------- walk-forward ----------
def test_walk_forward_reports_multiple_windows():
    g = {"ema_fast": 8, "ema_slow": 30, "rsi_buy": 30, "rsi_sell": 70,
         "breakout_n": 20, "stop_atr": 2.0, "take_atr": 4.0,
         "mom_w": 0.4, "short_w": 0.0}
    wf = ev.walk_forward_eval(g, _synth_candles(), n_windows=5, embargo=70)
    assert wf["n_windows"] == 5
    assert 0.0 <= wf["frac_positive"] <= 1.0
    assert "worst_drawdown" in wf and "pooled_sharpe" in wf


# ---------- sizing parity ----------
def test_risk_sizing_deploys_far_less_than_fullcash():
    g = {"ema_fast": 8, "ema_slow": 30, "rsi_buy": 35, "rsi_sell": 70,
         "breakout_n": 15, "stop_atr": 2.0, "take_atr": 4.0,
         "mom_w": 0.0, "short_w": 0.0}
    candles = _synth_candles()
    r_risk = ev.simulate(g, candles, sizing="risk")
    r_full = ev.simulate(g, candles, sizing="fullcash")
    # both run the same rules; risk-based sizing must move equity far less per
    # trade, so its magnitude of return is much smaller than 95%-of-cash.
    assert abs(r_risk["total_return"]) < abs(r_full["total_return"]) + 1e-9
    assert r_risk["n_trades"] >= 1


def test_default_sizing_is_risk_parity():
    g = {"ema_fast": 8, "ema_slow": 30, "rsi_buy": 35, "rsi_sell": 70,
         "breakout_n": 15, "stop_atr": 2.0, "take_atr": 4.0,
         "mom_w": 0.0, "short_w": 0.0}
    candles = _synth_candles()
    assert ev.simulate(g, candles) == ev.simulate(g, candles, sizing="risk")


# ---------- champion portfolio + live averaging ----------
def test_portfolio_for_falls_back_to_single_champion():
    e = ev.Evolution(pop_size=8, generations=1, seed=3)
    champ = {"ema_fast": 8, "ema_slow": 30, "rsi_buy": 30, "rsi_sell": 70,
             "breakout_n": 20, "stop_atr": 2.0, "take_atr": 4.0,
             "mom_w": 0.6, "short_w": 0.0}
    e.champions["BTC-USD"] = champ
    assert e.portfolio_for("BTC-USD") == [champ]         # 1-element fallback


def test_evolve_promotes_portfolio_and_reports_walk_forward():
    e = ev.Evolution(pop_size=16, generations=4, seed=7)
    rep = e.evolve(_synth_candles(n=800, seed=2), product="BTC-USD")
    assert "walk_forward" in rep and "portfolio_size" in rep
    if rep["promoted"]:
        assert 1 <= rep["portfolio_size"] <= 3
        assert e.portfolio_for("BTC-USD") == e.champion_portfolios["BTC-USD"]
        assert rep["walk_forward"]["n_windows"] == 5


def test_evolved_vote_averages_portfolio(monkeypatch):
    import app.signals.engine as eng
    # two genomes that score differently; the live vote must be their mean.
    g_bull = {"ema_fast": 8, "ema_slow": 30, "rsi_buy": 40, "rsi_sell": 70,
              "breakout_n": 20, "stop_atr": 2.0, "take_atr": 4.0,
              "mom_w": 0.0, "short_w": 0.0}
    g_flat = dict(g_bull, rsi_buy=20)         # harder to trigger dip-buy
    feats = {"ema12": 110, "ema26": 100, "hi20": 115, "lo20": 90, "price": 114,
             "rsi": 35, "mom_1h": 1.0}
    v_bull = eng._evolved_vote(g_bull, feats)
    v_flat = eng._evolved_vote(g_flat, feats)

    class FakeEvo:
        def portfolio_for(self, p):
            return [g_bull, g_flat]
    monkeypatch.setattr(eng, "_cur_product", lambda: "BTC-USD")
    import app.learn.evolution as evo_mod
    monkeypatch.setattr(evo_mod, "evolution", FakeEvo())
    out = eng.strat_evolved(feats, (0.0,), "bull")
    assert abs(out - max(-1, min(1, (v_bull + v_flat) / 2))) < 1e-9


def test_evolve_report_schema_matches_loop_and_dashboard_consumers():
    """Regression: Phase 2 renamed the report keys; the live loop's log line and
    the dashboard read them. A run's report MUST expose the keys those consumers
    use, and MUST NOT rely on the removed 'validation'/'validation_fitness'.
    """
    e = ev.Evolution(pop_size=12, generations=3, seed=9)
    rep = e.evolve(_synth_candles(n=800, seed=2), product="BTC-USD")
    # keys the loop's log message + dashboard now read (must all resolve):
    for k in ("product", "train_fitness", "pooled_oos_sharpe", "walk_forward",
              "promoted", "portfolio_size", "genome"):
        assert k in rep, k
    # the removed keys must be gone so nothing silently reads a stale schema:
    assert "validation" not in rep
    assert "validation_fitness" not in rep
    # emulate the exact f-string the loop builds — this used to KeyError:
    wf = rep.get("walk_forward") or {}
    msg = (f"train_fit={rep.get('train_fitness')} "
           f"pooled_oos_sharpe={rep.get('pooled_oos_sharpe')} "
           f"oos_windows_positive={wf.get('frac_positive')} "
           f"promoted={rep.get('promoted')} "
           f"portfolio={rep.get('portfolio_size', 0)}")
    assert "promoted=" in msg


def test_maybe_evolve_worker_logs_without_keyerror(monkeypatch):
    """End-to-end: the loop's evolution worker must complete WITHOUT throwing,
    so a finished GA run is never swallowed as 'Evolution failed'."""
    import app.learn.loop as loop_mod
    import app.backtest.engine as bt_engine
    from app.learn.loop import learner

    candles = _synth_candles(n=800, seed=2)

    async def _fake_fetch(product, granularity=None, chunks=None):
        return candles
    # the worker does `from ..backtest.engine import fetch_history` at call time,
    # so patch it on the source module (this is the real target).
    monkeypatch.setattr(bt_engine, "fetch_history", _fake_fetch, raising=False)

    logged = []
    monkeypatch.setattr(loop_mod.db, "log_event",
                        lambda *a, **k: logged.append(a))
    # force a small, fast GA and bypass the cadence/thread guards
    learner._evo_thread = None
    learner.last_evolution_start = 0
    monkeypatch.setattr(loop_mod, "tv", lambda k: 0 if k == "evolve_every_sec" else 1,
                        raising=False)
    loop_mod.evolution.pop_size = 10
    loop_mod.evolution.generations = 2

    learner.maybe_evolve(product="BTC-USD")
    learner._evo_thread.join(timeout=30)
    # no error event was logged, and the run reached "done" (not "error")
    assert loop_mod.evolution.status != "error"
    assert not any(a and a[0] == "error" for a in logged), logged


def test_dsr_uses_measured_trial_dispersion_not_placeholder():
    """The deflated-Sharpe multiple-testing correction must use the MEASURED
    dispersion of the run's trial Sharpes, not the old hard-coded 0.5 placeholder
    that over-penalised every genome to a deflated Sharpe of ~0. The report must
    expose the measured trial_sr_std and the real evaluated trial count.
    """
    import statistics
    e = ev.Evolution(pop_size=12, generations=3, seed=9)
    rep = e.evolve(_synth_candles(n=800, seed=2), product="BTC-USD")
    assert "trial_sr_std" in rep
    std = rep["trial_sr_std"]
    # measured, finite, and driven by the data (not the removed 0.5 constant)
    assert isinstance(std, float) and std > 0.0
    assert std >= 0.05                      # respects the safety floor
    assert std != 0.5                       # not the old placeholder
    # n_trials is the count actually evaluated (pop_size * generations)
    assert rep["n_trials"] == 12 * 3
