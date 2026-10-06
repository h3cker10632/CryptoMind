"""Regression tests for the learning-contract fixes:

  A. Unseal the Q-agent
     A1. sit-out (0.0) never zeroes a cost-viable conviction trade — the risk
         manager floors the conviction scale at RL_CONVICTION_FLOOR.
     A2. sit-out surfaces `rl_sit_out=True` so the orchestrator can skip probes.
     A3. the RL agent only LEARNS from an interval that was really tradable;
         a fee-gated (locked-door) interval must not update Q.
     A4. no blanket holding cost — a genuine tradable transition still updates.

  B. Learning sees NET P&L
     B1. votes + regime_at_entry are stamped INSIDE broker.open(), so a same-tick
         close still carries them into attribution.
     B2. trade_attributions is persisted + restored.
     B3. a confirmed in-regime loser gets NO 0.04 floor (weight can decay to ~0).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------- A1
def test_sitout_does_not_zero_conviction_scale(monkeypatch):
    from app import settings
    real_get = settings.get
    monkeypatch.setattr(settings, "get",                 # live default is off
                        lambda k: "on" if k == "rl_risk_mode" else real_get(k))
    from app.risk.manager import risk, RL_CONVICTION_FLOOR
    from app.learn.rl_risk import agent
    # force the agent to sit out
    import app.learn.rl_risk as rl
    orig_act = agent.act
    agent.act = lambda *a, **k: 0.0
    try:
        st = risk.update(100_000, {"vol_state": "normal", "trend": "sideways"})
        assert st["rl_scale"] == 0.0
        assert st["rl_sit_out"] is True
        # effective scale must still be >= floor * streak/regime, never 0
        assert st["effective_risk_scale"] >= RL_CONVICTION_FLOOR * 0.99
    finally:
        agent.act = orig_act


# ----------------------------------------------------------------- A3/A4
def test_rl_skips_learning_on_locked_door():
    from app.learn.rl_risk import QRiskAgent
    a = QRiskAgent(eps=0.0)
    reg = {"trend": "down", "vol_state": "normal"}
    # first act — establishes prev_state; mark the interval NON-tradable
    a.act(reg, 0.02, 0, 100_000.0, tradable=False)
    n0 = a.n_updates
    # equity dropped, but the previous interval was a locked door → no learning
    a.act(reg, 0.02, 0, 95_000.0, tradable=True)
    assert a.n_updates == n0, "agent learned from a fee-gated (non-tradable) interval"


def test_rl_learns_on_tradable_interval():
    from app.learn.rl_risk import QRiskAgent
    a = QRiskAgent(eps=0.0)
    reg = {"trend": "down", "vol_state": "normal"}
    a.act(reg, 0.02, 0, 100_000.0, tradable=True)
    n0 = a.n_updates
    a.act(reg, 0.02, 0, 99_000.0, tradable=True)
    assert a.n_updates == n0 + 1, "agent failed to learn from a real tradable interval"


def test_rl_no_blanket_holding_cost_on_flat_equity():
    from app.learn.rl_risk import QRiskAgent, ACTIONS
    a = QRiskAgent(eps=0.0)
    reg = {"trend": "up", "vol_state": "normal"}
    # take some risk (non-zero scale) with perfectly flat equity, tradable
    a.prev_state = a.encode(reg, 0.0, 0)
    a.prev_action = ACTIONS.index(1.0)
    a.prev_equity = 100_000.0
    a.prev_tradable = True
    a.act(reg, 0.0, 0, 100_000.0, tradable=True)
    # flat equity + no drawdown => reward ~0 (NOT negative from a holding cost)
    assert abs(a.last_reward) < 1e-9, f"blanket holding cost still present: {a.last_reward}"


# ----------------------------------------------------------------- B1
def test_votes_stamped_inside_open_survive_same_tick_close():
    from app.execution.paper import broker
    from app.data.market import market
    broker.cash = 1_000_000.0
    broker.positions.clear()
    market.tickers["BTC-USD"] = {"price": 100.0}
    pos = broker.open("BTC-USD", 1, 1000.0, 100.0, 90.0, 130.0, "test",
                      votes={"trend": 0.6, "meanrev": -0.02},
                      regime_at_entry="bull")
    assert pos is not None
    # baked in at creation, before return
    assert pos["votes"] == {"trend": 0.6}          # tiny |v|<0.05 dropped
    assert pos["regime_at_entry"] == "bull"
    # close on the SAME reference — trade dict still carries them
    t = broker.sell("BTC-USD", 110.0, "same-tick close")
    assert t["votes"] == {"trend": 0.6}
    assert t["regime_at_entry"] == "bull"


def test_attribution_fires_from_stamped_votes():
    from app.execution.paper import broker
    from app.data.market import market
    from app.learn.loop import learner
    broker.cash = 1_000_000.0
    broker.positions.clear()
    market.tickers["ETH-USD"] = {"price": 50.0}
    before = learner.trade_attributions
    broker.open("ETH-USD", 1, 1000.0, 50.0, 45.0, 65.0, "test",
                votes={"trend": 0.7}, regime_at_entry="bull")
    t = broker.sell("ETH-USD", 55.0, "tp")
    learner.on_trade_closed(t)
    assert learner.trade_attributions == before + 1


# ----------------------------------------------------------------- B2
def test_trade_attributions_persist_roundtrip(tmp_path, monkeypatch):
    from app.learn.loop import learner
    from app import persistence
    monkeypatch.setattr(persistence, "STATE_PATH",
                        str(tmp_path / "state.json"), raising=False)
    learner.trade_attributions = 42
    persistence.save()
    learner.trade_attributions = 0
    persistence.load()
    assert learner.trade_attributions == 42


# ----------------------------------------------------------------- B3
def test_confirmed_loser_gets_no_floor():
    from app.learn.bandit import RegimeBandit
    b = RegimeBandit(["trend", "meanrev"])
    # meanrev: well-sampled, confidently negative in-regime edge
    for _ in range(40):
        b.update("bear", "meanrev", -0.0012)
    assert b._net_edge_ok("bear", "meanrev") is False
    assert b._net_edge_ok("bear", "trend") is True    # no evidence → not floored
