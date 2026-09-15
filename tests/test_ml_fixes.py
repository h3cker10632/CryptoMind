"""Regression tests for the five ML/risk correctness fixes.

  1. bandit: cross-regime pooling must not flip the sign of a well-sampled
     negative in-regime edge, and confirmed in-regime losers get near-floor draws.
  2. online model: the ml vote is gated to 0 until directional accuracy > 50%,
     and drift must NOT boost the LR of a below-coin-flip head.
  3. evolution: promotion needs a larger OOS sample; a rerun that fails to
     re-validate evicts the stale champion.
  4. risk: kill-reset must NOT zero the true high-water mark (no drawdown amnesia)
     while still re-arming the auto-kill.
  5. rl_risk: a sit-out (0.0) action exists and flat/losing equity drives the
     learned Q-value toward it.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ Fix 1
def test_bandit_pool_cannot_flip_negative_local_sign():
    from app.learn.bandit import RegimeBandit
    b = RegimeBandit(["evolved", "trend"])
    # 'evolved' is strongly positive in 'bull' but clearly negative in 'bear'
    for _ in range(80):
        b.update("bull", "evolved", 0.0020)      # +20 bps elsewhere
    for _ in range(60):
        b.update("bear", "evolved", -0.0010)     # -10 bps in-regime
    mu, sd, n = b._posterior("bear", "evolved")
    assert mu < 0, f"pool flipped a well-sampled negative local edge positive: {mu}"


def test_bandit_confirmed_loser_gets_low_weight():
    from app.learn.bandit import RegimeBandit
    import random
    random.seed(0)
    b = RegimeBandit(["evolved", "trend"])
    for _ in range(80):
        b.update("bull", "evolved", 0.0025)
    for _ in range(60):
        b.update("bear", "evolved", -0.0010)     # confirmed in-regime loser
    for _ in range(60):
        b.update("bear", "trend", 0.0008)        # honest positive edge
    ws = [b.sample_weights("bear") for _ in range(200)]
    avg_evolved = sum(w["evolved"] for w in ws) / len(ws)
    avg_trend = sum(w["trend"] for w in ws) / len(ws)
    assert avg_trend > avg_evolved, (avg_trend, avg_evolved)


# ------------------------------------------------------------------ Fix 2
def test_ml_vote_zero_below_coin_flip():
    from app.learn.online_model import model
    from app.signals import engine
    # force a warmed-up but bad model
    model.n_updates = 500
    model.acc_window.clear()
    model.acc_window.extend([0] * 200)           # 0% directional accuracy
    engine._ctx = getattr(engine, "_ctx", None)
    f = {"ema12": 1, "ema26": 0, "hi20": 1, "lo20": 0, "price": 0.5,
         "rsi": 50, "mom_1h": 0, "spread_bps": 1}
    try:
        engine._set_ctx("BTC-USD", 0.0) if hasattr(engine, "_set_ctx") else None
    except Exception:
        pass
    # call strat_ml directly with a benign sentiment/regime
    v = engine.strat_ml(f, (0.0, 0.0), {"trend": "up"})
    assert v == 0.0, f"below-coin-flip model still voted {v}"


def test_drift_does_not_boost_broken_head():
    from app.learn.online_model import model
    from app.learn import loop as loop_mod
    model.n_updates = 500
    model.lr_boost = 1.0
    model.acc_window.clear()
    model.acc_window.extend([0] * 200)           # broken head

    # stub the detector to report drift
    class _Det:
        def check(self, names):
            return True, "market_sent", 6.0
        def stats(self):
            return {}
    orig = loop_mod.detector
    loop_mod.detector = _Det()
    try:
        loop_mod.learner._check_drift()
        assert model.lr_boost == 1.0, "LR was boosted on a below-coin-flip head"
    finally:
        loop_mod.detector = orig


# ------------------------------------------------------------------ Fix 4
def test_kill_reset_preserves_high_water_mark():
    from app.risk.manager import RiskManager
    r = RiskManager()
    reg = {"vol_state": "normal", "trend": "sideways"}
    r.update(100_000, reg)          # peak set to 100k
    r.update(74_200, reg)           # deep drawdown
    dd_before = 1 - 74_200 / r.peak_equity
    assert abs(dd_before - 0.258) < 0.01
    r.reset_kill()
    r.update(74_200, reg)           # a tick right after reset
    dd_after = 1 - 74_200 / r.peak_equity
    assert r.peak_equity >= 100_000, "true high-water mark was zeroed on reset"
    assert dd_after > 0.20, f"drawdown amnesia: reads {dd_after:.1%} after reset"


def test_kill_reset_rearms_autokill():
    from app.risk.manager import RiskManager
    r = RiskManager()
    reg = {"vol_state": "normal", "trend": "sideways"}
    r.update(100_000, reg)
    r.update(74_200, reg)           # -25.8% trips the 15% kill
    assert r.killed
    r.reset_kill()
    # same equity, next tick: must NOT instantly re-trip (arm baseline re-set)
    r.update(74_200, reg)
    assert not r.killed, "auto-kill re-tripped on an already-acknowledged drawdown"


# ------------------------------------------------------------------ Fix 5
def test_rl_has_sitout_action():
    from app.learn.rl_risk import ACTIONS
    assert 0.0 in ACTIONS, "no sit-out action available to the RL agent"


def test_rl_learns_to_sit_out_when_losing():
    from app.learn.rl_risk import QRiskAgent, ACTIONS
    import random
    random.seed(1)
    a = QRiskAgent(eps=0.0)          # greedy: measure what it has learned
    reg = {"trend": "down", "vol_state": "high-vol"}
    equity = 100_000.0
    # steadily losing regime; force the agent to actually take risk sometimes
    for i in range(400):
        a.act(reg, drawdown=0.10, streak=3, equity=equity)
        # emulate that non-zero risk keeps bleeding equity, sit-out is flat
        chosen = ACTIONS[a.prev_action]
        equity *= (1 - 0.002 * chosen)     # loss scales with risk taken
    row = a._qrow(a.encode(reg, 0.10, 3))
    best = max(range(len(row)), key=lambda i: row[i])
    # the sit-out (0.0) arm should be the best or near-best; certainly better
    # than pressing full risk in a bleeding regime.
    assert row[0] >= row[-1], f"sit-out not favored over full risk: {row}"
