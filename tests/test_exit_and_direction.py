"""Predictive loss-cut exit advisor + direction (long/short) learner."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.learn.exit_advisor import ExitAdvisor
from app.learn.direction import DirectionLearner
from app import tunables


# ---------------- exit advisor ----------------

def _regime(trend="bear", vol="normal"):
    return {"trend": trend, "vol_state": vol}


def test_cuts_losing_position_expecting_further_adverse_move():
    tunables.update({"exit_cut_threshold": 0.004, "exit_min_loss_pct": 0.003,
                     "exit_ml_weight": 1.0})
    ea = ExitAdvisor()
    action, why, exp = ea.decide("BTC-USD", side=1, unrealized_pct=-0.01,
                                 ml_pos=-0.008, regime=_regime(), atr_pct=0.005)
    assert action == "cut"
    assert exp < 0


def test_holds_winner_even_if_model_soft_negative():
    ea = ExitAdvisor()
    action, _, _ = ea.decide("BTC-USD", side=1, unrealized_pct=0.02,
                             ml_pos=-0.008, regime=_regime(), atr_pct=0.005)
    assert action == "hold"          # not losing beyond the buffer → never cut


def test_holds_loser_when_model_expects_recovery():
    ea = ExitAdvisor()
    action, _, _ = ea.decide("BTC-USD", side=1, unrealized_pct=-0.01,
                             ml_pos=0.012, regime=_regime(), atr_pct=0.005)
    assert action == "hold"


def test_cut_works_for_shorts_position_frame():
    ea = ExitAdvisor()
    # short losing (price rose) and model expects price to keep rising → cut
    action, _, exp = ea.decide("ETH-USD", side=-1, unrealized_pct=-0.012,
                               ml_pos=-0.01, regime=_regime("bull"), atr_pct=0.006)
    assert action == "cut"


def test_counterfactual_learning_updates_state_value():
    tunables.update({"exit_horizon_sec": 1800})
    ea = ExitAdvisor()
    # a long recorded at 100; price later fell to 98 → position-frame return < 0
    ea.record("BTC-USD", 1, 100.0, -0.01, -0.005, _regime(), 0.005)
    ea.pending[-1] = (time.time() - 99999,) + ea.pending[-1][1:]   # mature it
    n = ea.score_pending(lambda p, ts: 98.0)
    assert n == 1
    assert ea.stats()["states_learned"] == 1
    # the learned value for that state should be negative (holding lost money)
    assert any(w["mean_next_return"] < 0 for w in ea.stats()["worst_hold_states"]) \
        or ea.stats()["updates"] == 1


def test_learned_value_makes_advisor_cut_a_marginal_case():
    tunables.update({"exit_cut_threshold": 0.004, "exit_min_loss_pct": 0.003,
                     "exit_ml_weight": 0.0, "exit_horizon_sec": 1800})
    ea = ExitAdvisor()
    reg = _regime()
    # teach the state that holding here loses ~1% next horizon, many times
    for _ in range(25):
        ea.record("BTC-USD", 1, 100.0, -0.01, 0.0, reg, 0.005)
        ea.pending[-1] = (time.time() - 99999,) + ea.pending[-1][1:]
        ea.score_pending(lambda p, ts: 99.0)      # -1% each time
    # with ml weight 0, the cut is driven purely by the LEARNED value
    action, _, exp = ea.decide("BTC-USD", 1, -0.01, 0.0, reg, 0.005)
    assert action == "cut"
    tunables.update({"exit_ml_weight": 1.0})


def test_exit_advisor_capture_restore_roundtrip():
    ea = ExitAdvisor()
    ea.record("BTC-USD", 1, 100.0, -0.01, -0.005, _regime(), 0.005)
    ea.v[("with", "big_loss", "adv", "normal")] = (7, -0.009)
    snap = ea.capture()
    ea2 = ExitAdvisor()
    ea2.restore(snap)
    assert ea2.v[("with", "big_loss", "adv", "normal")] == (7, -0.009)
    assert len(ea2.pending) == 1


# ---------------- direction learner ----------------

def test_direction_bias_favours_side_that_paid():
    dl = DirectionLearner()
    bull = "bull/normal"
    for _ in range(20):
        dl.on_trade_closed({"regime_at_entry": bull, "side": -1, "qty": 1,
                            "entry": 100, "pnl": -3})   # shorts lose
        dl.on_trade_closed({"regime_at_entry": bull, "side": 1, "qty": 1,
                            "entry": 100, "pnl": 2})     # longs win
    assert dl.bias(bull, 0.0) > 0                        # pushed toward long


def test_direction_bias_can_flip_marginal_call():
    dl = DirectionLearner()
    bull = "bull/normal"
    for _ in range(20):
        dl.on_trade_closed({"regime_at_entry": bull, "side": -1, "qty": 1,
                            "entry": 100, "pnl": -4})
        dl.on_trade_closed({"regime_at_entry": bull, "side": 1, "qty": 1,
                            "entry": 100, "pnl": 3})
    new, flipped = dl.adjust(bull, -0.05, 0.2)           # marginal short
    assert flipped and new > 0


def test_direction_bias_capped_and_ignored_without_evidence():
    dl = DirectionLearner()
    assert dl.bias("unknown", 0.0) == 0.0                # no data → no bias
    bull = "bull/normal"
    for _ in range(30):
        dl.on_trade_closed({"regime_at_entry": bull, "side": 1, "qty": 1,
                            "entry": 100, "pnl": 50})     # huge long edge
        dl.on_trade_closed({"regime_at_entry": bull, "side": -1, "qty": 1,
                            "entry": 100, "pnl": -50})
    tunables.update({"direction_bias_cap": 0.25})
    assert abs(dl.bias(bull, 0.0)) <= 0.25 + 1e-9        # never exceeds the cap


def test_mtf_veto_blocks_fighting_strong_trend():
    dl = DirectionLearner()
    tunables.update({"mtf_veto_align": 0.75})
    v, why = dl.veto(-1, 0.9)                            # short vs strong bull
    assert v and "short vetoed" in why
    v2, _ = dl.veto(1, 0.9)                              # long agrees → allowed
    assert not v2
    v3, why3 = dl.veto(1, -0.9)                          # long vs strong bear
    assert v3 and "long vetoed" in why3


def test_mtf_veto_ignores_weak_alignment():
    dl = DirectionLearner()
    tunables.update({"mtf_veto_align": 0.75})
    assert dl.veto(-1, 0.3)[0] is False                 # weak trend → no veto


def test_direction_capture_restore_roundtrip():
    dl = DirectionLearner()
    dl.on_trade_closed({"regime_at_entry": "bull/normal", "side": 1,
                        "qty": 1, "entry": 100, "pnl": 5})
    snap = dl.capture()
    dl2 = DirectionLearner()
    dl2.restore(snap)
    assert dl2.edge.get(("bull/normal", 1)) is not None
    assert dl2.n_updates == 1
