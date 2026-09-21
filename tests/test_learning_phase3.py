"""Phase 3: COUNTERFACTUAL credit for skipped/declined conviction entries.

An actionable conviction signal the book was gated out of (cooldown,
max-positions, per-hour cap, funding, min-notional) is recorded and later
scored with the NET-of-cost return the trade WOULD have made, teaching the
bandit's voting strategies. This turns throttled-but-wanted decisions into
learning samples — the whole point of the phase (attacking data starvation).
"""
import os, sys, time
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn.loop import Learner, SKIP_EVAL_HORIZON_SEC
from app import tunables


class FakeMkt:
    def __init__(self, price):
        self._p = price
        self.candles = {}
    def price(self, p):
        return self._p


def _fresh_learner():
    L = Learner()
    return L


def test_on_entry_skipped_records_only_valid_candidates():
    tunables.update({"skip_learn_weight": 0.5})
    L = _fresh_learner()
    # valid: has votes for a real strategy
    L.on_entry_skipped("BTC-USD", 1, 100.0, "bull", {"trend": 0.8})
    assert len(L.pending_skips) == 1
    # no votes -> ignored
    L.on_entry_skipped("ETH-USD", 1, 100.0, "bull", {})
    # bad price -> ignored
    L.on_entry_skipped("SOL-USD", 1, 0.0, "bull", {"trend": 0.5})
    # votes only for non-bandit arms -> ignored
    L.on_entry_skipped("XRP-USD", 1, 100.0, "bull", {"not_a_strategy": 1.0})
    assert len(L.pending_skips) == 1
    tunables.reset()


def test_disabled_when_weight_zero():
    tunables.update({"skip_learn_weight": 0.0})
    L = _fresh_learner()
    L.on_entry_skipped("BTC-USD", 1, 100.0, "bull", {"trend": 0.8})
    assert len(L.pending_skips) == 0
    tunables.reset()


def test_winning_counterfactual_rewards_the_voting_arm():
    tunables.update({"skip_learn_weight": 0.5})
    L = _fresh_learner()
    now = time.time()
    ts = now - SKIP_EVAL_HORIZON_SEC - 5
    L.pending_skips = deque([(ts, "BTC-USD", 1, 100.0, "bull", {"trend": 1.0})],
                            maxlen=4000)
    # price rose enough to clear round-trip cost -> a winning long
    L.price_history["BTC-USD"] = [(ts, 100.0),
                                  (ts + SKIP_EVAL_HORIZON_SEC, 105.0)]
    before = L.bandit.arms.get(("bull", "trend"), (0, 0.0, 0.0))[1]
    n = L._score_skips(FakeMkt(105.0))
    assert n == 1
    assert L.skip_attributions == 1
    after = L.bandit.arms.get(("bull", "trend"), (0, 0.0, 0.0))[1]
    assert after > before                      # good skipped trade -> arm up
    assert len(L.pending_skips) == 0
    tunables.reset()


def test_losing_counterfactual_penalizes_the_voting_arm():
    tunables.update({"skip_learn_weight": 0.5})
    L = _fresh_learner()
    now = time.time()
    ts = now - SKIP_EVAL_HORIZON_SEC - 5
    L.pending_skips = deque([(ts, "BTC-USD", 1, 100.0, "bull", {"trend": 1.0})],
                            maxlen=4000)
    # price fell -> a losing long
    L.price_history["BTC-USD"] = [(ts, 100.0),
                                  (ts + SKIP_EVAL_HORIZON_SEC, 94.0)]
    before = L.bandit.arms.get(("bull", "trend"), (0, 0.0, 0.0))[1]
    L._score_skips(FakeMkt(94.0))
    after = L.bandit.arms.get(("bull", "trend"), (0, 0.0, 0.0))[1]
    assert after < before                      # bad skipped trade -> arm down
    tunables.reset()


def test_immature_skip_is_retained():
    tunables.update({"skip_learn_weight": 0.5})
    L = _fresh_learner()
    now = time.time()
    L.pending_skips = deque([(now, "BTC-USD", 1, 100.0, "bull", {"trend": 1.0})],
                            maxlen=4000)
    n = L._score_skips(FakeMkt(100.0))
    assert n == 0
    assert len(L.pending_skips) == 1           # not yet matured -> kept
    tunables.reset()


def test_cost_makes_a_flat_move_a_net_loss():
    """A skipped trade whose price barely moved must be scored as a LOSS after
    round-trip cost — the counterfactual is net-of-cost, like a real fill."""
    tunables.update({"skip_learn_weight": 0.5})
    L = _fresh_learner()
    now = time.time()
    ts = now - SKIP_EVAL_HORIZON_SEC - 5
    L.pending_skips = deque([(ts, "BTC-USD", 1, 100.0, "bull", {"trend": 1.0})],
                            maxlen=4000)
    L.price_history["BTC-USD"] = [(ts, 100.0),
                                  (ts + SKIP_EVAL_HORIZON_SEC, 100.0)]
    before = L.bandit.arms.get(("bull", "trend"), (0, 0.0, 0.0))[1]
    L._score_skips(FakeMkt(100.0))
    after = L.bandit.arms.get(("bull", "trend"), (0, 0.0, 0.0))[1]
    assert after < before                      # 0% gross - cost < 0 net
    tunables.reset()


def test_skips_round_trip_through_persistence():
    import app.persistence as P
    from app.learn.loop import learner
    from app import settings
    import tempfile
    tunables.update({"skip_learn_weight": 0.5})
    now = time.time()
    learner.pending_skips = deque(
        [(now, "BTC-USD", 1, 100.0, "bull", {"trend": 0.8})], maxlen=4000)
    learner.skip_attributions = 7

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.json")
        orig_path = P.STATE_PATH
        orig_get = settings.get
        P.STATE_PATH = path
        settings.get = lambda k, *a: True
        try:
            assert P.save()
            learner.pending_skips = deque(maxlen=4000)
            learner.skip_attributions = 0
            assert P.load()
        finally:
            P.STATE_PATH = orig_path
            settings.get = orig_get
    assert len(learner.pending_skips) == 1
    assert learner.skip_attributions == 7
    tunables.reset()
