"""Phase 2: auxiliary MULTI-HORIZON heads on the online model.

The primary median head still labels at the 30-min horizon. Extra aux heads
(fast 5-min, slow 2-h) predict the same snapshot's return at other horizons as
auxiliary tasks on top of the SHARED hidden layer. They must:
  * exist and be trainable via update_aux WITHOUT disturbing the primary head's
    n_updates / replay buffer / directional-accuracy window,
  * learn their own horizon's mapping (aux head predicts sign of its target),
  * flow through the Committee,
  * round-trip through persistence,
  * be driven by the Learner's pending_aux queue in _train_aux_heads.
"""
import os, sys, random, time
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn.online_model import TinyMLP, Committee, N_IN, AUX_HEADS


def _aux_out(m, x, head):
    import math
    h, _ = m._fwd(m._standardize(x))
    return math.tanh(sum(w * hi for w, hi in zip(m.aW[head], h)) + m.ab[head])


def test_aux_heads_exist():
    m = TinyMLP()
    assert len(m.aW) == AUX_HEADS
    assert len(m.ab) == AUX_HEADS
    assert all(len(row) == len(m.W1) for row in m.aW)


def test_update_aux_does_not_touch_primary_head_bookkeeping():
    m = TinyMLP()
    before_n = m.n_updates
    before_replay = len(m.replay)
    before_acc = len(m.acc_window)
    x = [0.5] + [0.0] * (N_IN - 1)
    for _ in range(50):
        m.update_aux(x, 0.003, head=0)
    assert m.n_updates == before_n          # aux training never bumps n_updates
    assert len(m.replay) == before_replay   # aux training never fills replay
    assert len(m.acc_window) == before_acc  # nor the dir-accuracy window


def test_aux_head_learns_its_horizon_sign():
    m = TinyMLP()
    r = random.Random(3)
    # feature 0 drives the aux-horizon return; head 0 should learn the sign.
    for _ in range(700):
        f0 = r.gauss(0, 1)
        x = [f0] + [r.gauss(0, 1) for _ in range(N_IN - 1)]
        m.update_aux(x, 0.003 * f0, head=0)
    pos = _aux_out(m, [1.5] + [0.0] * (N_IN - 1), 0)
    neg = _aux_out(m, [-1.5] + [0.0] * (N_IN - 1), 0)
    assert pos > neg                        # monotone in the driving feature


def test_committee_update_aux_trains_every_member():
    c = Committee(n_members=3)
    x = [0.7] + [0.0] * (N_IN - 1)
    baseline = [_aux_out(m, x, 1) for m in c.members]
    for _ in range(30):
        c.update_aux(x, 0.004, head=1)
    after = [_aux_out(m, x, 1) for m in c.members]
    assert all(a != b for a, b in zip(after, baseline))  # every member moved


def test_aux_heads_round_trip_through_persistence():
    from app import persistence as P
    m = TinyMLP()
    for _ in range(20):
        m.update_aux([0.3] + [0.0] * (N_IN - 1), 0.002, head=0)
    d = P._dump_mlp(m)
    assert "aW" in d and "ab" in d
    m2 = TinyMLP()
    assert P._load_mlp(m2, d)
    assert m2.aW == m.aW
    assert m2.ab == m.ab


def test_load_tolerates_snapshot_without_aux_heads():
    from app import persistence as P
    m = TinyMLP()
    d = P._dump_mlp(m)
    d.pop("aW"); d.pop("ab"); d.pop("gaW", None); d.pop("gab", None)
    m2 = TinyMLP()
    fresh_aW = [row[:] for row in m2.aW]
    assert P._load_mlp(m2, d)            # old snapshot still loads
    assert m2.aW == fresh_aW            # fresh aux heads kept


def test_learner_trains_aux_heads_from_pending_aux():
    from app.learn.loop import learner, AUX_HORIZONS
    from app.learn.online_model import committee
    now = time.time()

    class FakeMkt:
        candles = {}
        def price(self, p):
            return 100.0

    # a matured aux sample for each horizon (ts far enough in the past)
    learner.pending_aux = deque(maxlen=6000)
    for head, hz in AUX_HORIZONS:
        ts = now - hz - 10
        learner.pending_aux.append((ts, "BTC-USD", [0.1] * N_IN, head, hz))
    # prices so p0/p1 resolve
    learner.price_history["BTC-USD"] = [
        (now - hz - 10, 100.0) for _, hz in AUX_HORIZONS
    ] + [(now - hz - 10 + hz, 101.0) for _, hz in AUX_HORIZONS]

    trained = learner._train_aux_heads(FakeMkt())
    assert trained >= 1
    # matured samples were consumed
    assert len(learner.pending_aux) == 0
