"""Regression: the online model must actually accumulate training updates.

Root cause it guards: ML feature snapshots are only labeled ML_HORIZON_SEC
(30 min) after they're recorded, but the process restarts far more often than
that. The pending-label queue + price history live in memory; if they aren't
persisted, every sample is wiped before it matures and the model NEVER trains
(n_updates stuck at 0). These tests pin the persistence + maturation path.
"""
import os, sys, time, tempfile
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn.online_model import N_IN as _N_IN


def test_pending_ml_and_price_history_survive_snapshot(tmp_path, monkeypatch):
    import app.persistence as P
    from app import settings
    from app.learn.loop import learner

    now = time.time()
    ts0 = now - 1900
    x = [0.1] * _N_IN
    learner.pending_ml = deque([(ts0, "BTC-USD", x, 0.0)], maxlen=2000)
    learner.price_history = {"BTC-USD": [(ts0, 100.0), (now, 101.0)]}

    snap = tmp_path / "state.json"
    monkeypatch.setattr(P, "STATE_PATH", str(snap), raising=False)
    monkeypatch.setattr(settings, "get", lambda k, *a: True)
    assert P.save()

    # simulate a restart wiping in-memory working state
    learner.pending_ml = deque(maxlen=2000)
    learner.price_history = {}
    assert P.load()

    assert len(learner.pending_ml) == 1
    assert learner.price_history.get("BTC-USD")
    # tuples restored (not lists) so downstream unpacking stays correct
    assert isinstance(learner.price_history["BTC-USD"][0], tuple)


def test_matured_sample_trains_model_after_restart(tmp_path, monkeypatch):
    import app.persistence as P
    from app import settings
    from app.learn.loop import learner
    from app.learn.online_model import model

    now = time.time()
    ts0 = now - 1900                      # already past the 30-min horizon
    learner.pending_ml = deque([(ts0, "BTC-USD", [0.1] * _N_IN, 0.0)], maxlen=2000)
    learner.price_history = {"BTC-USD": [(ts0, 100.0), (now, 101.2)]}

    snap = tmp_path / "state.json"
    monkeypatch.setattr(P, "STATE_PATH", str(snap), raising=False)
    monkeypatch.setattr(settings, "get", lambda k, *a: True)
    assert P.save()
    learner.pending_ml = deque(maxlen=2000)
    learner.price_history = {}
    assert P.load()

    class FakeMkt:
        candles = {"BTC-USD": [[ts0, 100, 100, 100, 100.0, 1],
                               [ts0 + 1800, 101, 101, 101, 101.0, 1]]}
        def price(self, p):
            return 101.2

    before = model.n_updates
    trained = learner._train_online_model(FakeMkt())
    assert trained >= 1
    assert model.n_updates > before      # the model actually learned


def test_stale_feature_dim_pending_ml_is_dropped_not_crash():
    """Restart after N_IN grew used to IndexError the orchestrator tick:
    restored 17-d samples hit an 18-input net inside TinyMLP._sgd."""
    from app.learn.loop import learner
    from app.learn.online_model import model, N_IN
    now = time.time()
    ts0 = now - 1900
    orig_pend = learner.pending_ml
    orig_ph = learner.price_history
    try:
        stale = [0.1] * (N_IN - 1)
        fresh = [0.1] * N_IN
        learner.pending_ml = deque(
            [(ts0, "BTC-USD", stale, 0.0),
             (ts0, "ETH-USD", fresh, 0.0)],
            maxlen=2000)
        learner.price_history = {
            "BTC-USD": [(ts0, 100.0), (now, 101.0)],
            "ETH-USD": [(ts0, 200.0), (now, 202.0)],
        }

        class FakeMkt:
            candles = {}
            def price(self, p):
                return 101.0 if p == "BTC-USD" else 202.0

        before = model.n_updates
        trained = learner._train_online_model(FakeMkt())
        assert trained >= 1
        assert model.n_updates > before
        assert all(len(x) == N_IN for (_, _, x, _) in learner.pending_ml)
    finally:
        learner.pending_ml = orig_pend
        learner.price_history = orig_ph


def test_stale_pending_ml_filtered_on_load(tmp_path, monkeypatch):
    import json
    import app.persistence as P
    from app import settings
    from app.learn.loop import learner
    from app.learn.online_model import N_IN

    now = time.time()
    good = (now - 100, "ETH-USD", [0.1] * N_IN, 0.0)
    learner.pending_ml = deque([good], maxlen=2000)
    snap = tmp_path / "state.json"
    monkeypatch.setattr(P, "STATE_PATH", str(snap), raising=False)
    monkeypatch.setattr(settings, "get", lambda k, *a: True)
    assert P.save()
    with open(snap) as f:
        s = json.load(f)
    s.setdefault("learner", {})["pending_ml"] = [
        [now - 100, "BTC-USD", [0.1] * (N_IN - 1), 0.0],
        list(good),
    ]
    snap.write_text(json.dumps(s))
    learner.pending_ml = deque(maxlen=2000)
    assert P.load()
    assert len(learner.pending_ml) == 1
    assert len(learner.pending_ml[0][2]) == N_IN
