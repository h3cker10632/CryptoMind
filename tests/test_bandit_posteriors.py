"""Bandit posterior honesty: silent sleeves, fill-only updates, ML reset.

The live table looked "bad" mostly because (a) 1h net-of-taker labels
manufactured thousands of −60bps observations, (b) evolved kept a stale
+8bps mean with no champion and 52% of the book, (c) ML at 26% acc still
had a 4% floor. These tests pin the contract that each learning cycle
moves those numbers the right way.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _restore_model(n_updates, acc_window):
    from app.learn.online_model import model
    model.n_updates = n_updates
    model.acc_window.clear()
    model.acc_window.extend(acc_window)


def test_ghost_evolved_gets_zero_weight_and_dropped_arm():
    from app.learn.loop import Learner
    from app.learn.evolution import evolution
    orig_c = dict(evolution.champions)
    orig_p = dict(evolution.champion_portfolios)
    try:
        evolution.champions = {}
        evolution.champion_portfolios = {}
        L = Learner()
        L.bandit.update("sideways/normal", "evolved", 0.002)
        L.bandit.update("sideways/normal", "trend", 0.001)
        silent = L._silent_sleeves()
        assert "evolved" in silent
        w = L._allocate("sideways/normal", silent)
        assert w["evolved"] == 0.0
        assert w["trend"] > 0
        assert all(s != "evolved" for (_, s) in L.bandit.arms)
    finally:
        evolution.champions = orig_c
        evolution.champion_portfolios = orig_p


def test_evolved_keeps_weight_when_champion_exists():
    from app.learn.loop import Learner
    from app.learn.evolution import evolution
    orig_c = dict(evolution.champions)
    orig_p = dict(evolution.champion_portfolios)
    try:
        evolution.champions = {"BTC-USD": {"ema_fast": 8}}
        evolution.champion_portfolios = {"BTC-USD": [{"ema_fast": 8}]}
        L = Learner()
        silent = L._silent_sleeves()
        assert "evolved" not in silent
        w = L._allocate("bull", silent)
        assert w["evolved"] > 0
    finally:
        evolution.champions = orig_c
        evolution.champion_portfolios = orig_p


def test_ml_silent_when_below_coin_flip():
    from app.learn.loop import Learner
    from app.learn.online_model import model
    orig_n = model.n_updates
    orig_acc = list(model.acc_window)
    try:
        model.n_updates = 500
        model.acc_window.clear()
        model.acc_window.extend([0] * 200)
        L = Learner()
        silent = L._silent_sleeves()
        assert "ml" in silent
        w = L._allocate("bull", silent)
        assert w["ml"] == 0.0
        assert abs(sum(w.values()) - 1.0) < 1e-3
    finally:
        _restore_model(orig_n, orig_acc)


def test_ml_silent_when_unmeasured():
    from app.learn.loop import Learner
    from app.learn.online_model import model
    orig_n = model.n_updates
    orig_acc = list(model.acc_window)
    try:
        model.n_updates = 10
        model.acc_window.clear()
        L = Learner()
        assert "ml" in L._silent_sleeves()
        assert L._ml_is_broken() is False          # unmeasured ≠ reset
    finally:
        _restore_model(orig_n, orig_acc)


def test_broken_ml_resets_once_then_stops():
    from app.learn.loop import Learner
    from app.learn.online_model import model, committee
    orig_n = model.n_updates
    orig_acc = list(model.acc_window)
    L = Learner()
    try:
        for m in committee.members:
            m.n_updates = 500
            m.acc_window.clear()
            m.acc_window.extend([0] * 200)
        L.bandit.update("bull", "ml", -0.005)
        assert L._maybe_reset_broken_ml() is True
        assert model.n_updates == 0
        assert L._ml_resets == 1
        assert all(s != "ml" for (_, s) in L.bandit.arms)
        # after reset, n=0 so we must not thrash
        assert L._maybe_reset_broken_ml() is False
        assert L._ml_resets == 1
    finally:
        # put a usable committee back so later tests aren't on a wiped net
        committee.reset(base_seed=7)
        _restore_model(orig_n, orig_acc)


def test_signal_stream_feeds_bandit_when_enabled(monkeypatch):
    """The GROSS directional signal-scoring stream now teaches the bandit as a
    discounted secondary teacher (signal_learn_weight>0). A matured signal that
    was directionally RIGHT (price rose after a long) must add a bump to that
    (regime, strategy) arm on top of the per-cycle decay."""
    from app.learn.loop import Learner
    from app.learn.evolution import evolution
    from app import tunables
    orig_c = dict(evolution.champions)
    orig_p = dict(evolution.champion_portfolios)
    L = Learner()
    try:
        tunables.update({"signal_learn_weight": 0.15, "signal_learn_clip": 0.01})
        evolution.champions = {}
        evolution.champion_portfolios = {}
        for _ in range(40):
            L.bandit.update("bull", "trend", 0.002)
        n0, mean0, _ = L.bandit.arms[("bull", "trend")]

        monkeypatch.setattr("app.learn.loop.db.unscored_signals",
                            lambda older: [{"rowid": 1, "product": "BTC-USD",
                                            "ts": 0, "direction": 1,
                                            "strategy": "trend",
                                            "confidence": 1.0,
                                            "regime": "bull"}])
        monkeypatch.setattr("app.learn.loop.db.score_signal", lambda *a, **k: None)
        monkeypatch.setattr("app.learn.loop.db.strategy_scores", lambda *a, **k: [])
        monkeypatch.setattr("app.learn.loop.db.log_event", lambda *a, **k: None)
        monkeypatch.setattr("app.learn.loop.db.abandon_signal", lambda *a, **k: None)
        # price rose from 100 -> 102 over the horizon: a correct long
        monkeypatch.setattr(L, "_price_at",
                            lambda p, ts, m=None: 100.0 if ts == 0 else 102.0)
        monkeypatch.setattr(L, "maybe_evolve", lambda *a, **k: None)
        monkeypatch.setattr(L, "_train_online_model", lambda *a, **k: 0)
        monkeypatch.setattr(L, "_check_drift", lambda *a, **k: None)
        monkeypatch.setattr(L, "_maybe_reset_broken_ml", lambda: False)

        class M:
            tickers = {}
            def price(self, p):
                return 102.0

        L.run(M(), {"label": "bull"})
        n1, mean1, _ = L.bandit.arms[("bull", "trend")]
        # the signal update ADDS an observation on top of decay, so n barely
        # drops (or rises) vs the pure-decay case, and the arm is still positive
        assert n1 > n0 * 0.99                           # gained an obs vs decay-only
        assert mean1 > 0                                # correct long kept it positive
    finally:
        tunables.update({"signal_learn_weight": 0.15})
        evolution.champions = orig_c
        evolution.champion_portfolios = orig_p


def test_signal_stream_disabled_when_weight_zero(monkeypatch):
    """signal_learn_weight=0 restores the old behaviour: the 1h signal stream
    is dashboard-only and does NOT touch the bandit (decay only)."""
    from app.learn.loop import Learner
    from app.learn.evolution import evolution
    from app import tunables
    orig_c = dict(evolution.champions)
    orig_p = dict(evolution.champion_portfolios)
    L = Learner()
    try:
        tunables.update({"signal_learn_weight": 0.0})
        evolution.champions = {}
        evolution.champion_portfolios = {}
        for _ in range(40):
            L.bandit.update("bull", "trend", 0.002)
        n0, mean0, _ = L.bandit.arms[("bull", "trend")]

        monkeypatch.setattr("app.learn.loop.db.unscored_signals",
                            lambda older: [{"rowid": 1, "product": "BTC-USD",
                                            "ts": 0, "direction": 1,
                                            "strategy": "trend",
                                            "confidence": 1.0,
                                            "regime": "bull"}])
        monkeypatch.setattr("app.learn.loop.db.score_signal", lambda *a, **k: None)
        monkeypatch.setattr("app.learn.loop.db.strategy_scores", lambda *a, **k: [])
        monkeypatch.setattr("app.learn.loop.db.log_event", lambda *a, **k: None)
        monkeypatch.setattr("app.learn.loop.db.abandon_signal", lambda *a, **k: None)
        monkeypatch.setattr(L, "_price_at", lambda *a, **k: 100.0)
        monkeypatch.setattr(L, "maybe_evolve", lambda *a, **k: None)
        monkeypatch.setattr(L, "_train_online_model", lambda *a, **k: 0)
        monkeypatch.setattr(L, "_check_drift", lambda *a, **k: None)
        monkeypatch.setattr(L, "_maybe_reset_broken_ml", lambda: False)

        class M:
            tickers = {}
            def price(self, p):
                return 100.0

        L.run(M(), {"label": "bull"})
        n1, mean1, _ = L.bandit.arms[("bull", "trend")]
        assert n1 < n0                                 # decay only, no new obs
        assert abs(mean1) < abs(mean0)                 # idle mean toward 0
        assert L.last_cycle["ghost_weight"] == 0.0
        assert "evolved" in L.last_cycle["silent"]
        assert L.weights["evolved"] == 0.0
    finally:
        tunables.update({"signal_learn_weight": 0.15})
        evolution.champions = orig_c
        evolution.champion_portfolios = orig_p


def test_successive_cycles_shrink_idle_abs_mean(monkeypatch):
    """|mean_bps| of an idle live sleeve must fall every cycle (decay)."""
    from app.learn.loop import Learner
    from app.learn.evolution import evolution
    orig_c = dict(evolution.champions)
    orig_p = dict(evolution.champion_portfolios)
    L = Learner()
    try:
        evolution.champions = {}
        evolution.champion_portfolios = {}
        for _ in range(60):
            L.bandit.update("bull", "trend", 0.004)
        monkeypatch.setattr("app.learn.loop.db.unscored_signals", lambda older: [])
        monkeypatch.setattr("app.learn.loop.db.strategy_scores", lambda *a, **k: [])
        monkeypatch.setattr("app.learn.loop.db.log_event", lambda *a, **k: None)
        monkeypatch.setattr(L, "maybe_evolve", lambda *a, **k: None)
        monkeypatch.setattr(L, "_train_online_model", lambda *a, **k: 0)
        monkeypatch.setattr(L, "_check_drift", lambda *a, **k: None)
        monkeypatch.setattr(L, "_maybe_reset_broken_ml", lambda: False)

        class M:
            tickers = {}
            def price(self, p):
                return 100.0

        abs_means = []
        for _ in range(5):
            L.run(M(), {"label": "bull"})
            abs_means.append(L.last_cycle["mean_abs_bps"])
        assert all(abs_means[i] > abs_means[i + 1] for i in range(len(abs_means) - 1))
        assert L.last_cycle["mean_abs_bps_delta"] < 0
        assert abs_means[-1] < abs_means[0]
    finally:
        evolution.champions = orig_c
        evolution.champion_portfolios = orig_p
