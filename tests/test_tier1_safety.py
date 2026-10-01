"""Tier-1 safety-layer tests (NOFX-inspired):
  * peak-giveback trailing exit (price-basis, arms after a real move, min-hold)
  * guardian safe mode + per-hour entry throttle
  * decision audit trail (db.log_decision / recent_decisions)
  * launch preflight
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.execution.paper import PaperBroker
from app.guardian import Guardian
from app import tunables, db


class _Mkt:
    def __init__(self, px):
        self._px = dict(px)
        self.candles = {}
        self.healthy = True
        self.last_update = time.time()
    def price(self, p):
        return self._px.get(p)


# ---------------- peak-giveback exit ----------------

def test_giveback_exits_winner_after_peak_pullback():
    tunables.update({"trail_giveback_pct": 0.35, "trail_giveback_arm_pct": 0.01,
                     "min_hold_sec": 0})
    b = PaperBroker(); b.cash = 100_000.0
    b.open("BTC-USD", 1, 10_000.0, 100.0, 80.0, 500.0, "t")  # far stop/target
    # rally to 120 (peak gain +20/entry = 20% >> arm), then give back to 108
    b.manage(_Mkt({"BTC-USD": 120.0}), 2.5, lambda p: None)   # sets water=120
    assert "BTC-USD" in b.positions                            # still open at peak
    b.manage(_Mkt({"BTC-USD": 108.0}), 2.5, lambda p: None)   # gave back 60% > 35%
    assert "BTC-USD" not in b.positions
    assert any(t.get("exit_reason") == "peak-giveback" for t in b.closed_trades)


def test_giveback_does_not_arm_before_real_move():
    tunables.update({"trail_giveback_pct": 0.35, "trail_giveback_arm_pct": 0.02,
                     "min_hold_sec": 0})
    b = PaperBroker(); b.cash = 100_000.0
    b.open("BTC-USD", 1, 10_000.0, 100.0, 80.0, 500.0, "t")
    # peak gain only +1 (1% < 2% arm) then pulls back — must NOT giveback-exit
    b.manage(_Mkt({"BTC-USD": 101.0}), 2.5, lambda p: None)
    b.manage(_Mkt({"BTC-USD": 100.2}), 2.5, lambda p: None)
    assert "BTC-USD" in b.positions


def test_giveback_never_exits_into_a_loss():
    tunables.update({"trail_giveback_pct": 0.35, "trail_giveback_arm_pct": 0.01,
                     "min_hold_sec": 0})
    b = PaperBroker(); b.cash = 100_000.0
    b.open("BTC-USD", 1, 10_000.0, 100.0, 80.0, 500.0, "t")
    b.manage(_Mkt({"BTC-USD": 120.0}), 2.5, lambda p: None)   # peak +20
    # crash below entry: giveback must not fire (that's the stop's job); price
    # 95 is above the 80 hard stop so the position should simply stay open.
    b.manage(_Mkt({"BTC-USD": 95.0}), 2.5, lambda p: None)
    assert "BTC-USD" in b.positions


def test_giveback_respects_min_hold():
    tunables.update({"trail_giveback_pct": 0.35, "trail_giveback_arm_pct": 0.01,
                     "min_hold_sec": 10_000})
    b = PaperBroker(); b.cash = 100_000.0
    b.open("BTC-USD", 1, 10_000.0, 100.0, 80.0, 500.0, "t")
    b.manage(_Mkt({"BTC-USD": 120.0}), 2.5, lambda p: None)
    b.manage(_Mkt({"BTC-USD": 108.0}), 2.5, lambda p: None)
    assert "BTC-USD" in b.positions                            # too young to giveback
    tunables.update({"min_hold_sec": 300})                     # restore default


def test_giveback_works_for_shorts_price_basis():
    tunables.update({"trail_giveback_pct": 0.35, "trail_giveback_arm_pct": 0.01,
                     "min_hold_sec": 0})
    b = PaperBroker(); b.cash = 100_000.0
    b.open("BTC-USD", -1, 10_000.0, 100.0, 130.0, 10.0, "t")   # wide stop/target
    b.manage(_Mkt({"BTC-USD": 80.0}), 2.5, lambda p: None)     # short peak gain +20
    assert "BTC-USD" in b.positions
    b.manage(_Mkt({"BTC-USD": 92.0}), 2.5, lambda p: None)     # gave back 60%
    assert "BTC-USD" not in b.positions
    assert any(t.get("exit_reason") == "peak-giveback" for t in b.closed_trades)
    tunables.update({"min_hold_sec": 300})


# ---------------- guardian ----------------

def test_safe_mode_engages_and_clears():
    tunables.update({"safe_mode_fail_threshold": 3, "safe_mode_recover_sec": 0})
    g = Guardian()
    for _ in range(3):
        g.on_cycle(False, market=_Mkt({}), error="boom")
    assert g.safe_mode
    ok, why = g.can_enter()
    assert not ok and "safe mode" in why
    # a healthy tick with recover dwell 0 clears immediately
    g.on_cycle(True, market=_Mkt({}))
    assert not g.safe_mode
    ok, _ = g.can_enter()
    assert ok


def test_safe_mode_on_stale_data():
    tunables.update({"safe_mode_fail_threshold": 3, "data_stale_sec": 60})
    g = Guardian()
    m = _Mkt({}); m.last_update = time.time() - 999    # stale
    g.on_cycle(True, market=m)
    assert g.safe_mode
    ok, why = g.can_enter()
    assert not ok and "stale" in why


def test_entry_rate_throttle():
    tunables.update({"max_entries_per_hour": 3, "safe_mode_fail_threshold": 99})
    g = Guardian()
    for _ in range(3):
        assert g.can_enter()[0]
        g.note_entry()
    ok, why = g.can_enter()
    assert not ok and "entry-rate" in why
    assert g.entries_last_hour() == 3
    tunables.update({"max_entries_per_hour": 12})


# ---------------- decision audit trail ----------------

def test_decision_audit_roundtrip():
    db.log_decision(42, "ETH-USD", 1, 0.31, 0.62, 0.9, "skip",
                    "cooldown", size_pre=1234.0, size_post=0.0,
                    regime="bull/normal", votes={"trend": 0.5})
    db.flush()
    rows = db.recent_decisions(20, product="ETH-USD")
    assert rows and rows[0]["action"] == "skip"
    assert rows[0]["reason"] == "cooldown"
    assert rows[0]["votes"].get("trend") == 0.5
    assert rows[0]["size_pre"] == 1234.0


def test_decision_audit_action_filter():
    db.log_decision(43, "SOL-USD", -1, -0.4, 0.7, 1.0, "enter",
                    "opened", size_pre=5000.0, size_post=4800.0)
    db.flush()
    rows = db.recent_decisions(20, action="enter")
    assert all(r["action"] == "enter" for r in rows)
    assert any(r["product"] == "SOL-USD" for r in rows)


# ---------------- preflight ----------------

def test_preflight_reports_checks():
    g = Guardian()
    rep = g.preflight()
    names = {c["check"] for c in rep["checks"]}
    assert {"market_feed_healthy", "history_sufficient", "model_input_dim",
            "risk_not_killed", "universe_nonempty"} <= names
    assert rep["ok"] in (True, False)
    assert g.preflight_ok is rep["ok"]


def test_orchestrator_starts_paused_when_setting_is_persisted(monkeypatch):
    from app import settings
    from app.orchestrator import Orchestrator

    monkeypatch.setattr(settings, "get", lambda key: key == "trading_paused")

    assert Orchestrator().running is False


def test_pause_and_resume_persist_operator_setting(monkeypatch):
    from app import main, settings

    changes = []
    monkeypatch.setattr(settings, "update", lambda values: changes.append(values))
    monkeypatch.setattr(main.db, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.risk, "killed", False)
    monkeypatch.setattr(main.risk, "halted_today", False)

    assert main.pause()["trading_enabled"] is False
    assert changes[-1] == {"trading_paused": True}
    assert main.orch.running is False

    assert main.resume()["trading_enabled"] is True
    assert changes[-1] == {"trading_paused": False}
    assert main.orch.running is True
