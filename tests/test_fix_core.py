"""Review fixes: a universe-wide (ML) champion trades and reserves its slice,
unpriced picks count as off target, the research loop keeps an ML champion's
saved weights fresh, failed tool runs back off and run at lower priority, and
the ex-core drawdown peak survives a restart."""
import asyncio
import os
import time

_ML = ("ml_rank_top20_h7", {"selection": "ml_rank"})


class _Mkt:
    def __init__(self, px):
        self._px = dict(px)

    def price(self, p):
        return self._px.get(p)


def _settings(monkeypatch, **vals):
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda k, *a: vals[k] if k in vals else real(k))


def _ml_champion(monkeypatch, weights):
    from app.ml import strategies as MLS
    from app.strategies import core as core_mod
    _settings(monkeypatch, core_strategy="champion", core_allocation_pct=50,
              exploration_enabled=False, core_tracking_alert_days=2,
              core_tracking_alert_te=10.0)
    monkeypatch.setattr(core_mod, "_champion", lambda: _ML)
    monkeypatch.setattr(MLS, "latest_saved",
                        lambda cfg, today, max_age_days=3: (today - 1, dict(weights)))


def test_ml_champion_without_assets_trades_and_reserves_its_slice(monkeypatch):
    from app.execution.paper import PaperBroker
    from app.strategies.core import CoreBook
    _ml_champion(monkeypatch, {"SOL-USD": 0.3, "BTC-USD": 0.4})
    c = CoreBook()
    assert c.targets(100_000.0, {}) == {"SOL-USD": 15_000.0, "BTC-USD": 20_000.0}
    assert c.bot_equity(100_000.0, _Mkt({})) == 50_000.0
    b = PaperBroker()
    b.cash = 100_000.0
    acts = c.rebalance(b, _Mkt({"SOL-USD": 100.0, "BTC-USD": 50_000.0}), 100_000.0, {})
    assert set(c.positions) == {"SOL-USD", "BTC-USD"} and len(acts) == 2


def test_unpriced_pick_is_logged_and_counted_off_target(monkeypatch):
    from app.execution.paper import PaperBroker
    from app.strategies import core as core_mod
    _ml_champion(monkeypatch, {"XRP-USD": 0.5})
    logs = []
    monkeypatch.setattr(core_mod.db, "log_event", lambda kind, msg: logs.append(msg))
    c = core_mod.CoreBook()
    b = PaperBroker()
    b.cash = 100_000.0
    now = time.time()
    assert c.rebalance(b, _Mkt({}), 100_000.0, {}, now=now) == []
    assert any("XRP-USD" in m and "no live price" in m for m in logs)
    assert c.tracking[-1]["off_target"] == ["XRP-USD"]
    c.rebalance(b, _Mkt({}), 100_000.0, {}, now=now + core_mod.DAILY)
    assert "XRP-USD" in (c.tracking_alert() or "")


def test_held_coin_that_lost_its_price_is_flagged_once_a_day(monkeypatch):
    from app.execution.paper import PaperBroker
    from app.strategies import core as core_mod
    _ml_champion(monkeypatch, {})                       # champion dropped it: target 0
    logs = []
    monkeypatch.setattr(core_mod.db, "log_event", lambda kind, msg: logs.append(msg))
    c = core_mod.CoreBook()
    c.positions = {"DOGE-USD": {"qty": 100_000.0, "entry": 0.2, "opened": 0}}
    b = PaperBroker()
    b.cash = 80_000.0
    now = time.time()
    for k in range(3):                                  # hourly, same day
        c.rebalance(b, _Mkt({}), 100_000.0, {}, now=now + k * 3600)
    assert sum("DOGE-USD" in m and "held" in m for m in logs) == 1
    assert c.positions["DOGE-USD"]["qty"] == 100_000.0  # no price: never sold blind
    assert c.tracking[-1]["actual"]["DOGE-USD"] == 0.4  # valued at entry, like value()
    assert c.tracking[-1]["off_target"] == ["DOGE-USD"]


class _Proc:
    pid = 4242

    def __init__(self, code):
        self.code = code

    async def wait(self):
        return self.code


def _fake_exec(monkeypatch, code):
    from app import orchestrator as O
    calls = []

    async def fake(*args, **kw):
        calls.append(kw)
        return _Proc(code)
    monkeypatch.setattr(O.asyncio, "create_subprocess_exec", fake)
    return calls


def test_failed_tool_backs_off_and_runs_niced(monkeypatch):
    from app import orchestrator as O
    _settings(monkeypatch, research_extras_interval_sec=604800)
    calls = _fake_exec(monkeypatch, 1)
    niced = []
    monkeypatch.setattr(os, "setpriority", lambda *a: niced.append(a), raising=False)
    o = O.Orchestrator.__new__(O.Orchestrator)
    run = lambda now: asyncio.run(o._run_tool_weekly(
        "ml_lab.py", "ML lab", 0, "research_extras_interval_sec", now))
    t = 1_000_000_000.0
    assert run(t) is False and len(calls) == 1
    assert run(t + 600) is False and len(calls) == 1         # no retry storm
    assert run(t + 6 * 3600 + 1) is False and len(calls) == 2
    if os.name != "nt":                   # lowered after spawn, no preexec_fn
        assert "preexec_fn" not in calls[0] and niced[0][1:] == (4242, 10)


def test_ml_champion_research_loop_due_daily_while_the_core_trades_it(monkeypatch):
    from app import orchestrator as O
    from app.engine import challengers as C
    from app.strategies.core import core
    _settings(monkeypatch, research_loop_interval_sec=604800)
    calls = _fake_exec(monkeypatch, 0)
    now = 1_000_000_000.0
    ran = {"at": now - 1.5 * 86400}
    monkeypatch.setattr(C, "_load", lambda path, default: {"last_report": {"ran_at": ran["at"]}})
    monkeypatch.setattr(type(core), "follows_champion", staticmethod(lambda: True))
    monkeypatch.setattr(type(core), "enabled", lambda self: True)
    o = O.Orchestrator.__new__(O.Orchestrator)
    loop = lambda: asyncio.run(o._maybe_run_research_loop(now))
    monkeypatch.setattr(C, "champion", lambda: ("trend", {"selection": "trend"}))
    loop()
    assert calls == []                                        # weekly interval holds
    monkeypatch.setattr(C, "champion", lambda: _ML)
    ran["at"] = now - 12 * 3600                               # ran today: not forced
    loop()
    assert calls == []
    ran["at"] = now - 1.5 * 86400                             # a day old: refresh it
    loop()
    assert len(calls) == 1
    monkeypatch.setattr(type(core), "enabled", lambda self: False)
    loop()                                                    # core off: weekly holds
    assert len(calls) == 1


def test_full_data_sync_comes_due_even_while_the_loops_ingest_hourly(monkeypatch, tmp_path):
    """core_loop / exploration_loop append to ingest_log.jsonl every hour; the
    daily sync is scheduled from its OWN stamp (tools/data_sync.py)."""
    from app import orchestrator as O
    from app.data import store
    monkeypatch.setattr(store, "STORE", str(tmp_path))
    (tmp_path / "ingest_log.jsonl").write_text("{}\n")       # touched just now
    _settings(monkeypatch, data_sync_interval_sec=86400)
    seen = []

    async def fake_weekly(self, script, label, last, key, now=None):
        seen.append(last)
        return False
    monkeypatch.setattr(O.Orchestrator, "_run_tool_weekly", fake_weekly)
    o = O.Orchestrator.__new__(O.Orchestrator)
    asyncio.run(o._maybe_sync_data())
    assert seen == [0]                                        # no stamp yet: due
    (tmp_path / "last_sync").write_text("1")
    asyncio.run(o._maybe_sync_data())
    assert abs(seen[1] - os.path.getmtime(tmp_path / "last_sync")) < 1e-6


def test_ex_core_peak_below_start_cash_survives_restart(monkeypatch, tmp_path):
    from app import persistence
    from app.risk.manager import risk
    _settings(monkeypatch, carry_equity=True)
    monkeypatch.setattr(persistence, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(risk, "peak_equity", 92_000.0)
    monkeypatch.setattr(risk, "kill_arm_peak", 92_000.0)
    monkeypatch.setattr(risk, "dd_basis", "active_ex_core")
    assert persistence.save() is True
    risk.peak_equity = 0.0
    assert persistence.load() is True
    assert risk.peak_equity == 92_000.0 and risk.kill_arm_peak == 92_000.0


def test_held_coin_on_target_but_unpriced_counts_off_target(monkeypatch):
    from app.execution.paper import PaperBroker
    from app.strategies import core as core_mod
    _ml_champion(monkeypatch, {"XRP-USD": 0.5})
    monkeypatch.setattr(core_mod.db, "log_event", lambda *a: None)
    c = core_mod.CoreBook()
    c.positions = {"XRP-USD": {"qty": 10_000.0, "entry": 2.5, "opened": 0}}   # $25k: on target
    b = PaperBroker()
    b.cash = 75_000.0
    c.rebalance(b, _Mkt({}), 100_000.0, {}, now=time.time())
    assert c.tracking[-1]["off_target"] == ["XRP-USD"]                       # can't trade or price it
