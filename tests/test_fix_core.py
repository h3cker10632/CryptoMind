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


def test_feed_prices_the_cores_picks_for_the_core_only(monkeypatch):
    """A universe-wide champion's picks are priced through the feed's watch
    list for the core alone (price() is what the other sleeves trade on): the
    bid/ask mid, none from a thin book, the last mid kept through a failed
    poll until it ages out, dropped once no longer watched."""
    from app.data import market as M
    from app.strategies.core import live_price
    quotes = {"XRP-USD": (2.4, 2.6)}

    class Resp:
        def __init__(self, q):
            self.q = q

        def raise_for_status(self):
            if self.q is None:
                raise RuntimeError("404")

        def json(self):
            return {"price": "9.9", "bid": str(self.q[0]), "ask": str(self.q[1])}

    class Client:
        async def get(self, url, params=None, timeout=None):
            return Resp(quotes.get(url.split("/")[-2]))

    m = M.MarketData()
    m.watch = {"XRP-USD", "GONE-USD"}
    asyncio.run(m.refresh_watch(Client()))
    assert m.watch_price("XRP-USD") == 2.5 and m.watch_price("GONE-USD") is None
    assert live_price(m, "XRP-USD") == 2.5
    assert m.price("XRP-USD") is None and "XRP-USD" not in m.tickers   # not the bot's
    quotes["XRP-USD"] = (1.0, 2.6)                                      # walked book: ignored
    asyncio.run(m.refresh_watch(Client()))
    quotes.pop("XRP-USD")                                               # failed poll: keep mid
    asyncio.run(m.refresh_watch(Client()))
    assert m.watch_price("XRP-USD") == 2.5
    now = M.time.time()
    monkeypatch.setattr(M.time, "time", lambda: now + m.WATCH_MAX_AGE)  # ...until it ages out
    assert m.watch_price("XRP-USD") is None
    m.watch = set()
    asyncio.run(m.refresh_watch(Client()))
    assert "XRP-USD" not in m.watch_px


def test_core_loop_values_holdings_and_trades_picks_at_market_after_restart(monkeypatch):
    """After a restart the core's out-of-universe holdings go on the watch
    list at once and are priced before equity is measured; a new pick is
    priced before it is traded; targets are computed once."""
    import pytest
    from app import orchestrator as O
    from app.strategies.core import core
    mkt = type("M", (), {"healthy": True, "watch": set(), "px": {}})()
    mkt.price = lambda a: mkt.px.get(a)
    monkeypatch.setattr(O, "market", mkt)
    monkeypatch.setattr(core, "positions", {"DOGE-USD": {"qty": 1e5, "entry": 0.1, "opened": 0}})
    monkeypatch.setattr(core, "enabled", lambda: True)
    monkeypatch.setattr(core, "_settings", lambda: (0.5, [], False))
    monkeypatch.setattr(core, "tracking_alert", lambda: None)
    calls = []
    monkeypatch.setattr(core, "targets", lambda eq, daily, now=None: calls.append(("t", eq)) or
                        {"XRP-USD": 25_000.0, "DOGE-USD": 0.0})
    monkeypatch.setattr(core, "rebalance", lambda b, m, eq, daily, now=None, tgt=None:
                        calls.append(("r", tgt, m.price("XRP-USD"))) or [])
    waits, feed = [], [("DOGE-USD", 0.3), ("XRP-USD", 2.5)]

    async def sleep(s):
        waits.append(s)
        if s == 180:
            assert mkt.watch == {"DOGE-USD"}            # seeded before the first pass
        if s == 10:
            mkt.px.update([feed.pop(0)])                 # the feed's next poll prices one
        if s == 3600:
            raise asyncio.CancelledError
    monkeypatch.setattr(O.asyncio, "sleep", sleep)
    o = O.Orchestrator.__new__(O.Orchestrator)
    o._total_equity = lambda m: 50_000.0 + 1e5 * (m.price("DOGE-USD") or 0.1)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(o.core_loop())
    assert waits == [180, 10, 10, 3600] and mkt.watch == {"XRP-USD", "DOGE-USD"}
    assert calls == [("t", 80_000.0),                    # DOGE at market, not entry ($60k)
                     ("r", {"XRP-USD": 25_000.0, "DOGE-USD": 0.0}, 2.5)]
