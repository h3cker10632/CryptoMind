"""Efficiency round: limit (maker) orders, chop filter + auto A/B, long-history
fetch pacing, DB pruning, component audit, forward test, optional core
holding."""
import asyncio
import json
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _candles(n=400, seed=0, drift=0.0004, t0=1_700_000_000, vol=0.012, step=3600):
    r = random.Random(seed)
    px, out = 100.0, []
    for i in range(n):
        op = px
        px *= math.exp(drift + r.gauss(0, vol))
        out.append([t0 + i * step, min(op, px) * 0.996, max(op, px) * 1.004, op, px, 1000.0])
    return out


def _universe(k=7, n=400):
    names = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
             "DOGE-USD", "XRP-USD"][:k]
    return {p: _candles(n, seed=i, drift=0.0006 * (i - k / 2) / k) for i, p in enumerate(names)}


class _Mkt:
    def __init__(self, px):
        self._px = dict(px)
        self.candles = {}
        self.tickers = {p: {"price": v} for p, v in px.items()}
        self.healthy = True

    def price(self, p):
        return self._px.get(p)


# ------------------------------------------------------------ costs / broker
def test_round_trip_cost_follows_entry_order_type(monkeypatch):
    from app.execution import costs
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda k: "maker" if k == "entry_order_type" else real(k))
    rt_maker = costs.round_trip_cost()
    monkeypatch.setattr(settings, "get", lambda k: "taker" if k == "entry_order_type" else real(k))
    rt_taker = costs.round_trip_cost()
    assert rt_maker < rt_taker
    assert abs(rt_taker - 2 * costs.taker_side_cost()) < 1e-12


def test_maker_open_fills_at_limit_with_maker_fee():
    from app.execution.paper import PaperBroker
    from app.tunables import tv
    b = PaperBroker(); b.cash = 100_000.0
    pos = b.open("BTC-USD", 1, 10_000.0, 100.0, 90.0, 130.0, "t", maker=True)
    assert pos["entry"] == 100.0                       # no slippage
    assert abs(pos["fees"] - 10_000.0 * tv("maker_fee_rate")) < 1e-9
    assert pos.get("maker_entry")


def test_maker_entry_takes_profit_at_target_as_maker():
    from app.execution.paper import PaperBroker
    from app.tunables import tv
    b = PaperBroker(); b.cash = 100_000.0
    b.open("BTC-USD", 1, 10_000.0, 100.0, 90.0, 110.0, "t", maker=True)
    b.manage(_Mkt({"BTC-USD": 112.0}), 3.0, lambda p: None)
    t = b.closed_trades[-1]
    assert t["exit_reason"] == "take-profit" and t["exit"] == 110.0
    qty = 10_000.0 / 100.0
    expect = qty * 110.0 * (1 - tv("maker_fee_rate")) - 10_000.0 - 10_000.0 * tv("maker_fee_rate")
    assert abs(t["pnl"] - expect) < 1e-6


def test_stop_is_always_taker_even_for_maker_entry():
    from app.execution.paper import PaperBroker
    from app.tunables import tv
    b = PaperBroker(); b.cash = 100_000.0
    b.open("BTC-USD", 1, 10_000.0, 100.0, 95.0, 130.0, "t", maker=True)
    b.manage(_Mkt({"BTC-USD": 94.0}), 3.0, lambda p: None)
    t = b.closed_trades[-1]
    assert t["exit_reason"] == "stop-loss/trail"
    assert t["exit"] < 94.0                            # slippage applied


# ------------------------------------------------------------ orchestrator limits
def _sig(p="BTC-USD", d=1, px=100.0):
    return {"product": p, "direction": d, "price": px, "confidence": 0.7,
            "composite": 0.6, "regime": "bull/normal", "atr": 1.0, "mtf_align": 1.0}


def test_pending_limit_fills_only_when_traded_through(monkeypatch):
    import app.orchestrator as O
    from app.execution.paper import broker
    broker.positions.clear()
    if broker._portfolio is None:
        broker.cash = 100_000.0
    o = O.Orchestrator()
    monkeypatch.setattr(o, "_entries_enabled", lambda: True)
    assert o._place_limit_entry(_sig(), 5_000.0, 97.0, 106.0)
    o._process_pending_entries(_Mkt({"BTC-USD": 100.0}))      # touch: no fill
    assert "BTC-USD" in o.pending_entries and "BTC-USD" not in broker.positions
    o._process_pending_entries(_Mkt({"BTC-USD": 99.9}))       # through: fill
    assert "BTC-USD" not in o.pending_entries
    pos = broker.positions["BTC-USD"]
    assert pos["entry"] == 100.0 and pos.get("maker_entry")
    broker.positions.clear()


def test_pending_limit_expires(monkeypatch):
    import app.orchestrator as O
    from app.tunables import tv
    o = O.Orchestrator()
    monkeypatch.setattr(o, "_entries_enabled", lambda: True)
    o._place_limit_entry(_sig("ETH-USD", -1, 50.0), 2_000.0, 52.0, 44.0)
    o.pending_entries["ETH-USD"]["ts"] -= tv("maker_timeout_sec") + 1
    o._process_pending_entries(_Mkt({"ETH-USD": 49.0}))       # wrong side for a short
    assert not o.pending_entries


def test_pending_limits_hold_position_slots(monkeypatch):
    import app.orchestrator as O
    o = O.Orchestrator()
    monkeypatch.setattr(o, "_max_positions", lambda: 1)
    assert o._place_limit_entry(_sig("BTC-USD"), 1_000.0, 97.0, 106.0)
    assert not o._place_limit_entry(_sig("ETH-USD"), 1_000.0, 97.0, 106.0)


# ------------------------------------------------------------ chop filter
def test_trendiness_random_walk_vs_trend():
    from app.data.market import trendiness
    noise = {f"C{i}": _candles(200, seed=i, drift=0.0) for i in range(9)}
    trend = {f"C{i}": _candles(200, seed=i, drift=0.01, vol=0.004) for i in range(9)}
    assert trendiness(noise) < 0.3
    assert trendiness(trend) > 0.6
    assert trendiness({"A": _candles(200)}) is None           # too few coins


def test_chop_auto_follows_replay_ab(monkeypatch):
    import app.orchestrator as O
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda k: "auto" if k == "chop_filter_mode" else real(k))
    o = O.Orchestrator()
    o._apply_chop_auto({"chop_ab": {"helps_in_both_halves": True}}, announce=False)
    assert o.chop_filter_active()
    o._apply_chop_auto({"chop_ab": {"helps_in_both_halves": False}}, announce=False)
    assert not o.chop_filter_active()


def test_replay_maker_chop_and_ab_report():
    from app.backtest import replay as rp
    U = _universe()
    r = rp.run_replay(U, maker=True)
    assert r["ok"] and r["entry_orders"] == "limit (maker)"
    assert r["limit_orders"] >= r["trades"] - 1 and r["limit_fill_rate_pct"] is not None
    r2 = rp.run_replay(U, maker=False, chop=True, overrides={"chop_er_min": 0.99})
    assert r2["ok"] and r2["chop_blocked_pct"] > 90 and r2["trades"] == 0
    rep = rp.run_report(U)
    ab = rep["chop_ab"]
    assert set(ab) == {"off", "on", "helps_in_both_halves"}
    assert ab["off"]["full"] == rep["full"]["return_pct"]     # base run is filter-off


# ------------------------------------------------------------ long history fetch
def test_long_fetch_paces_and_retries_429(monkeypatch):
    from app.backtest import engine as E
    calls, sleeps = [], []

    class R:
        def __init__(self, code, data): self.status_code, self._d = code, data
        def json(self): return self._d
        def raise_for_status(self):
            if self.status_code >= 400: raise RuntimeError(self.status_code)

    class C:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None, timeout=None):
            calls.append(params)
            if len(calls) == 2:
                return R(429, [])
            end = params.get("end", 1_800_000_000)
            return R(200, [[end - 3600 * k, 1, 2, 1, 1.5, 1] for k in range(1, 301)])

    async def fake_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr(E.httpx, "AsyncClient", lambda **k: C())
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    out = asyncio.run(E._fetch_history_raw("BTC-USD", 3600, chunks=5))
    assert len(out) == 1500                                   # 5 pages, deduped
    assert 1.0 in sleeps                                      # backed off on 429
    assert sleeps.count(0.15) >= 3                            # paced between pages


# ------------------------------------------------------------ waste
def test_prune_signals_keeps_unscored_and_recent():
    from app import db
    now = time.time()
    with db._lock, db._conn() as c:
        c.execute("DELETE FROM signal_scores")
        rows = [(now - 90 * 86400, "trend_slow", "X", 1, 0.5, 0.0, 1, None),
                (now - 90 * 86400, "trend_slow", "X", 1, 0.5, 0.0, 0, None),
                (now - 1 * 86400, "trend_slow", "X", 1, 0.5, 0.0, 1, None)]
        c.executemany("INSERT INTO signal_scores(ts,strategy,product,direction,confidence,"
                      "fwd_return,scored,regime) VALUES(?,?,?,?,?,?,?,?)", rows)
    db.prune_signals()
    with db._lock, db._conn() as c:
        left = c.execute("SELECT scored, ts FROM signal_scores").fetchall()
    assert len(left) == 2 and all(r[0] == 0 or now - r[1] < 2 * 86400 for r in left)


def test_component_audit_flags_losers_and_unproven(monkeypatch, tmp_path):
    import app.orchestrator as O
    import app.export as export
    from app.learn.loop import learner
    monkeypatch.setattr(export, "REPORTS_DIR", str(tmp_path))
    saved = dict(learner.bandit.arms)
    try:
        learner.bandit.arms = {}
        for _ in range(80):
            learner.bandit.update("bull", "trend_slow", 0.002)
            learner.bandit.update("bull", "xsmom", -0.002)
        o = O.Orchestrator()
        a = o.component_audit()
        assert "trend_slow" in a["earning"]
        assert "xsmom" in a["candidates_to_switch_off"]
        assert "breakout_slow" in a["not_enough_evidence"]
        assert "meanrev" not in a["sleeves"]                  # disabled sleeves skipped
        sent = []
        monkeypatch.setattr("app.alerts.alert", lambda *x, **k: sent.append(x))
        assert o._maybe_component_audit() is None             # first call starts the clock
        assert not sent
        with open(tmp_path / "component_audit_latest.json", "w") as f:
            json.dump({"ts": time.time() - 8 * 86400}, f)
        assert o._maybe_component_audit() is not None and sent
    finally:
        learner.bandit.arms = saved


# ------------------------------------------------------------ forward test
def test_forward_test_baseline_and_comparison(monkeypatch, tmp_path):
    import app.orchestrator as O
    import app.export as export
    monkeypatch.setattr(export, "REPORTS_DIR", str(tmp_path))
    m = _Mkt({"BTC-USD": 100.0, "ETH-USD": 50.0, "SOL-USD": 10.0})
    monkeypatch.setattr(O, "market", m)
    monkeypatch.setattr("app.config.PRODUCTS", ["BTC-USD", "ETH-USD", "SOL-USD"])
    o = O.Orchestrator()
    eq = {"v": 100_000.0}
    monkeypatch.setattr(o, "_total_equity", lambda mk: eq["v"])
    o._ensure_forward_baseline()
    assert (tmp_path / "forward_test_baseline.json").exists()
    m._px = {"BTC-USD": 110.0, "ETH-USD": 55.0, "SOL-USD": 11.0}
    eq["v"] = 103_000.0
    ft = o.forward_test()
    assert ft["return_pct"] == 3.0 and ft["buy_hold_pct"] == 10.0
    o._ensure_forward_baseline()                              # never overwritten
    assert json.load(open(tmp_path / "forward_test_baseline.json"))["equity"] == 100_000.0


# ------------------------------------------------------------ core holding
def _daily(above=True, n=80):
    now = time.time()
    t0 = int(now // 86400 - n - 1) * 86400
    px = [100 + (i if above else -i) * 0.5 for i in range(n)]
    return [[t0 + i * 86400, p, p, p, p, 1.0] for i, p in enumerate(px)]


def _core_settings(monkeypatch, pct=50, assets="BTC-USD,ETH-USD", filt=True):
    from app import settings
    real = settings.get
    vals = {"core_allocation_pct": pct, "core_assets": assets, "core_trend_filter": filt,
            "core_selection": "trend", "core_sizing": "equal",     # the original rule
            "core_sma_days": 50, "core_hysteresis": 0.0, "core_strategy": "settings"}
    monkeypatch.setattr(settings, "get", lambda k: vals[k] if k in vals else real(k))


def test_core_off_by_default():
    from app.strategies.core import CoreBook
    assert not CoreBook().enabled()


def test_core_trend_filter_and_rebalance(monkeypatch):
    from app.strategies.core import CoreBook
    from app.execution.paper import PaperBroker
    _core_settings(monkeypatch)
    b = PaperBroker(); b.cash = 100_000.0
    c = CoreBook()
    m = _Mkt({"BTC-USD": 100.0, "ETH-USD": 50.0})
    acts = c.rebalance(b, m, 100_000.0, {"BTC-USD": _daily(True), "ETH-USD": _daily(False)})
    assert any("BTC-USD" in a for a in acts)
    assert "BTC-USD" in c.positions and "ETH-USD" not in c.positions   # ETH below its 50d avg
    assert abs(c.value(m) - 25_000.0 * (100.0 / (100.0 * (1 + 10 / 1e4)))) < 1.0
    assert b.cash < 100_000.0 - 25_000.0                               # cash really left
    # trend breaks -> exit immediately, even within the week
    acts = c.rebalance(b, m, 100_000.0, {"BTC-USD": _daily(False), "ETH-USD": _daily(False)})
    assert "BTC-USD" not in c.positions and any("sold" in a for a in acts)
    # round trip lost only fees + slippage
    assert 100_000.0 - b.cash < 25_000.0 * 0.015


def test_core_is_isolated_from_bot_and_persists(monkeypatch):
    import app.orchestrator as O
    from app.strategies.core import core, CoreBook
    from app.execution.paper import broker
    m = _Mkt({"BTC-USD": 200.0})
    monkeypatch.setattr(O, "market", m)
    saved = core.to_dict()
    try:
        core.positions = {"BTC-USD": {"qty": 10.0, "entry": 150.0, "opened": 0}}
        o = O.Orchestrator()
        total = o._total_equity(m)
        assert abs(total - o._bot_equity(m) - 2_000.0) < 1e-6
        assert "BTC-USD" not in broker.positions                        # separate book
        c2 = CoreBook(); c2.load_dict(json.loads(json.dumps(core.to_dict())))
        assert c2.positions["BTC-USD"]["qty"] == 10.0
    finally:
        core.load_dict(saved); core.positions = dict(saved["positions"])
