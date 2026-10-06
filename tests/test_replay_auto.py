"""Automatic universe replay: the replay engine, offline signal mode, new-coin
detection and the orchestrator's run/record/alert path."""
import asyncio
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _candles(n=400, seed=0, drift=0.0004, t0=1_700_000_000):
    r = random.Random(seed)
    px, out = 100.0, []
    for i in range(n):
        op = px
        px *= math.exp(drift + r.gauss(0, 0.012))
        hi, lo = max(op, px) * 1.004, min(op, px) * 0.996
        out.append([t0 + i * 3600, lo, hi, op, px, 1000.0])
    return out


def _universe(k=7, n=400):
    names = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
             "DOGE-USD", "XRP-USD", "NEAR-USD"][:k]
    return {p: _candles(n, seed=i, drift=0.0006 * (i - k / 2) / k) for i, p in enumerate(names)}


def test_run_replay_covers_every_coin():
    from app.backtest import replay as rp
    U = _universe()
    rep = rp.run_report(U)
    assert rep["full"]["ok"], rep["full"]
    for k in ("return_pct", "max_drawdown_pct", "trades", "buy_hold_equal_weight_pct"):
        assert k in rep["full"]
    assert set(rep["per_product"]) == set(U)          # every coin gets a row
    assert rep["verdict"] in ("positive in both halves",
                              "positive overall, but not in both halves", "negative")
    assert "_trades" not in rep["full"]


def test_replay_too_little_data_is_reported_not_raised():
    from app.backtest import replay as rp
    r = rp.run_replay({"BTC-USD": _candles(50)})
    assert r["ok"] is False and "error" in r


def test_offline_compute_does_not_touch_live_state(monkeypatch):
    from app.backtest import replay as rp
    from app.learn.direction import direction_learner
    from app.signals import engine as eng
    recorded = []
    monkeypatch.setattr(eng.db, "record_signal", lambda *a, **k: recorded.append(a))
    before = (direction_learner.n_vetoes, direction_learner.n_flips)
    rp.run_replay(_universe(6, 250))
    assert recorded == []
    assert (direction_learner.n_vetoes, direction_learner.n_flips) == before


def test_replay_restores_engine_products():
    from app.backtest import replay as rp
    from app.signals import engine as eng
    saved = list(eng.PRODUCTS)
    rp.run_replay(_universe(6, 250))
    assert list(eng.PRODUCTS) == saved


def test_replay_due_detects_new_coins_and_schedule():
    from app.orchestrator import Orchestrator
    o = Orchestrator()
    o.last_replay = None
    due, new = o._replay_due(["BTC-USD", "ETH-USD"])
    assert due and new == []                          # first run: due, not "new"
    o.last_replay = {"ran_at": time.time(), "universe": ["BTC-USD", "ETH-USD"]}
    assert o._replay_due(["BTC-USD", "ETH-USD"]) == (False, [])
    due, new = o._replay_due(["BTC-USD", "ETH-USD", "SUI-USD"])
    assert not due and new == ["SUI-USD"]             # joiner triggers a run
    due, _ = o._replay_due(["BTC-USD"], now=time.time() + 2 * 86400)
    assert due                                        # schedule elapsed


def test_run_replay_now_fetches_whole_universe_and_alerts(monkeypatch, tmp_path):
    import app.export as export
    from app import config, alerts
    from app.orchestrator import Orchestrator
    U = _universe(7)
    monkeypatch.setattr(export, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PRODUCTS", list(U) + ["NEWCOIN-USD"])
    fetched = []

    async def fake_fetch(p, granularity=3600, chunks=3, **k):
        fetched.append(p)
        return U.get(p)                              # NEWCOIN has no history
    monkeypatch.setattr("app.backtest.engine.fetch_history", fake_fetch)
    sent = []
    monkeypatch.setattr(alerts, "alert", lambda lvl, title, msg="": sent.append((lvl, title, msg)))
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    o = Orchestrator()
    rep = asyncio.run(o.run_replay_now(reason="test", new=["NEWCOIN-USD"]))
    assert sorted(fetched) == sorted(config.PRODUCTS)  # every universe coin fetched
    assert rep["universe"] == sorted(config.PRODUCTS)
    assert rep["missing_history"] == ["NEWCOIN-USD"]
    assert o.last_replay is rep
    assert (tmp_path / "replay_latest.json").exists()
    assert (tmp_path / "replay_history.jsonl").read_text().count("\n") == 1
    assert any("NEW NEWCOIN-USD" in m for _, t, m in sent if t.startswith("Strategy replay"))
    snap = o.replay_snapshot()
    assert snap["universe_size"] == len(config.PRODUCTS)
    # a restart reloads the last report from disk
    o2 = Orchestrator()
    o2._load_last_replay()
    assert o2.last_replay["universe"] == rep["universe"]


_real_sleep = asyncio.sleep


async def _no_sleep(*a, **k):
    await _real_sleep(0)


# ---------------------------------------------------------------- risk brake
def _rep(h1, h2, trades=100):
    return {"full": {"ok": True, "return_pct": h1 + h2, "trades": trades},
            "first_half": {"ok": True, "return_pct": h1},
            "second_half": {"ok": True, "return_pct": h2}}


def test_brake_only_when_both_halves_negative():
    from app.orchestrator import Orchestrator
    from app import tunables
    f = Orchestrator._replay_brake_for
    m = tunables.tv("replay_brake_mult")
    assert f(_rep(-6.8, -0.8))[0] == m and m < 1.0
    assert f(_rep(-6.8, +4.0))[0] == 1.0          # one bad half = regime luck
    assert f(_rep(+1.0, +2.0))[0] == 1.0
    assert f(_rep(-6.8, -0.8, trades=5))[0] == 1.0  # too few trades to judge
    assert f(None)[0] == 1.0 and f({"full": {"ok": False}})[0] == 1.0


def test_brake_respects_setting(monkeypatch):
    from app.orchestrator import Orchestrator
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get",
                        lambda k: False if k == "replay_brake_enabled" else real(k))
    assert Orchestrator._replay_brake_for(_rep(-5, -5))[0] == 1.0


def test_brake_scales_position_size_and_alerts(monkeypatch):
    from app.orchestrator import Orchestrator
    from app.risk.manager import risk
    from app import alerts, tunables
    sent = []
    monkeypatch.setattr(alerts, "alert", lambda lvl, t, m="": sent.append((lvl, t)))
    status = {"effective_risk_scale": 1.0}
    o = Orchestrator()
    o._apply_replay_brake(_rep(+1, +1))
    n_free, _, _ = risk.size(100_000, 100.0, 3.0, 0.8, status, product="BTC-USD")
    o._apply_replay_brake(_rep(-3, -1))
    n_brake, _, _ = risk.size(100_000, 100.0, 3.0, 0.8, status, product="BTC-USD")
    assert n_free > 0
    assert abs(n_brake - n_free * tunables.tv("replay_brake_mult")) < 1e-6 * n_free
    assert sent[-1] == ("warning", "Replay brake ON")
    o._apply_replay_brake(_rep(-3, +2))
    assert risk.replay_brake == 1.0 and sent[-1] == ("info", "Replay brake OFF")
    n = len(sent)
    o._apply_replay_brake(_rep(+1, +1))           # no state change -> no alert
    assert len(sent) == n
