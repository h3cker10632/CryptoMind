"""Review fixes for the research loop and the promotion replay: config
validation, per-candidate failure isolation, champion.json authority,
costs/taxes for the post-decision champion, and the replay's trial count,
candidate set and early replay start."""
import importlib.util
import math
import os
import random
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.engine import challengers as C
from app.engine import panel as P  # noqa: E402

TREND = {"assets": ["BTC-USD", "ETH-USD"], "selection": "trend", "sizing": "equal",
         "sma": 50, "hysteresis": 0.0}


def _panel(n=1400):
    t0 = int(time.time() // 86400 - n - 2) * 86400
    out = {}
    for i, p in enumerate(("BTC-USD", "ETH-USD")):
        r = random.Random(i)
        px, rows = 100.0, []
        for k in range(n):
            drift = 0.004 if (k // 200) % 2 == 0 else -0.004
            px *= math.exp(drift + r.gauss(0, 0.02))
            rows.append([t0 + k * 86400, px * 0.98, px * 1.02, px, px, 1e6])
        out[p] = rows
    return P.from_candles(out)


def _date(panel, i):
    import datetime as dt
    return (dt.date(1970, 1, 1) + dt.timedelta(days=int(panel.days[i]))).isoformat()


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def loop(tmp_path, monkeypatch):
    from app.engine import registry as Rg
    monkeypatch.setattr(Rg, "DIR", str(tmp_path / "exp"))
    monkeypatch.setattr(C, "STATE", str(tmp_path / "challengers.json"))
    monkeypatch.setattr(C, "CHAMPION", str(tmp_path / "champion.json"))
    monkeypatch.setattr(C, "QUEUE", str(tmp_path / "queue.json"))
    monkeypatch.setattr(C, "CANDIDATES", {
        "hold": {"assets": ["BTC-USD", "ETH-USD"], "selection": "hold"}, "trend": TREND})
    monkeypatch.setattr(C, "DEFAULT_CHAMPION", "hold")
    return C


# validate-config-accepts-crashing-configs / ml-vol-forecast-sizing-ignored
@pytest.mark.parametrize("cfg", [
    {"universe_top": 10, "selection": "hold"},
    dict(TREND, sma=150.0),
    {"universe_top": 10, "selection": "momentum", "top_k": 1.0},
    {"universe_top": 10, "selection": "momentum", "tranches": 3.0},
    {"universe_top": 20, "selection": "ml_rank", "sizing": "vol_forecast", "vol_target": 0.2},
    {"assets": ["BTC-USD"], "selection": "trend_meta", "sizing": "vol_forecast"},
])
def test_validate_rejects_configs_that_would_crash_or_mislead(cfg):
    assert C.validate_config(cfg) is not None


def test_validate_still_accepts_float_and_int_numbers():
    assert C.validate_config(dict(TREND, hysteresis=0, vol_target=1, sizing="vol_target")) is None


def test_one_failing_challenger_does_not_stop_the_run(loop, monkeypatch):
    monkeypatch.setitem(C.CANDIDATES, "bad", dict(TREND, sma=77))
    real, failing = C.weights, ["bad"]

    def weights(cfg, panel, **k):
        if cfg.get("sma") == 77 or cfg["selection"] in failing:
            raise ValueError("boom")
        return real(cfg, panel, **k)
    monkeypatch.setattr(C, "weights", weights)
    p = _panel()
    rep = loop.run(panel=p, backtest_from=_date(p, 100), min_dsr=0.0)
    assert "boom" in rep["candidates"]["bad"]["error"]
    assert "backtest" in rep["candidates"]["trend"]
    _tool("research_loop").show(rep)                     # the CLI prints error rows too
    failing.append("hold")
    with pytest.raises(ValueError):                      # ...but a failing champion raises
        loop.run(panel=p, backtest_from=_date(p, 100), min_dsr=0.0)


# reregister-champion-name-bypasses-promotion
def test_requeued_champion_name_cannot_change_what_it_trades(loop):
    cfg = dict(TREND, sma=150)
    assert loop.register("q1", cfg)[0]
    loop._save(loop.CHAMPION, {"name": "q1", "config": cfg})
    ok, msg = loop.register("q1", dict(cfg, sma=10))
    assert not ok and "champion" in msg
    q = loop._load(loop.QUEUE, {})                        # e.g. a hand-edited queue file
    q["candidates"]["q1"]["config"] = dict(cfg, sma=10)
    loop._save(loop.QUEUE, q)
    assert loop.champion() == ("q1", cfg)
    p = _panel(600)
    rep = loop.run(panel=p, backtest_from=_date(p, 100), min_dsr=0.0)
    assert rep["candidates"]["q1"]["config"] == cfg


# costs-taxes-for-replaced-champion
def test_costs_taxes_describe_the_new_champion(loop, monkeypatch):
    from app.engine import costs_tax as CT
    seen = []
    monkeypatch.setattr(CT, "scenarios", lambda W, panel, **k: seen.append(W) or {"scenarios": {}})
    p = _panel()
    early = P.Panel(p.days[:1000], p.coins, p.close[:1000], p.volume[:1000])
    loop.run(panel=early, backtest_from=_date(p, 100), min_dsr=0.0)
    rep = loop.run(panel=p, backtest_from=_date(p, 100), min_dsr=0.0, min_forward_days=90)
    assert rep.get("promoted") == "trend"
    assert rep["costs_taxes"]["strategy"] == "trend"
    np.testing.assert_array_equal(seen[-1], C.weights(TREND, p, start=100))


# replay-deflates-with-fewer-trials-than-live / replay-crash-replay-from-at-backtest-start
def _replay(**kw):
    from app.engine import promotion_backtest as PB
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal([0.0003, 0.004], 0.03, size=(1800, 2)), axis=0))
    p = P.Panel(np.arange(17000, 18800), ["BTC-USD", "ETH-USD"], close,
                np.full_like(close, 1e9), "test", close * 1.02, close * 0.98)

    def wfn(cfg, panel, start=0):
        W = np.zeros((panel.T, panel.N))
        W[:, :] = cfg["w"]
        return W
    cands = {"base": {"w": [1.0, 0.0]}, "better": {"w": [0.0, 1.0]}}
    return PB.run(p, cands, "base", backtest_from="2016-07-01", eval_every=30, min_dsr=0.5,
                  weights_fn=wfn, **kw)


def test_replay_deflates_with_the_live_trial_count():
    assert _replay()["n_trials"] == 2 and _replay()["promotions"]
    rep = _replay(n_trials=10 ** 9, trial_sr_std=0.5)
    assert rep["n_trials"] == 10 ** 9 and not rep["promotions"]


def test_replay_from_the_backtest_start_does_not_crash():
    rep = _replay(replay_from="2016-07-01")
    assert rep["from"] >= "2017-07-01"


# replay-includes-candidates-before-they-existed
def test_replay_tool_uses_built_in_candidates_and_registry_trials(loop, monkeypatch, tmp_path):
    from app.engine import promotion_backtest as PB
    assert loop.register("queued1", dict(TREND, sma=150))[0]
    got = {}

    def fake_run(panel, cands, champ, **kw):
        got.update(cands=cands, **kw)
        return {k: None for k in ("from", "days", "n_trials", "verdict", "promotions")}
    monkeypatch.setattr(PB, "run", fake_run)
    monkeypatch.setattr(P, "from_store", lambda *a, **k: None)
    tool = _tool("promotion_backtest")
    monkeypatch.setattr(tool, "ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["promotion_backtest.py"])
    tool.main()
    assert set(got["cands"]) == {"hold", "trend"}
    assert got["n_trials"] >= 2 and "trial_sr_std" in got
    # with more variants logged than candidates, the replay deflates by the
    # REGISTRY's count and spread, exactly as the live loop does
    from app.engine import registry as Rg
    rng = np.random.default_rng(0)
    for k in range(5):
        Rg.log(C.FAMILY, f"variant{k}", {"sma": 10 + k}, "v", rng.normal(0.001, 0.02, 300), {})
    tool.main()
    assert got["n_trials"] == sum(Rg.n_trials(f, floor=0) for f in C.RESEARCH_FAMILIES) == 5
    assert got["trial_sr_std"] == Rg.trial_sharpe_std(C.FAMILY) and got["trial_sr_std"] > 0
