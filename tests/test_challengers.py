"""Champion / challenger loop: no promotion without forward evidence, a
changed config restarts its clock, and a challenger that is better in the
backtest AND forward (with enough days) gets promoted."""
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from app.engine import panel as P, challengers as C


def _panel(n=1400):
    t0 = int(time.time() // 86400 - n - 2) * 86400
    out = {}
    for i, p in enumerate(("BTC-USD", "ETH-USD")):
        r = random.Random(i)
        px, rows = 100.0, []
        for k in range(n):
            # strong trends with crashes: trend-following clearly wins
            drift = 0.004 if (k // 200) % 2 == 0 else -0.004
            px *= math.exp(drift + r.gauss(0, 0.02))
            rows.append([t0 + k * 86400, px * 0.98, px * 1.02, px, px, 1e6])
        out[p] = rows
    return P.from_candles(out)


@pytest.fixture()
def loop(tmp_path, monkeypatch):
    from app.engine import registry as R
    monkeypatch.setattr(R, "DIR", str(tmp_path / "exp"))
    monkeypatch.setattr(C, "STATE", str(tmp_path / "challengers.json"))
    monkeypatch.setattr(C, "CHAMPION", str(tmp_path / "champion.json"))
    monkeypatch.setattr(C, "CANDIDATES", {
        "hold": {"assets": ["BTC-USD", "ETH-USD"], "selection": "hold"},
        "trend": {"assets": ["BTC-USD", "ETH-USD"], "selection": "trend", "sizing": "equal",
                  "sma": 50, "hysteresis": 0.0}})
    monkeypatch.setattr(C, "DEFAULT_CHAMPION", "hold")
    return C


def _date(panel, i):
    import datetime as dt
    return (dt.date(1970, 1, 1) + dt.timedelta(days=int(panel.days[i]))).isoformat()


def test_no_promotion_without_forward_evidence(loop):
    p = _panel()
    rep = loop.run(panel=p, backtest_from=_date(p, 100), min_dsr=0.0)
    assert rep["champion"] == "hold" and "promoted" not in rep
    assert rep["candidates"]["trend"]["forward_days"] == 0
    assert rep["candidates"]["trend"]["beats_champion_backtest_both_halves"]


def test_promotes_after_enough_better_forward_days(loop):
    p = _panel()
    cut = 1000
    early = P.Panel(p.days[:cut], p.coins, p.close[:cut], p.volume[:cut])
    loop.run(panel=early, backtest_from=_date(p, 100), min_dsr=0.0)    # register configs
    rep = loop.run(panel=p, backtest_from=_date(p, 100), min_dsr=0.0, min_forward_days=90)
    assert rep["candidates"]["trend"]["forward_days"] >= 90
    assert rep.get("promoted") == "trend"
    assert loop.champion()[0] == "trend"


def test_changed_config_restarts_the_forward_clock(loop, monkeypatch):
    p = _panel()
    cut = 1000
    early = P.Panel(p.days[:cut], p.coins, p.close[:cut], p.volume[:cut])
    loop.run(panel=early, backtest_from=_date(p, 100), min_dsr=0.0)
    changed = dict(loop.CANDIDATES)
    changed["trend"] = dict(changed["trend"], sma=60)
    monkeypatch.setattr(loop, "CANDIDATES", changed)
    rep = loop.run(panel=p, backtest_from=_date(p, 100), min_dsr=0.0, min_forward_days=90)
    assert rep["candidates"]["trend"]["forward_days"] == 0
    assert "promoted" not in rep


def test_risk_models_never_look_ahead():
    from app.engine import risk_models as M
    p = _panel(1000)
    p = P.Panel(p.days, p.coins, p.close, p.volume, "", p.close * 1.01, p.close * 0.99)
    vf = M.har_vol_forecast(p, min_train=300)
    cut = 600
    q = P.Panel(p.days, p.coins, p.close.copy(), p.volume, "", p.high.copy(), p.low.copy())
    q.high[cut + 1:] *= 3.0                       # wild future ranges
    vq = M.har_vol_forecast(q, min_train=300)
    assert np.allclose(vf[:cut + 1], vq[:cut + 1], equal_nan=True)
    X = M.regime_features(p)
    fwd = M.forward_return(p.close[:, 0], 30)
    pr = M.regime_probability(X, fwd, min_train=400, refit_days=60)
    fwd2 = fwd.copy()
    fwd2[cut - 30:] = -fwd2[cut - 30:]            # flip labels that end after `cut`
    pr2 = M.regime_probability(X, fwd2, min_train=400, refit_days=60)
    assert np.allclose(pr[:cut + 1], pr2[:cut + 1], equal_nan=True)
