"""Per-sleeve risk: the kill switch ignores the core's own P&L, the core has
its tracking monitor, Polymarket has an evidence gate (own bankroll), the
exploration sleeve reallocates on shrunk evidence and stops buying while
the account is halted."""
import math
import time

import pytest


class _Mkt:
    def __init__(self, px):
        self._px = dict(px)
        self.candles = {}
        self.tickers = {p: {"price": v} for p, v in px.items()}
        self.healthy = True

    def price(self, p):
        return self._px.get(p)


def _settings(monkeypatch, **vals):
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda k, *a: vals[k] if k in vals else real(k))


def _daily(above=True, n=80):
    now = time.time()
    t0 = int(now // 86400 - n - 1) * 86400
    px = [100 + (i if above else -i) * 0.5 for i in range(n)]
    return [[t0 + i * 86400, p, p, p, p, 1.0] for i, p in enumerate(px)]


# ------------------------------------------------------------------ kill switch basis
def test_rebase_keeps_a_basis_change_from_tripping_the_kill():
    from app.risk.manager import RiskManager
    r = RiskManager()
    r.update(120_000.0, {})            # old basis: account incl. a profitable core
    r.rebase(100_000.0, "active_ex_core")
    r.update(100_000.0, {})
    assert not r.killed and r.dd_basis == "active_ex_core"
    r.update(84_000.0, {})             # a real 16% loss in the active sleeves
    assert r.killed


def test_core_pnl_counts_realized_unrealized_and_buy_fees(monkeypatch):
    from app.strategies.core import CoreBook
    from app.execution.paper import PaperBroker
    _settings(monkeypatch, core_allocation_pct=50, core_assets="BTC-USD", core_trend_filter=False,
              core_strategy="settings")
    b = PaperBroker()
    b.cash = 100_000.0
    c = CoreBook()
    m = _Mkt({"BTC-USD": 100.0})
    c.rebalance(b, m, 100_000.0, {})
    cash_after_buy = b.cash
    qty, entry = c.positions["BTC-USD"]["qty"], c.positions["BTC-USD"]["entry"]
    assert entry > 100.0                                  # the fill paid slippage
    assert c.fees_paid > 0 and abs(c.pnl(m) - (qty * (100.0 - entry) - c.fees_paid)) < 1e-6
    m._px["BTC-USD"] = 110.0
    qty, entry = c.positions["BTC-USD"]["qty"], c.positions["BTC-USD"]["entry"]
    assert abs(c.pnl(m) - (qty * (110 - entry) - c.fees_paid)) < 1e-6
    # account equity minus core P&L doesn't move when the core's coin moves
    eq = lambda: b.cash + c.value(m)
    m._px["BTC-USD"] = 80.0
    a = eq() - c.pnl(m)
    m._px["BTC-USD"] = 130.0
    assert abs((eq() - c.pnl(m)) - a) < 1e-6
    assert cash_after_buy < 100_000.0


def test_core_tracking_monitor_alerts_once_when_off_target(monkeypatch):
    from app.strategies.core import CoreBook, DAILY
    _settings(monkeypatch, core_allocation_pct=100, core_assets="BTC-USD",
              core_trend_filter=False, core_strategy="settings",
              core_tracking_alert_days=2, core_tracking_alert_te=10.0)
    c = CoreBook()
    m = _Mkt({"BTC-USD": 100.0})
    now = time.time()
    # on target: holds what it should
    c.positions = {"BTC-USD": {"qty": 1000.0, "entry": 100.0, "opened": 0}}
    c._record_tracking(m, 100_000.0, {"BTC-USD": 100_000.0}, now)
    assert c.tracking_alert() is None
    # two days holding nothing while the target is fully invested
    c.positions = {}
    c._record_tracking(m, 100_000.0, {"BTC-USD": 100_000.0}, now + DAILY)
    assert c.tracking_alert() is None
    c._record_tracking(m, 100_000.0, {"BTC-USD": 100_000.0}, now + 2 * DAILY)
    msg = c.tracking_alert()
    assert msg and "BTC-USD" in msg
    assert c.tracking_alert() is None                    # once per episode
    rep = c.tracking_report()
    assert rep["off_target_days"] == 2


def test_tracking_error_measures_return_gap():
    from app.strategies.core import CoreBook
    c = CoreBook()
    for d in range(12):
        px = 100 * (1.02 if d % 2 else 1.0)
        c.tracking.append({"day": d, "actual": {"BTC-USD": 0.0}, "target": {"BTC-USD": 1.0},
                           "px": {"BTC-USD": px}, "off_target": ["BTC-USD"]})
    rep = c.tracking_report()
    assert rep["days"] == 11 and rep["tracking_error_annual"] > 0.1


# ------------------------------------------------------------------ Polymarket
def _rows(n, bot_better=True, per_market=2, seed=0):
    import random
    r = random.Random(seed)
    rows = []
    for i in range(n):
        y = r.random() < 0.5
        mkt = 0.5
        bot = (0.65 if y else 0.35) if bot_better else (0.35 if y else 0.65)
        for k in range(per_market):
            rows.append({"condition_id": f"c{i}", "predicted_p0": bot, "market_p0": mkt,
                         "resolved_outcome0": int(y)})
    return rows


def test_pm_skill_gate():
    from app.markets.polymarket.skill_gate import evaluate
    assert evaluate(_rows(150), min_markets=100)["open"]
    assert not evaluate(_rows(150, bot_better=False), min_markets=100)["open"]
    few = evaluate(_rows(40), min_markets=100)
    assert not few["open"] and "need 100" in few["why"]
    # buckets of one market are one piece of evidence
    assert evaluate(_rows(60, per_market=5), min_markets=100)["markets"] == 60


def test_polymarket_takes_no_share_of_the_main_account():
    # Polymarket runs its own bankroll (pm_start_cash; test_pm_standalone.py)
    from app.strategies import allocator as A
    assert not hasattr(A, "polymarket_pct") and not hasattr(A, "polymarket_budget")


# ------------------------------------------------------------------ exploration
def test_shrunk_reallocation_ignores_noise_but_follows_consistent_results(monkeypatch):
    from app.strategies.exploration import Exploration, DAY
    _settings(monkeypatch, exploration_shrink_tau=0.001)
    e = Exploration(path="/nonexistent/x.json")
    now = 1_800_000_000.0
    import random
    r = random.Random(1)
    noisy, steady = [], []
    a = b = 1.0
    for d in range(31):
        noisy.append([now - (30 - d) * DAY, a])
        steady.append([now - (30 - d) * DAY, b])
        a *= math.exp(0.004 + r.gauss(0, 0.04))          # +12% drift buried in 4%/day noise
        b *= math.exp(0.004 + r.gauss(0, 0.0005))         # the same drift, nearly noiseless
    e._m("noisy")["history"] = noisy
    e._m("steady")["history"] = steady
    rn, kn = e.r30_shrunk("noisy", now)
    rs, ks = e.r30_shrunk("steady", now)
    assert kn < 0.05 and abs(rn) < 0.02
    assert ks > 0.9 and rs > 0.08


def test_exploration_member_does_not_buy_while_account_halted(monkeypatch):
    from app.strategies import exploration as X
    from app.risk.manager import risk
    e = X.Exploration(path="/nonexistent/x.json")
    m = e._m("majors_trend_daily_30d")
    m.update(status="active", cash=10_000.0, capital=10_000.0, units=10_000.0)
    monkeypatch.setattr(e, "target_weights", lambda name, now=None: ({"BTC-USD": 1.0}, 123))
    monkeypatch.setattr(X.Exploration, "enabled", staticmethod(lambda: True))
    bought = []
    monkeypatch.setattr(e, "_buy", lambda *a, **k: bought.append(a) or True)
    monkeypatch.setattr(risk, "killed", False)
    monkeypatch.setattr(risk, "halted_today", True)
    assert e.step("majors_trend_daily_30d", object(), _Mkt({"BTC-USD": 100.0})) == []
    assert not bought
    monkeypatch.setattr(risk, "halted_today", False)
    m["last_bar"] = None
    e.step("majors_trend_daily_30d", object(), _Mkt({"BTC-USD": 100.0}))
    assert bought
