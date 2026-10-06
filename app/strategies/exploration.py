"""Exploration sleeve — fast strategies trading live paper side by side.

A share of the account (`exploration_allocation_pct`) is split between the
MEMBERS below. Each runs on its own book (positions + its own cash inside the
shared pool) and is measured like a fund: a NAV per unit, so moving money in
or out never distorts its return.

Every day:
  * BENCH: a member whose equity falls to (1 - `exploration_bench_drawdown`)
    of the capital it was given sells out and stops trading — still tracked
    (engine members by simulation on the stored prices since the bench day);
  * REINSTATE: after `exploration_reinstate_days` on the bench, a member comes
    back if its tracked return since benching is positive (the hourly bot: if
    its daily replay is positive in both halves);
  * REALLOCATE: the sleeve's money is split across active members by their
    last-30-day NAV return — weight exp(5 x r30), kept within 0.5x..2x of an
    equal share — so winners get more and losers less, on LIVE results.

Engine members trade the SAME strategy functions the backtests run
(app/engine/strategies.py) on bars from the data store; the hourly bot is the
existing hourly signal engine, sized against its slice. Trades pay the taker
fee + slippage, like the core.
"""
from __future__ import annotations
import json
import math
import os
import time
import uuid

from ..tunables import tv
from .. import db

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(ROOT, "reports", "exploration_state.json")
MAJORS = ["BTC-USD", "ETH-USD"]
HOUR, DAY = 3600, 86400
MIN_TRADE = 25.0
DRIFT_BAND = 0.10                  # fast strategies: re-balance at 10% off target
STALE_BARS = 3                     # older data than this many bars -> no trading
HISTORY_KEEP = 24 * 120            # hourly NAV points kept (~120 days)

MEMBERS = {
    "hourly_bot": {"kind": "hourly_bot"},
    "majors_trend_hourly_20d": {"kind": "engine", "bar_sec": HOUR, "assets": MAJORS,
                                "selection": "trend", "sizing": "equal", "sma": 480,
                                "hysteresis": 0.01},
    "majors_trend_hourly_5d": {"kind": "engine", "bar_sec": HOUR, "assets": MAJORS,
                               "selection": "trend", "sizing": "equal", "sma": 120,
                               "hysteresis": 0.005},
    "majors_trend_daily_30d": {"kind": "engine", "bar_sec": DAY, "assets": MAJORS,
                               "selection": "trend", "sizing": "equal", "sma": 30,
                               "hysteresis": 0.01},
    "alt_momentum_daily": {"kind": "engine", "bar_sec": DAY, "universe_top": 20,
                           "selection": "momentum", "sizing": "equal", "sma": 50,
                           "top_k": 5, "rebalance_days": 1},
}


def _setting(key, default):
    from .. import settings
    try:
        v = settings.get(key)
        return default if v is None else v
    except Exception:
        return default


class Exploration:
    def __init__(self, path=None):
        self.path = path or STATE
        self.members = {}              # name -> state dict
        self.last_review_day = None
        self.funded = False
        self.load()

    # ------------------------------------------------------------ persistence
    def load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        self.members = d.get("members", {})
        self.last_review_day = d.get("last_review_day")
        self.funded = bool(d.get("funded"))

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + ".tmp", "w") as f:
            json.dump({"members": self.members, "last_review_day": self.last_review_day,
                       "funded": self.funded}, f)
        os.replace(self.path + ".tmp", self.path)

    # ------------------------------------------------------------ state
    @staticmethod
    def enabled():
        from .allocator import exploration_pct
        return exploration_pct() > 0

    def _m(self, name):
        return self.members.setdefault(name, {
            "status": "active", "capital": 0.0, "cash": 0.0, "positions": {},
            "units": 0.0, "nav": 1.0, "history": [], "benched_at": None,
            "bot_base": None, "trades": 0, "last_bar": None})

    def is_active(self, name):
        return self.enabled() and self._m(name)["status"] == "active"

    # ------------------------------------------------------------ valuation
    @staticmethod
    def _bot_pnl(market, broker):
        """Hourly bot's realized + unrealized P&L (its own broker book)."""
        unreal = 0.0
        for p, pos in broker.positions.items():
            px = market.price(p) or pos["entry"]
            unreal += pos["qty"] * (px - pos["entry"]) * (1 if pos.get("side", 1) > 0 else -1)
        return broker.realized_pnl + unreal

    def member_equity(self, name, market, broker=None):
        m = self._m(name)
        if MEMBERS[name]["kind"] == "hourly_bot":
            if broker is None:
                from ..execution.paper import broker as broker
            base = m["bot_base"] if m["bot_base"] is not None else self._bot_pnl(market, broker)
            return m["capital"] + self._bot_pnl(market, broker) - base
        v = m["cash"]
        for a, pos in m["positions"].items():
            v += pos["qty"] * (market.price(a) or pos["entry"])
        return v

    def value(self, market):
        """Marked value of every exploration-book POSITION (their cash already
        sits in the shared pool) — added to account equity like the core."""
        v = 0.0
        for name, m in self.members.items():
            if MEMBERS.get(name, {}).get("kind") == "engine":
                for a, pos in m["positions"].items():
                    v += pos["qty"] * (market.price(a) or pos["entry"])
        return v

    def bot_equity(self, market, broker=None):
        """What the hourly bot sizes against: its slice, or 0 when benched."""
        if not self.is_active("hourly_bot"):
            return 0.0
        return max(0.0, self.member_equity("hourly_bot", market, broker))

    def _mark(self, name, market, now, broker=None):
        m = self._m(name)
        eq = self.member_equity(name, market, broker)
        if m["units"] > 0:
            m["nav"] = eq / m["units"]
        m["history"].append([now, round(m["nav"], 6)])
        del m["history"][:-HISTORY_KEEP]
        return eq

    def r30(self, name, now):
        """Last-30-day NAV return (0 with under 7 days of history)."""
        h = self._m(name)["history"]
        if not h or now - h[0][0] < 7 * DAY:
            return 0.0
        past = [nav for ts, nav in h if ts <= now - 30 * DAY] or [h[0][1]]
        return h[-1][1] / past[-1] - 1 if past[-1] > 0 else 0.0

    # ------------------------------------------------------------ money flows
    def _transfer(self, name, amount, market, broker=None):
        """Give (+) or take (-) capital; units move at the current NAV."""
        m = self._m(name)
        eq = self.member_equity(name, market, broker)
        nav = eq / m["units"] if m["units"] > 0 and eq > 0 else 1.0
        m["units"] = max(0.0, m["units"] + amount / nav)
        m["capital"] += amount
        if MEMBERS[name]["kind"] == "engine":
            m["cash"] += amount
        m["nav"] = nav

    def fund(self, account_equity, market, broker=None, now=None):
        """First funding: equal slices for every member."""
        if self.funded or not self.enabled():
            return False
        from .allocator import exploration_pct
        now = now or time.time()
        share = account_equity * exploration_pct() / len(MEMBERS)
        for name in MEMBERS:
            m = self._m(name)
            if MEMBERS[name]["kind"] == "hourly_bot":
                if broker is None:
                    from ..execution.paper import broker as broker
                m["bot_base"] = self._bot_pnl(market, broker)
            self._transfer(name, share, market, broker)
            m["started_at"] = now
            self._mark(name, market, now, broker)
        self.funded = True
        self.save()
        return True

    # ------------------------------------------------------------ trading
    def _fill(self, px, side):
        slip = px * tv("slippage_bps") / 1e4
        return px + slip if side == "buy" else px - slip

    def _buy(self, broker, name, a, notional, px):
        m = self._m(name)
        notional = min(notional, max(0.0, m["cash"]) / (1 + tv("fee_rate")),
                       broker.cash / (1 + tv("fee_rate")))
        if notional < MIN_TRADE:
            return False
        fill = self._fill(px, "buy")
        fee = notional * tv("fee_rate")
        if not broker._reserve(f"explore-buy-{uuid.uuid4().hex}", notional + fee,
                               f"explore {name} buy {a}"):
            return False
        qty = notional / fill
        pos = m["positions"].get(a)
        if pos:
            tot = pos["qty"] + qty
            pos["entry"] = (pos["entry"] * pos["qty"] + fill * qty) / tot
            pos["qty"] = tot
        else:
            m["positions"][a] = {"qty": qty, "entry": fill, "opened": time.time()}
        m["cash"] -= notional + fee
        m["trades"] += 1
        db.log_trade(a, "buy", qty, fill, fee, f"EXPLORE {name}")
        return True

    def _sell(self, broker, name, a, qty, px, reason):
        m = self._m(name)
        pos = m["positions"].get(a)
        if not pos or qty <= 0:
            return False
        qty = min(qty, pos["qty"])
        fill = self._fill(px, "sell")
        gross = qty * fill
        fee = gross * tv("fee_rate")
        pnl = gross - fee - qty * pos["entry"]
        broker._settle(f"explore-sell-{uuid.uuid4().hex}", gross - fee,
                       f"explore {name} sell {a}")
        m["cash"] += gross - fee
        pos["qty"] -= qty
        if pos["qty"] * fill < 1.0:
            m["positions"].pop(a, None)
        m["trades"] += 1
        db.log_trade(a, "sell", qty, fill, fee, f"EXPLORE {name}: {reason}", pnl)
        return True

    def target_weights(self, name, now=None):
        """Today's / this hour's weights for an engine member, from the store —
        the last CLOSED bar's row of its backtest function. None when the data
        is stale (never trade blind)."""
        from ..engine import panel as P, strategies as S, universe as U
        cfg = MEMBERS[name]
        bar = cfg["bar_sec"]
        now = now or time.time()
        if "assets" in cfg:
            # enough history for the average + its buffer state
            start = int(now - (cfg["sma"] * 4 + 50) * bar)
            p = P.from_store(cfg["assets"], start=start, bar_sec=bar)
            uni = None
        else:
            p = P.from_store(start=int(now - 400 * DAY), bar_sec=bar)
            uni = U.liquid_mask(p, cfg["universe_top"]) if p.T else None
        if not p.T or int(now // bar) - int(p.days[-1]) > STALE_BARS:
            return None, None
        W = S.trend_portfolio(p, cfg["selection"], cfg.get("sizing", "equal"),
                              sma=cfg["sma"], top_k=cfg.get("top_k", 5),
                              rebalance_days=cfg.get("rebalance_days", 7),
                              hysteresis=cfg.get("hysteresis", 0.0), universe=uni)
        return {c: float(w) for c, w in zip(p.coins, W[-1]) if w > 0}, int(p.days[-1])

    def step(self, name, broker, market, now=None):
        """Rebalance one active engine member to its latest closed bar (once
        per new bar). Returns a list of action strings."""
        m = self._m(name)
        if MEMBERS[name]["kind"] != "engine" or m["status"] != "active":
            return []
        from ..execution.paper import _valid_price
        w, bar = self.target_weights(name, now)
        if w is None or bar == m["last_bar"]:
            return []
        eq = self.member_equity(name, market, broker)
        acts = []
        for a in set(w) | set(m["positions"]):
            px = market.price(a)
            if not _valid_price(px):
                continue
            tgt = max(0.0, eq) * w.get(a, 0.0)
            pos = m["positions"].get(a)
            cur = pos["qty"] * px if pos else 0.0
            if tgt == 0 and pos:
                if self._sell(broker, name, a, pos["qty"], px, "signal exit"):
                    acts.append(f"{name}: sold {a}")
            elif cur == 0 or abs(cur - tgt) > DRIFT_BAND * tgt:
                if cur < tgt and self._buy(broker, name, a, tgt - cur, px):
                    acts.append(f"{name}: bought ${tgt - cur:,.0f} {a}")
                elif cur > tgt and pos and self._sell(broker, name, a, (cur - tgt) / px, px,
                                                      "rebalance trim"):
                    acts.append(f"{name}: trimmed ${cur - tgt:,.0f} {a}")
        m["last_bar"] = bar
        return acts

    # ------------------------------------------------------------ daily review
    def _shadow_return(self, name, since, now):
        """Simulated return of an engine member from `since` to now on stored
        prices (how a benched member WOULD have done)."""
        from ..engine import panel as P, strategies as S, backtest as B, universe as U
        cfg = MEMBERS[name]
        bar = cfg["bar_sec"]
        start = int(since - (cfg["sma"] * 4 + 50) * bar)
        p = P.from_store(cfg.get("assets"), start=start, bar_sec=bar)
        if not p.T:
            return None
        uni = None if "assets" in cfg else U.liquid_mask(p, cfg["universe_top"])
        W = S.trend_portfolio(p, cfg["selection"], cfg.get("sizing", "equal"),
                              sma=cfg["sma"], top_k=cfg.get("top_k", 5),
                              rebalance_days=cfg.get("rebalance_days", 7),
                              hysteresis=cfg.get("hysteresis", 0.0), universe=uni)
        s0 = int((p.days < since // bar).sum())
        r, _, _ = B.simulate_drift(W, p.returns(), tv("fee_rate") + tv("slippage_bps") / 1e4,
                                   start=s0, band_rel=DRIFT_BAND)
        return float(math.prod(1 + x for x in r) - 1) if len(r) else 0.0

    def review(self, account_equity, market, broker, now=None, replay_verdict=None):
        """Once a day: bench, reinstate, reallocate. Returns event strings."""
        from .allocator import exploration_pct
        now = now or time.time()
        day = int(now // DAY)
        if self.last_review_day == day or not self.funded:
            return []
        events = []
        dd = float(_setting("exploration_bench_drawdown", 0.20))
        back_days = float(_setting("exploration_reinstate_days", 30))
        for name in MEMBERS:
            m = self._m(name)
            eq = self.member_equity(name, market, broker)
            if m["status"] == "active" and m["capital"] > 0 and eq <= (1 - dd) * m["capital"]:
                for a, pos in list(m["positions"].items()):
                    px = market.price(a)
                    if px:
                        self._sell(broker, name, a, pos["qty"], px, "benched")
                m["status"], m["benched_at"] = "benched", now
                events.append(f"BENCHED {name}: equity ${eq:,.0f} is {eq / m['capital'] - 1:+.1%} "
                              f"of the ${m['capital']:,.0f} it was given")
            elif m["status"] == "benched" and now - (m["benched_at"] or now) >= back_days * DAY:
                if MEMBERS[name]["kind"] == "hourly_bot":
                    ok = replay_verdict == "positive in both halves"
                    why = f"daily replay: {replay_verdict}"
                else:
                    sr = self._shadow_return(name, m["benched_at"], now)
                    ok = sr is not None and sr > 0
                    why = f"tracked return since benching {sr:+.1%}" if sr is not None else "no data"
                if ok:
                    m["status"], m["benched_at"] = "active", None
                    events.append(f"REINSTATED {name} ({why})")
        # withdraw benched members' money; split the sleeve over active ones
        active = [n for n in MEMBERS if self._m(n)["status"] == "active"]
        for name in MEMBERS:
            m = self._m(name)
            if m["status"] == "benched" and not m["positions"]:
                eq = self.member_equity(name, market, broker)
                if MEMBERS[name]["kind"] == "engine" or not broker.positions:
                    self._transfer(name, -eq, market, broker)
        total = account_equity * exploration_pct()
        if active:
            raw = {n: min(2.0, max(0.5, math.exp(5 * self.r30(n, now)))) for n in active}
            z = sum(raw.values())
            for n in active:
                target = total * raw[n] / z
                self._transfer(n, target - self.member_equity(n, market, broker), market, broker)
        for name in MEMBERS:
            self._mark(name, market, now, broker)
        self.last_review_day = day
        self.save()
        return events

    # ------------------------------------------------------------ reporting
    def snapshot(self, market, broker=None):
        out = {}
        if broker is None:
            from ..execution.paper import broker as broker
        for name, cfg in MEMBERS.items():
            m = self._m(name)
            eq = self.member_equity(name, market, broker)
            if cfg["kind"] == "hourly_bot":
                # it trades on the broker's own book: count fills since the reset
                m["trades"] = 2 * len(broker.closed_trades) + len(broker.positions)
            out[name] = {"status": m["status"], "capital": round(m["capital"], 2),
                         "equity": round(eq, 2), "nav_return_pct": round((m["nav"] - 1) * 100, 2),
                         "r30_pct": round(self.r30(name, time.time()) * 100, 2),
                         "trades": m["trades"], "benched_at": m["benched_at"],
                         "positions": {a: {"qty": p["qty"], "entry": p["entry"],
                                           "price": market.price(a)}
                                       for a, p in m["positions"].items()},
                         "config": cfg}
        return out


manager = Exploration()
