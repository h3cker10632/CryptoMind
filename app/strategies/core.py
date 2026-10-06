"""Optional CORE holding — a low-turnover position next to the trading bot.

OFF by default (setting `core_allocation_pct` = 0). When set to e.g. 50, half
of account equity is held in the `core_assets` (default BTC-USD, ETH-USD),
split equally, and the active bot trades the rest. With `core_trend_filter`
on, each asset is held only while its last daily close is above its 50-day
average and sits in cash otherwise — a slow, rarely-trading rule (a handful of
switches a year), so fees barely matter. `core_sma_days` sets the average
length; `core_assets` = "UNIVERSE" applies the rule to every tracked coin.

On the real Sep-2025..Oct-2026 hourly history (18 coins, 0.6%/side costs):
per-coin "hold while above the 100-day average" +75% (max DD -12%), BTC/ETH
with the 50-day filter +20% (DD -22%), vs the hourly trading bot's replay
-92%. (All-coin figures carry survivorship bias: today's universe.)

Isolation from the bot:
  * its own book (`core.positions`), so the bot can trade the same coin
    without the two colliding in broker.positions;
  * never touched by stops / flips / exit advisors / protections / learning;
  * counted in ACCOUNT equity (kill switch, reporting, forward test) but not
    in the bot's sizing equity (orchestrator subtracts it);
  * persisted in state.json and never flattened on shutdown.

Cash moves through the paper broker's ledger (taker fee + slippage), so the
account balance stays exact.

Selection and sizing (with `core_trend_filter` on) come from the daily lab
(app/backtest/daily_lab.py) via the SAME `target_weights` function its
simulation uses: `core_selection` trend / momentum / rank and `core_sizing`
equal / inverse_vol / vol_target, each settable or "auto" = whatever the
latest weekly lab run found to beat trend+equal on Sharpe in both halves of
years of out-of-sample history (else trend+equal, the original rule).
"""
import time
import uuid
from ..tunables import tv
from .. import db


DAILY = 86400
SMA_DAYS = 50
REBALANCE_SEC = 7 * DAILY          # drift rebalance at most weekly
STALE_DAYS = 3                     # champion mode: older data -> no trading
DRIFT_BAND = 0.20                  # ...and only when >20% off target
MIN_TRADE = 25.0                   # $ — skip dust rebalances


class CoreBook:
    def __init__(self):
        self.positions = {}        # asset -> {"qty", "entry", "opened"}
        self.last_rebalance = 0.0
        self.last_decision = {}    # asset -> {"hold": bool, "close", "sma"}
        self.realized_pnl = 0.0
        self.picks = None          # weekly momentum/rank pick {"day", "picks"}
        self.applied_pct = None    # allocation the book was last rebalanced to
        self.mode_used = ("trend", "equal")

    # ---------- settings ----------
    @staticmethod
    def _settings():
        from .. import settings
        try:
            pct = float(settings.get("core_allocation_pct"))
        except Exception:
            pct = 0.0
        if CoreBook.follows_champion():
            _, cfg = _champion()
            return (max(0.0, min(100.0, pct)) / 100.0, list(cfg.get("assets") or []), True)
        assets = [a.strip().upper() for a in
                  str(settings.get("core_assets") or "").split(",") if a.strip()]
        if assets == ["UNIVERSE"]:
            # every coin the bot tracks (each held only while above its average)
            from ..config import PRODUCTS
            assets = sorted(PRODUCTS)
        return max(0.0, min(100.0, pct)) / 100.0, assets, bool(settings.get("core_trend_filter"))

    @staticmethod
    def follows_champion():
        """`core_strategy` = "champion": hold whatever the research loop's
        current champion (app/engine/challengers.py) says — the same function
        it backtested and forward-tracked. "settings": the core_* settings."""
        from .. import settings
        try:
            return settings.get("core_strategy") == "champion"
        except Exception:
            return False

    @staticmethod
    def sma_days():
        from .. import settings
        try:
            return max(10, min(250, int(settings.get("core_sma_days"))))
        except Exception:
            return SMA_DAYS

    @staticmethod
    def modes():
        """(selection, sizing, why) - settings, with "auto" resolved from the
        latest daily-lab report (stale/missing -> the original trend+equal)."""
        from .. import settings
        sel, siz = settings.get("core_selection"), settings.get("core_sizing")
        why = "set manually"
        if "auto" in (sel, siz):
            dec, why = _lab_decision()
            sel = dec["selection"] if sel == "auto" else sel
            siz = dec["sizing"] if siz == "auto" else siz
        return sel, siz, why

    def enabled(self):
        pct, assets, _ = self._settings()
        return pct > 0 and bool(assets)

    # ---------- accounting ----------
    def value(self, market):
        v = 0.0
        for a, pos in self.positions.items():
            px = market.price(a) or pos["entry"]
            v += pos["qty"] * px
        return v

    def bot_equity(self, account_equity, market):
        """Equity the trading bot may size against. While the core is enabled
        its WHOLE allocation is reserved — including the cash it holds for
        coins currently below their average — so the bot can never spend the
        core's idle slots. Off: just the account minus anything still held."""
        from .exploration import manager as explore
        if explore.enabled():
            return explore.bot_equity(market)       # its exploration slice
        pct, assets, _ = self._settings()
        if pct > 0 and assets:
            return account_equity * (1 - pct)
        return account_equity - self.value(market)

    # ---------- decisions ----------
    @staticmethod
    def trend_ok(daily_candles, days=SMA_DAYS):
        """(hold?, last_close, sma) from DAILY candles over `days`; uses only
        COMPLETED days (drops a bar that hasn't closed yet)."""
        cs = sorted(daily_candles or [], key=lambda c: c[0])
        now = time.time()
        cs = [c for c in cs if c[0] + DAILY <= now] or cs
        if len(cs) < days:
            return None, None, None
        closes = [c[4] for c in cs[-days:]]
        sma = sum(closes) / days
        return closes[-1] > sma, closes[-1], sma

    def targets(self, account_equity, daily_by_asset, now=None):
        """{asset: target notional}."""
        pct, assets, use_filter = self._settings()
        if pct <= 0 or not assets:
            return {a: 0.0 for a in self.positions}
        if self.follows_champion():
            w = self._champion_weights(now)
            if w is None:                # no / stale data: never trade blind
                return None
            out = {a: account_equity * pct * v for a, v in w.items()}
            for a in assets:
                out.setdefault(a, 0.0)
        elif not use_filter:             # plain equal-weight hold
            out = {a: account_equity * pct / len(assets) for a in assets}
        else:
            w = self._lab_weights(assets, daily_by_asset, now)
            out = {a: account_equity * pct * w.get(a, 0.0) for a in assets}
        for a in self.positions:         # assets removed from the list -> exit
            out.setdefault(a, 0.0)
        return out

    def _lab_weights(self, assets, daily_by_asset, now=None):
        """Today's weights on completed daily candles (unknown / short history
        -> not eligible -> cash). trend / momentum run the portfolio engine's
        `trend_portfolio` — the exact function the backtests run — taking the
        row for today; the rank model keeps the daily lab's path."""
        from ..backtest import daily_lab as dl
        from .. import settings
        now = now or time.time()
        sma = self.sma_days()
        sel, siz, _ = self.modes()
        if sel in ("trend", "momentum"):
            return self._engine_weights(assets, daily_by_asset, now, sel, siz, sma)
        feats, day = {}, 0
        for a in assets:
            cs = sorted(c for c in (daily_by_asset.get(a) or []) if c[0] + DAILY <= now)
            if not cs:
                continue
            day = max(day, int(cs[-1][0]) // DAILY)
            f = dl.coin_features([c[4] for c in cs], [c[5] for c in cs], sma)
            if f:
                feats[a] = f
        dl.add_cross_features(feats)
        preds = None
        if sel == "rank":
            model = _rank_model()
            if model is None:
                sel = "trend"            # no trained model yet -> original rule
            elif feats:
                ps = list(feats)
                scores = model.predict([dl.x_row(feats[p]) for p in ps])
                preds = {p: float(s) for p, s in zip(ps, scores)}
        w, self.picks = dl.target_weights(
            feats, sel, siz, preds=preds, prev=self.picks if sel != "trend" else None,
            day=day, sma_days=sma, top_k=int(settings.get("core_top_k")),
            vol_target=float(settings.get("core_vol_target")))
        self.mode_used = (sel, siz)
        for a in assets:
            f = feats.get(a)
            gap = f["d_core"] if f else float("nan")
            self.last_decision[a] = {
                "hold": a in w, "weight": round(w.get(a, 0.0), 4),
                "close": f["close"] if f else None,
                "sma": f["close"] / (1 + gap) if f and gap == gap else None}
        return w

    def _champion_weights(self, now=None):
        """Today's weights of the champion, from the data store (the core loop
        ingests fresh daily candles into it first). The last CLOSED day's row
        of the champion's own backtest function."""
        from ..engine import panel as P, challengers as C
        name, cfg = _champion()
        p = P.from_store(cfg.get("assets"))
        today = int((now or time.time()) // DAILY)
        if not p.T or today - int(p.days[-1]) > STALE_DAYS:
            self.mode_used = ("champion", f"{name} (no fresh data: holding)")
            db.log_event("warn", f"CORE: no daily data newer than {STALE_DAYS} days for "
                         f"{cfg.get('assets')} — holding positions, no trades")
            return None
        W = C.weights(cfg, p)
        self.mode_used = ("champion", name)
        w = {c: float(v) for c, v in zip(p.coins, W[-1]) if v > 0}
        for j, c in enumerate(p.coins):
            px = p.close[-1, j]
            self.last_decision[c] = {"hold": c in w, "weight": round(w.get(c, 0.0), 4),
                                     "close": None if px != px else float(px),
                                     "as_of_day": int(p.days[-1]), "strategy": name}
        return w

    def _engine_weights(self, assets, daily_by_asset, now, sel, siz, sma):
        from ..engine import panel as P, strategies as S, features as F
        from .. import settings
        p = P.from_candles({a: daily_by_asset.get(a) or [] for a in assets}, now=now)
        if not p.T:
            return {}
        # replay history so path state (weekly picks, the trend buffer's
        # in/out state) is rebuilt from data exactly as the backtest has it
        w = S.latest(p, lookback_days=None, selection=sel, sizing=siz, sma=sma,
                     hysteresis=float(settings.get("core_hysteresis")),
                     top_k=int(settings.get("core_top_k")),
                     vol_target=float(settings.get("core_vol_target")),
                     tranches=int(settings.get("core_tranches")))
        self.mode_used = (sel, siz)
        sma_now = F.rolling_mean(p.close, sma)[-1]
        for j, a in enumerate(p.coins):
            c, m = p.close[-1, j], sma_now[j]
            self.last_decision[a] = {"hold": a in w, "weight": round(w.get(a, 0.0), 4),
                                     "close": None if c != c else float(c),
                                     "sma": None if m != m else float(m)}
        return w

    # ---------- execution ----------
    def _fill(self, px, side):
        slip = px * tv("slippage_bps") / 1e4
        return px + slip if side == "buy" else px - slip

    def _buy(self, broker, a, notional, px):
        fill = self._fill(px, "buy")
        # Buy what the cash affords after the fee: at a 100% allocation the
        # targets add up to all of equity, so fees paid on the first buy leave
        # the last one a little short — refusing it outright left that slot in
        # cash forever (the target never changes, so it was refused hourly).
        notional = min(notional, broker.cash / (1 + tv("fee_rate")))
        fee = notional * tv("fee_rate")
        if notional < MIN_TRADE:
            return False
        if not broker._reserve(f"core-buy-{uuid.uuid4().hex}", notional + fee, f"core buy {a}"):
            return False
        qty = notional / fill
        pos = self.positions.get(a)
        if pos:
            tot = pos["qty"] + qty
            pos["entry"] = (pos["entry"] * pos["qty"] + fill * qty) / tot
            pos["qty"] = tot
        else:
            self.positions[a] = {"qty": qty, "entry": fill, "opened": time.time()}
        db.log_trade(a, "buy", qty, fill, fee, "CORE holding")
        return True

    def _sell(self, broker, a, qty, px, reason):
        pos = self.positions.get(a)
        if not pos or qty <= 0:
            return False
        qty = min(qty, pos["qty"])
        fill = self._fill(px, "sell")
        gross = qty * fill
        fee = gross * tv("fee_rate")
        pnl = gross - fee - qty * pos["entry"]
        broker._settle(f"core-sell-{uuid.uuid4().hex}", gross - fee, f"core sell {a}")
        self.realized_pnl += pnl
        pos["qty"] -= qty
        if pos["qty"] * fill < 1.0:
            self.positions.pop(a, None)
        db.log_trade(a, "sell", qty, fill, fee, f"CORE {reason}", pnl)
        return True

    def rebalance(self, broker, market, account_equity, daily_by_asset, now=None):
        """Bring holdings to target. Trend switches (target 0 <-> share) act
        immediately; otherwise drift is only corrected weekly and when >20%
        off. Returns a list of action strings."""
        from ..execution.paper import _valid_price
        now = now or time.time()
        tgt = self.targets(account_equity, daily_by_asset, now)
        if tgt is None:
            return []
        weekly = now - self.last_rebalance >= REBALANCE_SEC
        # An operator change of `core_allocation_pct` is a decision, not price
        # drift: rebalance every holding to the new targets once (the 20%
        # drift band would otherwise ignore e.g. 100% -> 90%).
        pct = self._settings()[0]
        realloc = self.applied_pct is not None and abs(pct - self.applied_pct) > 1e-9
        actions = []
        for a, target in tgt.items():
            px = market.price(a)
            if not _valid_price(px):
                continue
            pos = self.positions.get(a)
            cur = pos["qty"] * px if pos else 0.0
            switch = (target == 0) != (cur < MIN_TRADE)
            drift = target > 0 and abs(cur - target) > DRIFT_BAND * target
            # checked DAILY, as simulated (engine.backtest.simulate_drift
            # with band_rel = DRIFT_BAND); the weekly-only check never was
            if realloc and abs(cur - target) >= MIN_TRADE:
                drift = True
            if not (switch or drift):
                continue
            if target == 0 and pos:
                if self._sell(broker, a, pos["qty"], px,
                              f"exit ({'+'.join(self.mode_used)} no longer holds it)"):
                    actions.append(f"sold all {a}")
            elif cur < target:
                if self._buy(broker, a, target - cur, px):
                    actions.append(f"bought ${target - cur:,.0f} {a}")
            elif cur > target and pos:
                if self._sell(broker, a, (cur - target) / px, px, "rebalance trim"):
                    actions.append(f"trimmed ${cur - target:,.0f} {a}")
        if weekly:
            self.last_rebalance = now
        self.applied_pct = pct
        return actions

    # ---------- persistence / reporting ----------
    def to_dict(self):
        return {"positions": self.positions, "last_rebalance": self.last_rebalance,
                "realized_pnl": self.realized_pnl, "picks": self.picks,
                "applied_pct": self.applied_pct}

    def load_dict(self, d):
        if not d:
            return
        self.positions = {a: dict(p) for a, p in (d.get("positions") or {}).items()}
        self.last_rebalance = d.get("last_rebalance", 0.0)
        self.realized_pnl = d.get("realized_pnl", 0.0)
        self.picks = d.get("picks")
        self.applied_pct = d.get("applied_pct")

    def snapshot(self, market):
        pct, assets, use_filter = self._settings()
        sel, siz, why = self.modes()
        if self.follows_champion():
            name, cfg = _champion()
            sel, siz, why = "champion", name, f"research-loop champion: {cfg}"
        return {"enabled": self.enabled(), "allocation_pct": round(pct * 100, 1),
                "assets": assets, "trend_filter": use_filter, "sma_days": self.sma_days(),
                "selection": sel, "sizing": siz, "mode_why": why,
                "value": round(self.value(market), 2),
                "positions": {a: {"qty": p["qty"], "entry": p["entry"],
                                  "price": market.price(a)} for a, p in self.positions.items()},
                "trend": self.last_decision, "realized_pnl": round(self.realized_pnl, 2)}


_LAB = {"mtime": None, "report": None}
_MODEL = {"mtime": None, "model": None}


def _reports_dir():
    import os
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "reports")


def _lab_decision(now=None):
    """({"selection", "sizing"}, why) from reports/daily_lab_latest.json; the
    original trend+equal when missing or older than `daily_lab_max_age_days`."""
    import json
    import os
    base = {"selection": "trend", "sizing": "equal"}
    path = os.path.join(_reports_dir(), "daily_lab_latest.json")
    try:
        m = os.path.getmtime(path)
        if _LAB["mtime"] != m:
            with open(path) as f:
                _LAB.update(mtime=m, report=json.load(f))
    except Exception:
        return base, "no daily-lab report yet"
    rep = _LAB["report"] or {}
    from .. import settings
    max_age = float(settings.get("daily_lab_max_age_days")) * DAILY
    if not rep.get("ok") or (now or time.time()) - rep.get("ran_at", 0) > max_age:
        return base, "daily-lab report missing or stale"
    dec = rep.get("decision") or {}
    if (dec.get("selection") not in ("trend", "momentum", "rank")
            or dec.get("sizing") not in ("equal", "inverse_vol", "vol_target")):
        return base, "daily-lab decision unreadable"
    return dec, f"daily lab: {dec.get('variant')} ({dec.get('why')})"


def _champion():
    from ..engine.challengers import champion
    return champion()


def _rank_model():
    import os
    from ..backtest import daily_lab as dl
    path = os.path.join(_reports_dir(), "daily_rank_model.pkl")
    try:
        m = os.path.getmtime(path)
    except OSError:
        return None
    if _MODEL["mtime"] != m:
        _MODEL.update(mtime=m, model=dl.load_model(path))
    return _MODEL["model"]


core = CoreBook()
