"""Polymarket trading engine — the orchestration that wires the standalone parts
together and (optionally) runs itself on a background loop.

One `tick()` does, in order:
  1. fetch the most-liquid active binary markets (read-only),
  2. settle any held position whose market has RESOLVED (the honest exit) — this
     is what un-starves the trade + signal learners,
  3. run early take-profit / stop-loss on open positions,
  4. score EVERY fetched market's signal against nothing yet (candidates) and, if
     auto-trade is on, open paper bets that clear the sizer's edge/cost gate,
  5. snapshot equity + decisions for the dashboard.

Everything is OFF by default: the loop only runs when `polymarket_enabled` is set,
and it only OPENS bets when `pm_auto_trade` is also set. Live on-chain execution
is a further, separately-gated switch (see execution.py). Decisions are recorded
(opened / skipped-with-reason) so the dashboard can answer "why aren't we betting"
the same way the crypto side does.
"""
from __future__ import annotations

import asyncio
import math
import time

from ... import db
from ... import settings as app_settings
from ...tunables import tv
from . import signals
from .broker import broker, PM_START_CASH
from .client import client
from .execution import executor, status as exec_status
from .learner import learner, regime_of
from .llm import llm_advisor
from .risk import size as size_bet, exit_levels


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


class PolymarketEngine:
    def __init__(self):
        self.running = False
        self.task: asyncio.Task | None = None
        self.last_tick_ts = 0.0
        self.last_error = ""
        self.decisions: list[dict] = []      # rolling window for the dashboard
        self._price_cache: dict[str, float] = {}   # token_id -> latest mid

    # ----------------------------- helpers -----------------------------
    def _mid_lookup(self, token_id: str):
        return self._price_cache.get(token_id)

    def _refresh_prices(self, markets):
        for m in markets:
            for i, tid in enumerate(m["token_ids"]):
                self._price_cache[tid] = m["prices"][i]

    def _record(self, market, action, reason, extra=None):
        d = {"ts": time.time(), "question": market.get("question", "")[:90],
             "category": market.get("category", ""),
             "action": action, "reason": reason}
        if extra:
            d.update(extra)
        self.decisions.insert(0, d)
        del self.decisions[200:]

    # ----------------------------- one cycle ----------------------------
    def tick(self) -> dict:
        """Run a single decision cycle synchronously. Safe to call by hand."""
        fee = tv("pm_fee_rate")
        slip = tv("pm_slippage")
        markets = client.fetch_markets(
            limit=int(tv("pm_universe_size")),
            min_liquidity=tv("pm_min_liquidity"))
        if not markets and client.last_error:
            self.last_error = client.last_error
        self._refresh_prices(markets)
        by_condition = {m["condition_id"]: m for m in markets}

        # (2) settle resolved holdings ------------------------------------
        for tid in list(broker.positions.keys()):
            pos = broker.positions[tid]
            res = client.resolution(pos["condition_id"])
            if res and res["resolved"]:
                won = (res["winning_index"] == pos["outcome_index"])
                t = broker.resolve(tid, won)
                if t:
                    learner.on_trade_closed(t)
                    # signal teacher: outcome 0 winning?
                    leans0 = _leans_from_votes(pos)
                    learner.score_resolution(
                        {**pos, "prices": res["prices"],
                         "ttl_hours": None, "spread": 0.0,
                         "mom_1d": 0.0, "liquidity": 0.0,
                         "llm_lean": leans0.get("llm", 0.0)},
                        leans0,
                        1 if res["winning_index"] == 0 else 0)
                    db.log_event("pm", f"RESOLVED {'WON' if won else 'LOST'} "
                                 f"{pos['outcome']} pnl={t['pnl']:+.2f}")

        # (3) early exits on open positions -------------------------------
        for t in broker.manage(self._mid_lookup, fee, slip):
            learner.on_trade_closed(t)
            db.log_event("pm", f"EXIT {t['exit_reason']} {t['outcome']} "
                         f"pnl={t['pnl']:+.2f}")

        # (4) evaluate + (optionally) open --------------------------------
        auto = bool(app_settings.get("pm_auto_trade"))
        edge_scale = tv("pm_edge_scale")
        max_pos = int(tv("pm_max_positions"))
        llm_infl = tv("pm_llm_influence")
        equity = broker.equity(self._mid_lookup)
        # refresh a bounded batch of LLM leans (cached; no-op if the LLM sleeve
        # is off) BEFORE evaluating, so the `llm` strategy + ML feature see them.
        try:
            llm_advisor.refresh(markets, int(tv("pm_llm_max_queries")))
        except Exception as e:                        # noqa: BLE001
            self.last_error = f"LLM refresh: {e}"
        opened = 0
        candidates = []
        for m in markets:
            if m["condition_id"] in {p["condition_id"]
                                     for p in broker.positions.values()}:
                continue
            ll = llm_advisor.lean(m["condition_id"])
            m["llm_lean"] = ll               # for the online-model ML feature
            w = learner.weights(m.get("ttl_hours"))
            sig = signals.evaluate(m, w, edge_scale, llm_lean=ll,
                                   llm_influence=llm_infl)
            if sig["outcome_index"] is None:
                continue
            stake, why = size_bet(equity, broker.cash, sig["price"], sig["edge"],
                                  sig["confidence"], m)
            cand = {"market": m, "sig": sig, "stake": stake, "why": why}
            candidates.append(cand)

        # rank actionable candidates by edge*confidence
        actionable = sorted(
            [c for c in candidates if c["stake"] > 0],
            key=lambda c: c["sig"]["edge"] * c["sig"]["confidence"], reverse=True)

        for c in candidates:
            if c["stake"] <= 0:
                self._record(c["market"], "skip", c["why"],
                             {"edge": c["sig"]["edge"],
                              "confidence": c["sig"]["confidence"]})

        if auto:
            for c in actionable:
                if len(broker.positions) >= max_pos:
                    self._record(c["market"], "skip", "max positions reached")
                    continue
                m, sig = c["market"], c["sig"]
                sl, tp = exit_levels(sig["price"])
                pos = broker.open(
                    m, sig["outcome_index"], sig["price"], c["stake"],
                    fee, slip, sl, tp, reason="signal",
                    votes=sig["votes"], regime=regime_of(m.get("ttl_hours")),
                    edge=sig["edge"])
                if pos:
                    opened += 1
                    self._record(m, "open", "opened",
                                 {"outcome": sig["outcome"], "stake": c["stake"],
                                  "edge": sig["edge"]})
                    db.log_event("pm", f"OPEN {sig['outcome']} @ {sig['price']:.3f} "
                                 f"${c['stake']:.2f} edge={sig['edge']:.3f}")
                else:
                    self._record(m, "skip", "broker rejected fill")
        else:
            for c in actionable[:5]:
                self._record(c["market"], "would-open (auto-trade off)",
                             c["why"], {"outcome": c["sig"]["outcome"],
                                        "stake": c["stake"],
                                        "edge": c["sig"]["edge"]})

        learner.decay()
        self.last_tick_ts = time.time()
        eq = broker.equity(self._mid_lookup)
        return {"ok": True, "markets": len(markets), "opened": opened,
                "actionable": len(actionable), "equity": round(eq, 2)}

    # ----------------------------- loop --------------------------------
    async def _loop(self):
        db.log_event("system", "Polymarket engine started")
        while self.running:
            try:
                await asyncio.to_thread(self.tick)
                self.last_error = ""
            except Exception as e:                    # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                db.log_event("warn", f"Polymarket tick failed: {e}")
            interval = int(tv("pm_interval_sec"))
            for _ in range(max(5, interval)):
                if not self.running:
                    break
                await asyncio.sleep(1)
        db.log_event("system", "Polymarket engine stopped")

    def start(self) -> dict:
        if not app_settings.get("polymarket_enabled"):
            return {"ok": False,
                    "error": "polymarket_enabled is off — enable it in Settings"}
        if self.running:
            return {"ok": True, "already": True, **self.snapshot()}
        self.running = True
        try:
            self.task = asyncio.create_task(self._loop())
        except RuntimeError:
            # no running event loop (e.g. called from a sync test) — caller can
            # still drive tick() by hand.
            self.running = False
            return {"ok": False, "error": "no event loop; call tick() directly"}
        return {"ok": True, **self.snapshot()}

    def stop(self) -> dict:
        self.running = False
        return {"ok": True, **self.snapshot()}

    def reset(self, start_cash: float | None = None) -> dict:
        broker.reset(start_cash if start_cash is not None else PM_START_CASH)
        self.decisions.clear()
        return {"ok": True, **broker.stats()}

    # ---------------------------- reporting ----------------------------
    def peek(self, limit=15) -> dict:
        """Read-only: top markets with the current signal read (no trading)."""
        markets = client.fetch_markets(limit=limit,
                                       min_liquidity=tv("pm_min_liquidity"))
        edge_scale = tv("pm_edge_scale")
        rows = []
        for m in markets:
            w = learner.weights(m.get("ttl_hours"))
            sig = signals.evaluate(m, w, edge_scale)
            rows.append({
                "question": m["question"], "category": m["category"],
                "outcomes": m["outcomes"], "prices": m["prices"],
                "liquidity": round(m["liquidity"], 0),
                "volume_24h": round(m["volume_24h"], 0),
                "ttl_hours": round(m["ttl_hours"], 1) if m["ttl_hours"] else None,
                "pick": sig["outcome"], "edge": sig["edge"],
                "confidence": sig["confidence"], "fair": sig["fair"],
            })
        return _json_safe({"ok": bool(markets), "count": len(rows),
                           "markets": rows, "error": client.last_error})

    def snapshot(self) -> dict:
        eq = broker.equity(self._mid_lookup)
        positions = []
        for tid, p in broker.positions.items():
            mid = self._mid_lookup(tid) or p["entry"]
            positions.append({
                "question": p["question"][:80], "outcome": p["outcome"],
                "shares": round(p["shares"], 2), "entry": round(p["entry"], 3),
                "mid": round(mid, 3), "cost": round(p["cost"], 2),
                "value": round(p["shares"] * mid, 2),
                "unrealized": round(p["shares"] * mid - p["cost"], 2),
                "edge_at_entry": p.get("edge_at_entry", 0.0),
            })
        return _json_safe({
            "running": self.running,
            "enabled": bool(app_settings.get("polymarket_enabled")),
            "auto_trade": bool(app_settings.get("pm_auto_trade")),
            "execution": exec_status(),
            "last_tick_ts": self.last_tick_ts,
            "last_error": self.last_error or client.last_error,
            "equity": round(eq, 2),
            "broker": broker.stats(),
            "positions": positions,
            "decisions": self.decisions[:25],
            "learning": learner.stats(),
            "llm": llm_advisor.stats(),
        })

    def trades(self, limit=100) -> dict:
        rows = [{
            "question": t["question"][:80], "outcome": t["outcome"],
            "entry": round(t["entry"], 3), "exit": round(t["exit"], 3),
            "cost": round(t["cost"], 2), "pnl": round(t["pnl"], 2),
            "return_pct": round(t["return_pct"], 4),
            "reason": t["exit_reason"], "closed": t["closed"],
        } for t in broker.closed_trades[-limit:][::-1]]
        return _json_safe({"trades": rows, "stats": broker.stats()})


def _leans_from_votes(pos: dict) -> dict:
    """Recover per-strategy leans (signed toward outcome 0) from a stored
    position's votes (which were signed toward the traded side)."""
    sign = 1.0 if pos.get("outcome_index", 0) == 0 else -1.0
    return {k: v * sign for k, v in (pos.get("votes") or {}).items()}


engine = PolymarketEngine()
