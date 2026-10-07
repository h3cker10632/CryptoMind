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
from .research import research_cache
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


def _ttl_bucket(ttl_hours):
    if ttl_hours is None:
        return "unknown"
    if ttl_hours <= 1:
        return "0-1h"
    if ttl_hours <= 24:
        return "1-24h"
    if ttl_hours <= 24 * 7:
        return "1-7d"
    if ttl_hours <= 24 * 30:
        return "1-4w"
    return ">1mo"


def _forecast_metrics(rows, now=None):
    now = time.time() if now is None else float(now)
    resolved = [row for row in rows if row.get("resolved_outcome0") in (0, 1)]
    unresolved = [row for row in rows if row.get("resolved_outcome0") is None]

    def log_loss(probability, outcome):
        probability = max(1e-6, min(1.0 - 1e-6, probability))
        return -(outcome * math.log(probability)
                 + (1 - outcome) * math.log(1.0 - probability))

    bot_brier, market_brier = [], []
    bot_log_loss, market_log_loss = [], []
    for row in resolved:
        outcome = int(row["resolved_outcome0"])
        bot_p = max(0.0, min(1.0, float(row["predicted_p0"])))
        market_p = max(0.0, min(1.0, float(row["market_p0"])))
        bot_brier.append((bot_p - outcome) ** 2)
        market_brier.append((market_p - outcome) ** 2)
        bot_log_loss.append(log_loss(bot_p, outcome))
        market_log_loss.append(log_loss(market_p, outcome))

    def mean(values):
        return round(sum(values) / len(values), 6) if values else None

    covered = [row for row in rows if row.get("evidence")]
    abstentions = 0
    evidence_by_id = {}
    for row in rows:
        features = row.get("features") or {}
        citations = (features.get("research_citations", [])
                    if isinstance(features, dict) else [])
        if not citations:
            abstentions += 1
        for doc in row.get("evidence") or []:
            doc_id = doc.get("id")
            if doc_id:
                evidence_by_id.setdefault(str(doc_id), doc)
    fresh_ages = [max(0.0, now - float(doc.get("published_ts", 0))) / 3600.0
                  for doc in evidence_by_id.values()
                  if 0 <= now - float(doc.get("published_ts", 0)) <= 72 * 3600]

    strategy_performance = {}
    for strategy in signals.STRATEGIES:
        brier, losses, active_market_brier = [], [], []
        for row in resolved:
            outcome = int(row["resolved_outcome0"])
            features = row.get("features") or {}
            edge_scale = (features.get("edge_scale", 0.06)
                          if isinstance(features, dict) else 0.06)
            lean = (row.get("votes") or {}).get(strategy, 0.0)
            if abs(float(lean)) < 1e-6:
                continue
            probability = max(0.0, min(1.0,
                               float(row["market_p0"]) + float(lean) * float(edge_scale)))
            brier.append((probability - outcome) ** 2)
            losses.append(log_loss(probability, outcome))
            active_market_brier.append((float(row["market_p0"]) - outcome) ** 2)
        strategy_performance[strategy] = {
            "n": len(brier),
            "brier": mean(brier),
            "log_loss": mean(losses),
            "brier_delta_vs_market": (
                round(mean(brier) - mean(active_market_brier), 6)
                if brier else None),
        }

    bot_brier_mean = mean(bot_brier)
    market_brier_mean = mean(market_brier)
    bot_loss_mean = mean(bot_log_loss)
    market_loss_mean = mean(market_log_loss)
    return {
        "forecast_count": len(rows),
        "scored_forecast_count": len(resolved),
        "unresolved_forecast_count": len(unresolved),
        "research_covered_count": len(covered),
        "research_abstention_count": abstentions,
        "research_coverage": round(len(covered) / len(rows), 4) if rows else 0.0,
        "bot_brier": bot_brier_mean,
        "market_brier": market_brier_mean,
        "brier_delta": (round(bot_brier_mean - market_brier_mean, 6)
                         if bot_brier_mean is not None and market_brier_mean is not None
                         else None),
        "bot_log_loss": bot_loss_mean,
        "market_log_loss": market_loss_mean,
        "log_loss_delta": (round(bot_loss_mean - market_loss_mean, 6)
                           if bot_loss_mean is not None and market_loss_mean is not None
                           else None),
        "strategy_performance": strategy_performance,
        "fresh_evidence_documents": len(fresh_ages),
        "stale_evidence_documents": len(evidence_by_id) - len(fresh_ages),
        "evidence_freshness_hours": mean(fresh_ages),
    }


class PolymarketEngine:
    def __init__(self):
        self.running = False
        self.task: asyncio.Task | None = None
        self.last_tick_ts = 0.0
        self.last_error = ""
        self.decisions: list[dict] = []      # rolling window for the dashboard
        self._price_cache: dict[str, float] = {}   # token_id -> latest mid
        self._replayed_resolutions = False
        self._tick_resolution_cache = {}

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

    def _resolution(self, condition_id):
        if condition_id not in self._tick_resolution_cache:
            self._tick_resolution_cache[condition_id] = client.resolution(condition_id)
        return self._tick_resolution_cache[condition_id]

    def _record_forecast(self, market, signal, evidence, research_result, edge_scale):
        forecast_ts = time.time()
        ttl_hours = market.get("ttl_hours")
        due_ts = (forecast_ts + max(0.0, float(ttl_hours)) * 3600.0
                  if ttl_hours is not None else forecast_ts + 30 * 86400.0)
        feature_snapshot = learner.online.features(market)
        features = {
            "values": feature_snapshot,
            "ttl_hours": ttl_hours,
            "edge_scale": edge_scale,
            "question": str(market.get("question") or "")[:280],
            "category": str(market.get("category") or "")[:80],
            "research_citations": research_result.get("citations", []),
        }
        try:
            db.record_pm_forecast({
                "condition_id": market["condition_id"],
                "ttl_bucket": _ttl_bucket(ttl_hours),
                "forecast_ts": forecast_ts,
                "predicted_p0": signal["fair_p0"],
                "market_p0": market["prices"][0],
                "votes": signal["leans"],
                "features": features,
                "evidence": evidence,
                "resolution_due_ts": due_ts,
            })
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Forecast ledger: {exc}"[:200]

    def _apply_resolved_forecast(self, row):
        feature_data = row.get("features") or {}
        feature_snapshot = (feature_data.get("values")
                            if isinstance(feature_data, dict) else feature_data)
        ttl_hours = (feature_data.get("ttl_hours")
                     if isinstance(feature_data, dict) else None)
        if not isinstance(feature_snapshot, list):
            return False
        try:
            return learner.score_forecast(
                row["id"], ttl_hours, row.get("votes") or {},
                row["resolved_outcome0"], feature_snapshot)
        except (KeyError, TypeError, ValueError) as exc:
            self.last_error = f"Forecast learning: {exc}"[:200]
            return False

    def _settle_forecasts(self):
        if not self._replayed_resolutions:
            after_id = 0
            while True:
                rows = db.resolved_pm_forecasts(after_id=after_id, limit=1000)
                if not rows:
                    break
                for row in rows:
                    self._apply_resolved_forecast(row)
                after_id = rows[-1]["id"]
                if len(rows) < 1000:
                    break
            self._replayed_resolutions = True

        now = time.time()
        checked = 0
        for row in db.pending_pm_forecasts(limit=100, now=now):
            if checked >= 3:
                break
            db.mark_pm_resolution_attempt(row["id"], now)
            checked += 1
            try:
                resolution = self._resolution(row["condition_id"])
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"Resolution lookup: {exc}"[:200]
                continue
            if not resolution or not resolution.get("resolved"):
                continue
            winner = resolution.get("winning_index")
            prices = resolution.get("prices") or []
            if (winner not in (0, 1) or len(prices) != 2
                    or prices[winner] < 0.99 or prices[1 - winner] > 0.01):
                continue
            outcome0_won = int(winner == 0)
            if db.label_pm_forecast(row["id"], outcome0_won):
                row["resolved_outcome0"] = outcome0_won
                self._apply_resolved_forecast(row)

    # ----------------------------- one cycle ----------------------------
    def tick(self) -> dict:
        """Run a single decision cycle synchronously. Safe to call by hand."""
        fee = tv("pm_fee_rate")
        slip = tv("pm_slippage")
        markets = client.fetch_markets(
            limit=int(tv("pm_universe_size")),
            min_liquidity=tv("pm_min_liquidity"))
        self._tick_resolution_cache = {}
        if not markets and client.last_error:
            self.last_error = client.last_error
        self._refresh_prices(markets)

        # (2) settle resolved holdings ------------------------------------
        for tid in list(broker.positions.keys()):
            pos = broker.positions[tid]
            res = self._resolution(pos["condition_id"])
            if res and res["resolved"]:
                won = (res["winning_index"] == pos["outcome_index"])
                t = broker.resolve(tid, won)
                if t:
                    learner.on_trade_closed(t)
                    db.log_event("pm", f"RESOLVED {'WON' if won else 'LOST'} "
                                 f"{pos['outcome']} pnl={t['pnl']:+.2f}")

        # (3) early exits on open positions -------------------------------
        for t in broker.manage(self._mid_lookup, fee, slip):
            learner.on_trade_closed(t)
            db.log_event("pm", f"EXIT {t['exit_reason']} {t['outcome']} "
                         f"pnl={t['pnl']:+.2f}")

        self._settle_forecasts()

        # (4) evaluate + (optionally) open --------------------------------
        # Polymarket's own bankroll: the main account's ledger migration and
        # kill switch / daily halt don't apply to it. Its own evidence gate
        # (skill_gate.py) does: bets only once the forecast ledger shows the
        # bot beating the market price (or pm_trade_mode "on").
        auto = bool(app_settings.get("pm_auto_trade"))
        from .skill_gate import status as _skill
        gate = _skill()
        self.trade_gate = gate
        auto = auto and gate["open"]
        edge_scale = tv("pm_edge_scale")
        max_pos = int(tv("pm_max_positions"))
        llm_infl = tv("pm_llm_influence")
        research_infl = tv("pm_research_influence")
        # its own bankroll: bets are sized off what Polymarket itself has grown
        # (or shrunk) to, never the main account
        pm_equity, pm_cash = broker.equity(self._mid_lookup), broker.cash
        # refresh a bounded batch of LLM leans (cached; no-op if the LLM sleeve
        # is off) BEFORE evaluating, so the `llm` strategy + ML feature see them.
        try:
            llm_advisor.refresh(markets, int(tv("pm_llm_max_queries")))
        except Exception as e:                        # noqa: BLE001
            self.last_error = f"LLM refresh: {e}"
        try:
            research_cache.refresh(markets)
        except Exception as e:                        # noqa: BLE001
            self.last_error = f"Research refresh: {e}"[:200]
        try:
            llm_advisor.refresh_research(
                markets, research_cache, int(tv("pm_llm_max_queries")))
        except Exception as e:                        # noqa: BLE001
            self.last_error = f"Research advisor: {e}"[:200]
        opened = 0
        candidates = []
        for m in markets:
            ll = llm_advisor.lean(m["condition_id"])
            m["llm_lean"] = ll               # for the online-model ML feature
            evidence = research_cache.evidence(m["condition_id"])
            research_result = llm_advisor.research_result(
                m["condition_id"], evidence=evidence)
            m["research_lean"] = research_result.get("lean", 0.0)
            w = learner.weights(m.get("ttl_hours"))
            sig = signals.evaluate(m, w, edge_scale, llm_lean=ll,
                                   llm_influence=llm_infl,
                                   research_lean=m["research_lean"],
                                   research_influence=research_infl)
            self._record_forecast(m, sig, evidence, research_result, edge_scale)
            if gate.get("use_calibration"):            # the ledger keeps the raw forecast
                from . import calibration
                sig = calibration.adjust(m, sig, calibration.status()["model"])
            if m["condition_id"] in {p["condition_id"]
                                     for p in broker.positions.values()}:
                continue
            if sig["outcome_index"] is None:
                continue
            stake, why = size_bet(pm_equity, pm_cash, sig["price"], sig["edge"],
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
        """Wipe Polymarket's book and restart its bankroll (default: the
        `pm_start_cash` setting). The main account is untouched."""
        if start_cash is None:
            start_cash = float(app_settings.get("pm_start_cash") or PM_START_CASH)
        broker.reset(start_cash)
        self.decisions.clear()
        return {"ok": True, **broker.stats()}

    # ---------------------------- reporting ----------------------------
    def learning_report(self) -> dict:
        report = learner.stats()
        report_limit = 10000
        report.update(_forecast_metrics(db.pm_forecast_rows(limit=report_limit)))
        report["forecast_total_count"] = db.pm_forecast_count()
        report["forecast_metrics_window"] = min(
            report_limit, report["forecast_total_count"])
        report["forecast_metrics_scope"] = "newest_forecasts"
        return report

    def forecasts(self, limit=25) -> dict:
        """Return recent forecast snapshots and their source evidence for inspection."""
        selected = db.pm_forecast_rows(limit=max(1, min(100, int(limit))))
        forecasts = []
        for row in selected:
            features = row.get("features") or {}
            if not isinstance(features, dict):
                features = {}
            forecasts.append({
                "condition_id": row["condition_id"],
                "question": features.get("question") or row["condition_id"],
                "category": features.get("category") or "",
                "forecast_ts": row["forecast_ts"],
                "ttl_bucket": row["ttl_bucket"],
                "predicted_p0": row["predicted_p0"],
                "market_p0": row["market_p0"],
                "resolved_outcome0": row.get("resolved_outcome0"),
                "votes": row.get("votes") or {},
                "research_citations": features.get("research_citations") or [],
                "evidence": row.get("evidence") or [],
            })
        return _json_safe({"forecasts": forecasts})

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
            "trade_gate": getattr(self, "trade_gate", None),
            "execution": exec_status(),
            "last_tick_ts": self.last_tick_ts,
            "last_error": self.last_error or client.last_error,
            "equity": round(eq, 2),
            # its own bankroll: growth since the last reset
            "return_pct": round((eq / broker.start_cash - 1) * 100, 2)
                          if broker.start_cash else 0.0,
            "broker": broker.stats(),
            "positions": positions,
            "decisions": self.decisions[:25],
            "learning": self.learning_report(),
            "llm": llm_advisor.stats(),
            "research": research_cache.stats(),
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


engine = PolymarketEngine()
