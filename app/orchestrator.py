"""Orchestrator / Control Plane — schedules all engines, runs the decision
loop, enforces risk gates, and drives the self-improvement cycle."""
import asyncio, time, random
from .config import TICK_SEC
from .tunables import tv
from . import db
from .data.market import market
from .data.research import research
from .nlp.sentiment import nlp
from .signals.engine import engine
from .execution.paper import broker
from .risk.manager import risk
from .learn.loop import learner
from .risk.stance import stance
from .strategies.hedge import hedger


def _calendar_stats():
    from .data.calendar import calendar
    return calendar.stats()


class Orchestrator:
    def __init__(self):
        self.running = True      # trading enabled (human-in-the-loop toggle)
        self.tick_count = 0
        self.last_risk_status = {}
        self.started = time.time()
        self._blackout_logged = False
        self.shadow = None       # ShadowBroker, set by main.py

    async def decision_loop(self):
        await asyncio.sleep(10)  # let feeds warm up
        consecutive_failures = 0
        while True:
            try:
                self.tick()
                self.last_tick_ts = time.time()
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                db.log_event("error", f"Orchestrator tick failed: {e}")
                if consecutive_failures == 5:
                    from .alerts import alert
                    alert("critical", "Decision loop failing",
                          f"5 consecutive tick failures. Last error: {str(e)[:150]}")
            await asyncio.sleep(TICK_SEC)

    async def watchdog(self):
        """Alerts if the decision loop stalls (deadlock / event-loop jam)."""
        from .alerts import alert
        await asyncio.sleep(120)
        while True:
            await asyncio.sleep(60)
            stalled = time.time() - getattr(self, "last_tick_ts", time.time())
            if stalled > TICK_SEC * 6:
                alert("critical", "Decision loop STALLED",
                      f"No tick for {int(stalled)}s (expected every {TICK_SEC}s).")

    async def reconcile_loop(self):
        """Periodically reconcile the shadow OMS against its venue (source of
        truth) and prune the equity table. Runs off the hot decision path."""
        await asyncio.sleep(45)
        n = 0
        while True:
            try:
                if self.shadow is not None:
                    await asyncio.to_thread(self.shadow.reconcile)
                n += 1
                if n % 30 == 0:          # ~ every 30 min: retention on equity
                    await asyncio.to_thread(db.prune_equity)
            except Exception as e:
                db.log_event("error", f"reconcile loop failed: {e}")
            await asyncio.sleep(60)

    def tick(self):
        self.tick_count += 1
        if not market.tickers:
            return

        # Did THIS interval offer a real chance to trade? Used to tell the RL
        # agent whether flat equity was a wise sit-out or just a locked door
        # (every candidate fee-gated). Seeded True if we already hold exposure —
        # an open position is live risk being managed, so the interval counts.
        tradable = len(broker.positions) > 0

        # 1. NLP over latest research corpus
        if research.documents:
            nlp.process(research.documents, research.fear_greed)

        # 2. record prices + feature snapshots for the learning stack
        learner.observe_prices(market)
        learner.collect_features(market, nlp)

        # 3. signals with current adaptive weights
        engine.set_weights(learner.weights)
        signals = engine.compute(market, nlp)

        # 4. risk update
        regime = market.regime()
        equity = broker.equity(market)
        self.last_risk_status = risk.update(equity, regime)

        # 5. manage open positions (stops / take-profits / trailing)
        # during a macro blackout, tighten trailing stops (1.2x ATR vs 2.5x)
        from .data.calendar import calendar
        blackout_active, blackout_ev = calendar.blackout()
        trail_mult = 1.2 if blackout_active else tv("trail_atr_mult")
        if blackout_active and not self._blackout_logged:
            db.log_event("risk", f"⛔ MACRO BLACKOUT active: {blackout_ev['title']} — "
                                 f"new entries blocked, trailing stops tightened to 1.2x ATR")
            from .alerts import alert
            alert("warning", "Macro blackout active",
                  f"{blackout_ev['title']} — new entries blocked, "
                  f"trailing stops tightened.")
            self._blackout_logged = True
        elif not blackout_active and self._blackout_logged:
            db.log_event("risk", "Macro blackout lifted — normal trading resumed")
            self._blackout_logged = False
        # market-neutral pair hedging (statistical divergence, net exposure ~0)
        try:
            hedger.tick(market, broker, risk)
        except Exception as e:
            db.log_event("error", f"hedger tick failed: {e}")

        n_before = len(broker.closed_trades)
        # trailing stop keys off the SAME swing ATR the entry was sized on, so
        # the whole trade lifecycle lives on one honest (higher-tf) horizon.
        broker.manage(market, trail_mult,
                      lambda p: (market.features(p) or {}).get("atr_swing"))
        for t in broker.closed_trades[n_before:]:
            risk.on_trade_closed(t)
            learner.on_trade_closed(t)
            if self.shadow is not None and t["product"] in self.shadow.positions:
                try:
                    self.shadow.mirror_close(t["product"], t.get("exit", t["entry"]))
                except Exception as e:
                    db.log_event("error", f"shadow close mirror failed: {e}")

        # 6. exits on signal flip (direction-aware: close a long on a
        # confident bearish signal, close a short on a confident bullish one)
        for p in list(broker.positions.keys()):
            sig = signals.get(p)
            if not sig:
                continue
            # NEVER let a directional signal flip close a hedge leg — a pair
            # hedge is market-neutral and managed as a unit by the hedger; a
            # confident directional call on one leg would orphan the other.
            if broker.positions[p].get("hedge"):
                continue
            side = broker.positions[p].get("side", 1)
            if sig["direction"] * side < 0 and sig["confidence"] > 0.5:
                t = broker.sell(p, market.price(p), "signal flip")
                if t:
                    risk.on_trade_closed(t)
                    learner.on_trade_closed(t)

        # 7. entries — conviction trades at full size
        if self.running:
            ranked = sorted((s for s in signals.values() if s["actionable"]),
                            key=lambda s: -s["confidence"])
            for sig in ranked:
                p = sig["product"]
                ok, why = risk.can_open(p, broker, market, market.healthy)
                if not ok:
                    continue
                ok, why = risk.funding_gate(p, sig["direction"])
                if not ok:
                    continue
                notional, stop, take = risk.size(
                    equity, sig["price"], sig["atr"], sig["confidence"],
                    self.last_risk_status, direction=sig["direction"], product=p)
                if notional < tv("min_notional"):
                    continue
                # A cost-viable conviction candidate cleared the gate — this
                # interval WAS a real chance to trade, whether or not the fill
                # ultimately succeeds. That's what makes flat equity meaningful
                # (or not) to the RL agent.
                tradable = True
                pos = broker.open(p, sig["direction"], notional, sig["price"],
                                  stop, take,
                                  f"composite={sig['composite']} regime={sig['regime']}",
                                  votes=engine.per_strategy.get(p, {}),
                                  regime_at_entry=sig["regime"])
                if pos is not None:
                    # rich open notification (Telegram/webhook) with full detail
                    try:
                        from .alerts import notify_trade_open
                        notify_trade_open(pos)
                    except Exception:
                        pass
                    # mirror into the shadow OMS to measure execution divergence
                    if self.shadow is not None:
                        try:
                            self.shadow.mirror_open(p, sig["direction"], notional,
                                                    pos["entry"])
                        except Exception as e:
                            db.log_event("error", f"shadow mirror failed: {e}")

            # 7b. exploration entries — small probing positions on moderate
            # signals so the learning stack earns real trade experience.
            # Skipped when the RL agent has chosen to SIT OUT: that is exactly
            # what a sit-out means — no discretionary probes / extra names —
            # WITHOUT ever zeroing a cost-viable conviction trade above.
            if (not self.last_risk_status.get("rl_sit_out")
                    and random.random() < tv("explore_prob") * stance.current()["explore_mult"]):
                explorable = sorted(
                    (s for s in signals.values() if s.get("explorable")),
                    key=lambda s: -s["confidence"])
                for sig in explorable[:1]:            # at most one probe/tick
                    p = sig["product"]
                    ok, why = risk.can_open(p, broker, market, market.healthy)
                    if not ok:
                        continue
                    ok, why = risk.funding_gate(p, sig["direction"])
                    if not ok:
                        continue
                    notional, stop, take = risk.size(
                        equity, sig["price"], sig["atr"], sig["confidence"],
                        self.last_risk_status, direction=sig["direction"], product=p)
                    notional *= tv("explore_size_factor")
                    if notional < tv("min_notional"):
                        continue
                    tradable = True
                    pos = broker.open(p, sig["direction"], notional,
                                      sig["price"], stop, take,
                                      f"EXPLORE composite={sig['composite']} "
                                      f"regime={sig['regime']}",
                                      votes=engine.per_strategy.get(p, {}),
                                      regime_at_entry=sig["regime"])
                    if pos is not None:
                        try:
                            from .alerts import notify_trade_open
                            notify_trade_open(pos)
                        except Exception:
                            pass

        # Tell the RL agent whether THIS interval was a genuine chance to trade,
        # so next tick's learning update only fires when flat/negative equity
        # actually reflects a decision (not a fee-gated locked door).
        risk._tradable_next = tradable

        # 8. equity log + periodic self-improvement
        db.log_equity(equity, broker.cash, broker.exposure(market))
        if self.tick_count % 9 == 0:    # every ~3 min
            learner.run(market, regime)

        # 9. periodic state snapshot (survives crashes/restarts) — run OFF the
        # event loop so a large JSON serialize can't stall stop management.
        if self.tick_count % 3 == 0:    # every ~1 min
            from . import persistence
            try:
                asyncio.get_running_loop().run_in_executor(None, persistence.save)
            except RuntimeError:
                persistence.save()

    def snapshot(self):
        eq = broker.equity(market)
        return {
            "ts": time.time(),
            "uptime_sec": int(time.time() - self.started),
            "trading_enabled": self.running,
            "kill_switch": risk.killed,
            "kill_reason": risk.kill_reason,
            "kill_ts": risk.kill_ts,
            "halted_today": risk.halted_today,
            "halt_reason": risk.halt_reason,
            "data_healthy": market.healthy,
            "research_healthy": research.healthy,
            "equity": round(eq, 2),
            "cash": round(broker.cash, 2),
            "exposure": round(broker.exposure(market), 2),
            "realized_pnl": round(broker.realized_pnl, 2),
            # unrealized = sum of per-position mark-to-market vs entry, correct
            # for BOTH longs and shorts (was a convoluted, short-wrong expr).
            "unrealized_pnl": round(sum(
                pos.get("side", 1) * ((market.price(p) or pos["entry"]) - pos["entry"])
                * pos["qty"] for p, pos in broker.positions.items()), 2),
            "risk": self.last_risk_status,
            "stance": stance.current(),
            "hedge": hedger.snapshot(),
            "macro_blackout": _calendar_stats(),
            "regime": market.regime(),
            "fear_greed": research.fear_greed,
            "market_sentiment": round(nlp.market_sentiment, 3),
            "strategy_weights": learner.weights,
            "positions": [
                {**{k: pos[k] for k in ("product", "qty", "entry", "stop", "take")},
                 "side": pos.get("side", 1),
                 "price": market.price(p),
                 "unrealized": round(pos.get("side", 1) *
                                     ((market.price(p) or pos["entry"]) - pos["entry"])
                                     * pos["qty"], 2)}
                for p, pos in broker.positions.items()],
            "stats": broker.stats(),
        }


orch = Orchestrator()
