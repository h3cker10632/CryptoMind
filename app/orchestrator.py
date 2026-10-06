"""Orchestrator / Control Plane — schedules all engines, runs the decision
loop, enforces risk gates, and drives the self-improvement cycle."""
import asyncio, os, time, random

# the online model is warm-started from history until it has this many updates
WARM_START_MIN_UPDATES = 2000
from .config import TICK_SEC
from .tunables import tv
from . import db
from . import settings
from .data.market import market
from .data.research import research
from .nlp.sentiment import nlp
from .signals.engine import engine
from .execution.paper import broker
from .risk.manager import risk
from .learn.loop import learner
from .risk.stance import stance
from .strategies.hedge import hedger
from .guardian import guardian
from .portfolio import portfolio as paper_portfolio


def _calendar_stats():
    from .data.calendar import calendar
    return calendar.stats()


class Orchestrator:
    def __init__(self):
        self.running = not settings.get("trading_paused")
        self.tick_count = 0
        self.last_risk_status = {}
        self.started = time.time()
        self._blackout_logged = False
        self.shadow = None       # ShadowBroker, set by main.py
        self.last_replay = None  # latest automatic replay report (replay_loop)
        # resting LIMIT entry orders (entry_order_type = "maker"):
        # product -> {limit, direction, notional, stop, take, sig, ts, ...}
        self.pending_entries = {}
        # chop filter: auto-mode decision (set from the replay A/B) and whether
        # it is blocking entries right now (for state-change logging)
        self.chop_auto_on = False
        self._chop_blocking = False

    async def decision_loop(self):
        await asyncio.sleep(10)  # let feeds warm up
        # LAUNCH PREFLIGHT — verify preconditions before the loop is trusted to
        # open positions. A non-pass keeps the guardian in safe mode (entries
        # held, open risk still managed) until health is genuinely restored.
        try:
            guardian.preflight()
        except Exception as e:
            db.log_event("error", f"Preflight failed to run: {e}")
        # LAUNCH PREFLIGHT — verify preconditions before the loop is trusted to
        # open positions. A non-pass keeps the guardian in safe mode (entries
        # held, open risk still managed) until health is genuinely restored.
        try:
            guardian.preflight()
        except Exception as e:
            db.log_event("error", f"Preflight failed to run: {e}")
        consecutive_failures = 0
        while True:
            try:
                self.tick()
                self.last_tick_ts = time.time()
                consecutive_failures = 0
                guardian.on_cycle(True, market=market)
            except Exception as e:
                consecutive_failures += 1
                import traceback
                db.log_event("error",
                             f"Orchestrator tick failed: {e}\n"
                             f"{traceback.format_exc()}")
                guardian.on_cycle(False, market=market, error=str(e))
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

    async def llm_advisor_loop(self):
        """Refresh the optional LLM advisor's lean cache OFF the hot decision
        path. No-ops entirely (and cheaply) unless the advisor is enabled and a
        key is configured; a failure here can never affect trading."""
        from .learn.llm_advisor import advisor
        from .data.research import research
        from .config import PRODUCTS
        await asyncio.sleep(30)
        while True:
            try:
                if advisor.configured():
                    await advisor.refresh(market, nlp, list(PRODUCTS))
            except Exception as e:
                db.log_event("warn", f"LLM advisor loop error: {e}")
            await asyncio.sleep(30)

    async def llm_advisor_loop(self):
        """Refresh the optional LLM advisor's lean cache OFF the hot decision
        path. No-ops entirely (and cheaply) unless the advisor is enabled and a
        key is configured; a failure here can never affect trading."""
        from .learn.llm_advisor import advisor
        from .data.research import research
        from .config import PRODUCTS
        await asyncio.sleep(30)
        while True:
            try:
                if advisor.configured():
                    await advisor.refresh(market, nlp, list(PRODUCTS))
            except Exception as e:
                db.log_event("warn", f"LLM advisor loop error: {e}")
            await asyncio.sleep(30)

    async def model_advisor_loop(self):
        """Refresh the optional ML-model advisor's lean cache OFF the hot
        decision path. No-ops entirely (and cheaply) unless the advisor is
        enabled and a validated crypto_ml_lab artifact is loadable; a failure
        here can never affect trading."""
        from .learn.model_advisor import advisor as model_advisor
        from .config import PRODUCTS
        await asyncio.sleep(35)
        while True:
            try:
                if model_advisor.configured():
                    await model_advisor.refresh(market, list(PRODUCTS))
            except Exception as e:
                db.log_event("warn", f"Model advisor loop error: {e}")
            await asyncio.sleep(30)

    async def ml_trainer_loop(self):
        """Autonomous, metric-gated ML retraining. Checks the data-driven
        trigger on a cadence and, when enough NEW labeled rows have matured,
        runs the full crypto_ml_lab pipeline off the hot path (in a worker
        thread) and auto-promotes the artifact ONLY if the backtest gates pass.
        No-ops entirely unless `ml_autotrain_enabled` is set; a failure here can
        never affect trading."""
        from .learn.ml_trainer import trainer
        from . import settings
        await asyncio.sleep(90)  # let feeds + db warm up
        while True:
            try:
                if trainer.enabled():
                    ok, reason = trainer.should_run()
                    if ok:
                        db.log_event("learn", "Auto-trainer: enough new labels "
                                     "matured — running crypto_ml_lab pipeline")
                        rep = await asyncio.to_thread(trainer.run_pipeline, False)
                        if rep.get("promoted"):
                            db.log_event("learn", "Auto-trainer PROMOTED a new "
                                         f"model (gates passed): {rep.get('metrics')}")
                        elif rep.get("started"):
                            db.log_event("learn", "Auto-trainer ran but did not "
                                         f"promote: {rep.get('reason')}")
            except Exception as e:
                db.log_event("warn", f"ML trainer loop error: {e}")
            try:
                interval = max(60, int(settings.get("ml_autotrain_check_sec")))
            except Exception:
                interval = 3600
            await asyncio.sleep(interval)

    async def researcher_loop(self):
        """Autonomous strategy DISCOVERY. On a cadence, searches the safe rule DSL
        for new strategy shapes, validates each on a purged walk-forward +
        deflated-Sharpe gate that prices the multiple testing, and auto-promotes
        passers into the bandit-weighted `discovered` ensemble arm. No-ops unless
        `researcher_enabled` is set; runs in a worker thread off the hot path and
        can never affect trading (a promoted arm still earns its weight from
        realized PnL like every other sleeve)."""
        from .learn.researcher import researcher
        from . import settings
        await asyncio.sleep(150)     # let feeds + history caches warm up
        while True:
            try:
                if settings.get("researcher_enabled"):
                    rep = await asyncio.to_thread(researcher.run_universe)
                    if rep.get("promoted_total"):
                        db.log_event("learn", "Strategy Researcher promoted "
                                     f"{rep['promoted_total']} discovered "
                                     "strategy(ies) into the ensemble")
            except Exception as e:
                db.log_event("warn", f"Researcher loop error: {e}")
            await asyncio.sleep(6 * 3600)     # discovery every ~6h

    # ------------------------------------------------ optional core holding
    async def core_loop(self):
        """Hourly: keep the optional core holding on target (no-op while
        `core_allocation_pct` is 0 and nothing is held)."""
        from .strategies.core import core
        from .backtest.engine import fetch_history
        await asyncio.sleep(180)
        while True:
            try:
                if (core.enabled() or core.positions) and market.healthy:
                    _, assets, use_filter = core._settings()
                    daily = {}
                    if use_filter:
                        for a in set(assets) | set(core.positions):
                            try:
                                daily[a] = await fetch_history(a, granularity=86400,
                                                               chunks=1, ttl=3600)
                            except Exception:
                                pass
                    if daily and core.follows_champion():
                        # keep the store fresh with the newest CLOSED days
                        from .data import store
                        for a, cs in daily.items():
                            try:
                                store.ingest_candles(a, 86400, cs, source="core-loop")
                            except Exception as e:
                                db.log_event("warn", f"store ingest {a} failed: {e}")
                    actions = core.rebalance(broker, market,
                                             self._total_equity(market), daily)
                    if actions:
                        from .alerts import alert
                        msg = "; ".join(actions)
                        db.log_event("trade", f"CORE holding: {msg}")
                        alert("info", "Core holding rebalanced", msg)
            except Exception as e:
                db.log_event("error", f"core loop failed: {e}")
            await asyncio.sleep(3600)

    @staticmethod
    def _exploration_snapshot():
        from .strategies.exploration import manager as explore
        try:
            return explore.snapshot(market, broker) if explore.enabled() else {}
        except Exception:
            return {}

    async def exploration_loop(self):
        """Exploration sleeve (app/strategies/exploration.py): fund once, keep
        each engine member on its latest closed bar (hourly members hourly,
        daily members daily), mark NAVs hourly, bench / reinstate / reallocate
        once a day. Data for the members is refreshed into the store first."""
        from .strategies.exploration import manager as explore, MEMBERS, MAJORS
        from .backtest.engine import fetch_history
        from .data import store
        await asyncio.sleep(200)
        last_mark = 0.0
        while True:
            try:
                if explore.enabled() and market.healthy:
                    for gran in (3600, 86400):
                        for a in MAJORS:
                            try:
                                cs = await fetch_history(a, granularity=gran, chunks=1,
                                                         ttl=600)
                                store.ingest_candles(a, gran, cs, source="explore-loop")
                            except Exception as e:
                                db.log_event("warn", f"explore data {a} {gran}s: {e}")
                    eq = self._total_equity(market)
                    if explore.fund(eq, market, broker):
                        db.log_event("trade", f"EXPLORATION sleeve funded with "
                                     f"{len(MEMBERS)} strategies")
                    acts = []
                    for name, cfg in MEMBERS.items():
                        if cfg["kind"] == "engine":
                            acts += await asyncio.to_thread(explore.step, name, broker, market)
                    events = explore.review(eq, market, broker,
                                            replay_verdict=(self.last_replay or {}).get("verdict"))
                    if time.time() - last_mark >= 3600:
                        for name in MEMBERS:
                            explore._mark(name, market, time.time(), broker)
                        last_mark = time.time()
                        explore.save()
                    elif acts:
                        explore.save()
                    if acts:
                        db.log_event("trade", "EXPLORATION: " + "; ".join(acts))
                    for ev in events:
                        from .alerts import alert
                        db.log_event("learn", f"EXPLORATION {ev}")
                        alert("info", "Exploration sleeve", ev)
            except Exception as e:
                db.log_event("error", f"exploration loop failed: {e}")
            await asyncio.sleep(300)

    async def _maybe_warm_start(self, candles):
        """Pre-train the online model on history once (while it is fresh)."""
        from .learn.online_model import model as online
        if online.n_updates >= WARM_START_MIN_UPDATES or not candles:
            return 0
        from .learn.pretrain import pretrain_online
        before = online.n_updates
        n = await asyncio.to_thread(pretrain_online, candles)
        if n:
            st = online.stats()
            acc = st.get("directional_accuracy")
            msg = (f"Online model pre-trained on {n:,} historical samples "
                   f"(updates {before:,} -> {online.n_updates:,}; one-step-ahead "
                   f"accuracy on tradeable moves: "
                   f"{'n/a' if acc is None else format(acc, '.1%')}).")
            db.log_event("learn", msg)
            from .alerts import alert
            alert("info", "ML warm start complete", msg)
        return n

    # ------------------------------------------------ learned trade filter
    def _trade_filter_ok(self, sig):
        from .learn.trade_filter import trade_filter, features
        if not trade_filter.active():
            return True, ""
        p = sig["product"]
        f = market.features(p)
        if not f:
            return True, ""
        x = features(f, sig["direction"], sig["confidence"],
                     market.features("BTC-USD"), market.trendiness(tv("chop_lookback_bars")))
        ok, prob = trade_filter.allows(x)
        if ok:
            return True, ""
        return False, (f"trade filter: P(net win)={prob:.2f} < "
                       f"{trade_filter.threshold():.2f}")

    def _apply_filter_auto(self, rep, announce=True):
        from .learn.trade_filter import trade_filter
        ab = (rep or {}).get("filter_ab") or {}
        new = bool(ab.get("helps_in_both_halves"))
        was = trade_filter.auto_on
        trade_filter.auto_on = new
        trade_filter.last_eval = ab or None
        if announce and new != was and settings.get("trade_filter_mode") == "auto":
            from .alerts import alert
            msg = f"Walk-forward replay A/B — filter off: {ab.get('off')}, on: {ab.get('on')}."
            alert("info", "Trade filter " + ("ENABLED (auto)" if new else "DISABLED (auto)"), msg)
            db.log_event("learn", f"Trade filter auto -> {'ON' if new else 'OFF'}. {msg}")

    # ------------------------------------------------ chop filter
    def chop_filter_active(self):
        mode = settings.get("chop_filter_mode")
        return mode == "on" or (mode == "auto" and self.chop_auto_on)

    def _chop_blocks_entries(self):
        """True while the chop filter is active AND the whole market is no
        more directional than noise. Logs only on state changes."""
        block = False
        if self.chop_filter_active():
            tr = market.trendiness(tv("chop_lookback_bars"))
            block = tr is not None and tr < tv("chop_er_min")
            if block != self._chop_blocking:
                db.log_event("risk", ("⏸ CHOP FILTER: market trendiness "
                                      f"{tr:.3f} < {tv('chop_er_min'):.2f} — new "
                                      "entries paused") if block else
                             "Chop filter: market trending again — entries resumed")
        self._chop_blocking = block
        return block

    def _apply_chop_auto(self, rep, announce=True):
        """Auto mode: the filter is ON only while the latest replay shows it
        beating no-filter in BOTH halves of the history window."""
        ab = (rep or {}).get("chop_ab") or {}
        new = bool(ab.get("helps_in_both_halves"))
        was = self.chop_auto_on
        self.chop_auto_on = new
        if announce and new != was and settings.get("chop_filter_mode") == "auto":
            from .alerts import alert
            msg = (f"Replay A/B — filter off: {ab.get('off')}, on: {ab.get('on')}.")
            alert("info", "Chop filter " + ("ENABLED (auto)" if new else "DISABLED (auto)"), msg)
            db.log_event("risk", f"Chop filter auto -> {'ON' if new else 'OFF'}. {msg}")

    # ------------------------------------------------ limit (maker) entries
    def _max_positions(self):
        return max(1, round(tv("max_open_positions") * stance.current()["max_pos_mult"]))

    def _place_limit_entry(self, sig, notional, stop, take):
        """Rest a post-only LIMIT order at the signal price. A working order
        holds a position slot (so the book can't over-commit) and counts
        against the guardian's entries-per-hour cap when placed."""
        p = sig["product"]
        if len(broker.positions) + len(self.pending_entries) >= self._max_positions():
            self._audit(sig, "skip", "position slots held by working limit orders",
                        size_pre=notional)
            return False
        self.pending_entries[p] = {
            "limit": sig["price"], "direction": sig["direction"],
            "notional": notional, "stop": stop, "take": take,
            "ts": time.time(), "sig": dict(sig),
            "votes": dict(engine.per_strategy.get(p, {})),
        }
        guardian.note_entry()
        self._audit(sig, "limit", f"limit {'buy' if sig['direction'] > 0 else 'sell'} "
                                  f"placed @ {sig['price']:.6g}", size_pre=notional)
        return True

    def _process_pending_entries(self, market):
        """Fill limits that price traded THROUGH (strictly beyond the limit —
        a mere touch is not assumed to fill, which keeps the simulation from
        being flattered by queue position), cancel expired ones, and cancel
        everything when new entries are disabled."""
        if not self.pending_entries:
            return
        if not self._entries_enabled():
            for p, o in list(self.pending_entries.items()):
                self._audit(o["sig"], "skip", "limit cancelled: entries disabled")
            self.pending_entries.clear()
            return
        now = time.time()
        for p, o in list(self.pending_entries.items()):
            px = market.price(p)
            crossed = px is not None and (
                (o["direction"] > 0 and px < o["limit"]) or
                (o["direction"] < 0 and px > o["limit"]))
            if crossed:
                del self.pending_entries[p]
                if p in broker.positions:
                    continue
                sig = o["sig"]
                pos = broker.open(p, o["direction"], o["notional"], o["limit"],
                                  o["stop"], o["take"],
                                  f"LIMIT composite={sig['composite']} regime={sig['regime']}",
                                  votes=o["votes"], regime_at_entry=sig["regime"],
                                  atr_at_entry=sig.get("atr"),
                                  mtf_at_entry=sig.get("mtf_align"), maker=True)
                self._audit(sig, "enter" if pos is not None else "reject",
                            "limit filled" if pos is not None else "broker rejected limit fill",
                            size_pre=o["notional"],
                            size_post=(pos["qty"] * pos["entry"]) if pos else 0.0)
                if pos is not None:
                    try:
                        from .alerts import notify_trade_open
                        notify_trade_open(pos)
                    except Exception:
                        pass
                    if self.shadow is not None:
                        try:
                            self.shadow.mirror_open(p, o["direction"], o["notional"],
                                                    pos["entry"])
                        except Exception as e:
                            db.log_event("error", f"shadow mirror failed: {e}")
            elif now - o["ts"] > tv("maker_timeout_sec"):
                del self.pending_entries[p]
                self._audit(o["sig"], "skip", "limit not filled before timeout — cancelled")

    # ------------------------------------------------ automatic replay
    REPLAY_CHECK_SEC = 600          # how often to look for new universe members
    REPLAY_MIN_BARS = 150           # hourly bars a coin needs to be judged

    async def replay_loop(self):
        """Automatic strategy replay over the WHOLE live universe.

        Re-backtests the live strategy (app/backtest/replay.py) on fresh hourly
        history for every coin currently in config.PRODUCTS:
          * on a schedule (`replay_interval_sec`, default daily), and
          * as soon as a coin joins the universe that the last replay did not
            include — the alert then reports that coin's own replayed trades.
        Runs in a worker thread; informational only, never changes trading."""
        from .config import PRODUCTS
        from .strategies.core import core
        self._load_last_replay()
        self._apply_replay_brake(self.last_replay, announce=False)  # survive restarts
        self._apply_chop_auto(self.last_replay, announce=False)
        self._apply_filter_auto(self.last_replay, announce=False)
        await asyncio.sleep(240)        # let the universe + feeds settle
        while True:
            try:
                if settings.get("replay_enabled"):
                    self._ensure_forward_baseline()
                    self._maybe_component_audit()
                    due, new = self._replay_due(sorted(PRODUCTS))
                    from .learn.trade_filter import trade_filter
                    from .learn.online_model import model as _online
                    # the trade filter + online-model warm start are trained by
                    # the replay, so the first loop after a restart always runs
                    needs_models = ((trade_filter.model is None and
                                     settings.get("trade_filter_mode") != "off")
                                    or _online.n_updates < WARM_START_MIN_UPDATES)
                    if due or new or needs_models:
                        await self.run_replay_now(
                            reason=("new coin(s) joined the universe" if new
                                    else "scheduled"), new=new)
                    await self._maybe_sync_data()
                    await self._maybe_run_research_loop()
                    if self.hourly_bot_active():
                        # these only inform the hourly bot / legacy core modes
                        await self._maybe_run_learner_ablation()
                        self._apply_learner_gate()
                    if not core.follows_champion():
                        await self._maybe_run_daily_lab()
            except Exception as e:
                db.log_event("warn", f"Replay loop error: {e}")
            await asyncio.sleep(self.REPLAY_CHECK_SEC)

    async def _run_tool_weekly(self, script, label, last_ran_at, interval_key, now=None):
        """Run tools/<script> when `interval_key` seconds have passed since
        `last_ran_at`. A separate low-priority PROCESS: these are minutes of
        CPU, and a thread would hold the GIL against stop management."""
        import os, sys
        try:
            interval = int(settings.get(interval_key))
        except Exception:
            interval = 604800
        if (now or time.time()) - (last_ran_at or 0) < interval:
            return False
        procs = self.__dict__.setdefault("_tool_procs", {})
        if procs.get(script) is not None:
            return False
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kw = {}
        if os.name == "nt":
            import subprocess
            kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        db.log_event("learn", f"{label} started")
        procs[script] = await asyncio.create_subprocess_exec(
            sys.executable, os.path.join(root, "tools", script),
            cwd=root, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL, **kw)
        try:
            code = await procs[script].wait()
        finally:
            procs[script] = None
        db.log_event("learn" if code == 0 else "warn", f"{label} finished (exit {code})")
        return code == 0

    async def _maybe_sync_data(self, now=None):
        """Daily: bring the versioned data store (app/data/store.py) current —
        candles for every USD coin incl. delisted, hourly universe, funding."""
        import os
        from .data import store
        try:
            last = os.path.getmtime(os.path.join(store.STORE, "ingest_log.jsonl"))
        except OSError:
            last = 0
        return await self._run_tool_weekly("data_sync.py", "Market data sync", last,
                                           "data_sync_interval_sec", now)

    async def _maybe_run_learner_ablation(self, now=None):
        """Weekly learner ablation on the hourly history the replay just
        cached; its report drives the learner gate (app/learn/gate.py)."""
        from .learn import gate
        return await self._run_tool_weekly(
            "learner_ablation.py", "Learner ablation (evidence for the learner gate)",
            (gate._report() or {}).get("ran_at"), "learner_ablation_interval_sec", now)

    async def _maybe_run_research_loop(self, now=None):
        """Weekly champion / challenger loop (app/engine/challengers.py):
        backtest + forward-track every candidate; promote only on evidence.
        Alerts when the champion changes."""
        from .engine import challengers as C
        before = C.champion()[0]
        last = (C._load(C.STATE, {}).get("last_report") or {}).get("ran_at")
        ok = await self._run_tool_weekly("research_loop.py",
                                         "Research loop (champion / challengers)",
                                         last, "research_loop_interval_sec", now)
        after = C.champion()[0]
        if ok and after != before:
            from .alerts import alert
            rep = (C._load(C.STATE, {}).get("last_report") or {}).get("candidates", {})
            fwd = (rep.get(after) or {}).get("forward_sharpe_vs_champion")
            msg = (f"{before} -> {after}: beat the champion in both backtest halves "
                   f"and over its forward-tracking period (forward Sharpe {fwd}).")
            alert("info", "New champion strategy", msg)
            db.log_event("learn", f"Champion promoted: {msg}")
        return ok

    async def _maybe_run_daily_lab(self, now=None):
        """Weekly daily lab (multi-year daily history): re-tests core selection
        and sizing and retrains the rank model; the core's "auto" modes follow
        its decision. Announces when that decision changes."""
        from .strategies import core as core_mod
        before = core_mod.CoreBook.modes()[:2]       # also loads the last report
        rep = core_mod._LAB["report"]
        ok = await self._run_tool_weekly(
            "daily_lab.py", "Daily lab (core selection/sizing + rank model)",
            (rep or {}).get("ran_at"), "daily_lab_interval_sec", now)
        after = core_mod.CoreBook.modes()
        if ok and after[:2] != before:
            from .alerts import alert
            msg = f"{'+'.join(before)} -> {'+'.join(after[:2])}: {after[2]}"
            alert("info", "Core holding strategy changed", msg)
            db.log_event("learn", f"Core strategy changed: {msg}")
        return ok

    def _apply_learner_gate(self):
        """Announce learners the evidence gate switched on/off; a learner that
        just switched on is warm-started from its replay-trained state."""
        from .learn import gate
        prev = self.__dict__.setdefault("_gate_prev", {})
        for name, st in gate.status().items():
            was = prev.get(name)
            prev[name] = st["active"]
            if was is None:                 # first look after a restart: quiet
                if st["active"]:
                    gate.warm_start(name)
                continue
            if was == st["active"]:
                continue
            warmed = gate.warm_start(name) if st["active"] else False
            from .alerts import alert
            title = f"Learner {name} " + ("ENABLED" if st["active"] else "DISABLED") +                     f" ({st['mode']})"
            msg = st["why"] + ("; warm-started from replay" if warmed else "")
            alert("info", title, msg)
            db.log_event("learn", f"{title}: {msg}")

    def _replay_due(self, universe, now=None):
        """(scheduled run due?, [coins in the universe the last replay lacked]).
        A first-ever run counts as due, not as 'new coins'."""
        last = self.last_replay or {}
        covered = set(last.get("universe") or [])
        new = [p for p in universe if covered and p not in covered]
        try:
            interval = int(settings.get("replay_interval_sec"))
        except Exception:
            interval = 86400
        due = (now or time.time()) - last.get("ran_at", 0) >= interval
        return due, new

    async def run_replay_now(self, reason="manual", new=None):
        """Fetch hourly history for every universe coin, replay, record, alert.
        Serialized: a second call while one is running waits for it."""
        from .config import PRODUCTS
        from .backtest import replay as rp
        from .backtest.engine import fetch_history
        lock = self.__dict__.setdefault("_replay_lock", asyncio.Lock())
        async with lock:
            universe = sorted(PRODUCTS)
            candles, missing = {}, []
            try:
                days = int(settings.get("replay_history_days"))
            except Exception:
                days = 365
            chunks = max(3, -(-days * 24 // 300))       # 300 hourly bars per page
            for p in universe:
                try:
                    cs = await fetch_history(p, granularity=3600, chunks=chunks,
                                             ttl=6 * 3600)
                except Exception:
                    cs = None
                if cs:
                    candles[p] = cs
                else:
                    missing.append(p)
                await asyncio.sleep(0.2)            # be polite to the API
            chop_now = self.chop_filter_active()
            from .learn.trade_filter import trade_filter
            filt_now = trade_filter.active()
            rep = await asyncio.to_thread(
                lambda: rp.run_report(candles, chop=chop_now, chop_ab=True,
                                      filt_on=filt_now, filter_ab=True))
            samples = rep.pop("_filter_samples", None) or []
            if samples:
                await asyncio.to_thread(trade_filter.fit, samples)
            rep["trade_filter"] = {k: v for k, v in trade_filter.snapshot().items()
                                   if k != "last_eval"}
            await self._maybe_warm_start(candles)
            prev = self.last_replay
            rep.update(universe=universe, reason=reason,
                       new_products=list(new or []), missing_history=missing)
            self.last_replay = rep
            self._announce_replay(rep, prev)     # adds rep["forward_test"]
            self._save_replay(rep)
            self._apply_replay_brake(rep)
            self._apply_chop_auto(rep)
            self._apply_filter_auto(rep)
            return rep

    def _replay_path(self, name):
        import os
        from .export import REPORTS_DIR
        os.makedirs(REPORTS_DIR, exist_ok=True)
        return os.path.join(REPORTS_DIR, name)

    def _load_last_replay(self):
        import json
        try:
            with open(self._replay_path("replay_latest.json")) as f:
                self.last_replay = json.load(f)
        except Exception:
            self.last_replay = None

    def _save_replay(self, rep):
        import json
        try:
            with open(self._replay_path("replay_latest.json"), "w") as f:
                json.dump(rep, f, indent=1)
            full = rep.get("full") or {}
            line = {"ran_at": rep.get("ran_at"), "reason": rep.get("reason"),
                    "verdict": rep.get("verdict"),
                    "return_pct": full.get("return_pct"),
                    "max_drawdown_pct": full.get("max_drawdown_pct"),
                    "trades": full.get("trades"),
                    "first_half_pct": (rep.get("first_half") or {}).get("return_pct"),
                    "second_half_pct": (rep.get("second_half") or {}).get("return_pct"),
                    "buy_hold_pct": full.get("buy_hold_equal_weight_pct"),
                    "universe": rep.get("universe"),
                    "new_products": rep.get("new_products")}
            with open(self._replay_path("replay_history.jsonl"), "a") as f:
                f.write(json.dumps(line) + "\n")
        except Exception as e:
            db.log_event("warn", f"Replay save failed: {e}")

    def _announce_replay(self, rep, prev=None):
        from .alerts import alert
        full = rep.get("full") or {}
        if not full.get("ok"):
            db.log_event("backtest", f"Replay could not run: {full.get('error')}")
            return
        h1, h2 = rep.get("first_half") or {}, rep.get("second_half") or {}
        lines = [
            f"Universe: {len(rep.get('universe') or [])} coins, "
            f"{full['days']} days of hourly history ({rep.get('reason')}).",
            f"Return {full['return_pct']:+.2f}% (max DD {full['max_drawdown_pct']}%), "
            f"{full['trades']} trades, win {full['win_rate_pct']}%.",
            f"Halves: {h1.get('return_pct', 'n/a')}% / {h2.get('return_pct', 'n/a')}% "
            f"— {rep.get('verdict')}.",
            f"Equal-weight buy & hold: {full.get('buy_hold_equal_weight_pct')}%.",
            f"Entries: {full.get('entry_orders')}"
            + (f", {full.get('limit_fill_rate_pct')}% of limits filled"
               if full.get('limit_fill_rate_pct') is not None else "")
            + f"; chop filter {'on' if full.get('chop_filter') else 'off'}.",
        ]
        sb = rep.get("slow_benchmarks") or {}
        slow = [(k, v) for k, v in sb.items() if isinstance(v, dict) and "return_pct" in v]
        if slow:
            lines.append("Slow daily rules over the same history (after costs): " + "; ".join(
                f"{k.replace('_', ' ')} {v['return_pct']:+.1f}% "
                f"(halves {v['first_half_pct']:+.1f}% / {v['second_half_pct']:+.1f}%, "
                f"DD {v['max_drawdown_pct']}%)" for k, v in slow))
        fab = rep.get("filter_ab")
        if fab:
            lines.append(f"Trade filter A/B (halves, out-of-sample): off "
                         f"{fab['off'].get('first_half')}% / {fab['off'].get('second_half')}%, on "
                         f"{fab['on'].get('first_half')}% / {fab['on'].get('second_half')}% — "
                         + ("helps in both halves" if fab.get("helps_in_both_halves")
                            else "does not help in both halves"))
        ab = rep.get("chop_ab")
        if ab:
            lines.append(f"Chop filter A/B (halves): off {ab['off'].get('first_half')}% / "
                         f"{ab['off'].get('second_half')}%, on {ab['on'].get('first_half')}% / "
                         f"{ab['on'].get('second_half')}% — "
                         + ("helps in both halves" if ab.get("helps_in_both_halves")
                            else "does not help in both halves"))
        pp = rep.get("per_product") or {}
        for p in rep.get("new_products") or []:
            r = pp.get(p)
            if not r:
                lines.append(f"NEW {p}: no hourly history available yet.")
                continue
            short = (r.get("history_bars") or 0) < self.REPLAY_MIN_BARS
            lines.append(
                f"NEW {p}: {r['trades']} replayed trade(s), net ${r['net_usd']:+,.0f}"
                + (f", avg {r['avg_net_bps']:+.0f} bps" if r.get("avg_net_bps") is not None else "")
                + (f" — only {r['history_bars']} bars of history, too short to judge"
                   if short else ""))
        if rep.get("missing_history"):
            lines.append("No history for: " + ", ".join(rep["missing_history"]))
        try:
            ft = self.forward_test()
        except Exception:
            ft = None
        if ft and ft.get("return_pct") is not None:
            rep["forward_test"] = ft
            lines.append(f"LIVE paper since restart ({ft['days']} d): {ft['return_pct']:+.2f}% "
                         f"vs buy & hold {ft['buy_hold_pct']:+.2f}%.")
        drop = None
        pf = (prev or {}).get("full") or {}
        if pf.get("ok"):
            drop = full["return_pct"] - pf["return_pct"]
            lines.append(f"Change vs last replay: {drop:+.2f} pts.")
        msg = "\n".join(lines)
        db.log_event("backtest", "Strategy replay: " + " ".join(lines[:3]),
                     {k: rep.get(k) for k in ("verdict", "reason", "new_products")})
        level = ("warning" if full["return_pct"] < 0 or (drop is not None and drop < -5)
                 else "info")
        alert(level, "Strategy replay: " + str(rep.get("verdict")), msg)

    @staticmethod
    def _replay_brake_for(rep):
        """(multiplier, reason) the replay report implies for position risk.

        Brake ONLY when the strategy lost money in BOTH halves of the replay
        window over a meaningful number of trades — one bad half is regime luck,
        not evidence. Anything else (incl. a failed/absent replay, or the
        feature switched off) releases the brake."""
        if not settings.get("replay_brake_enabled") or not settings.get("replay_enabled"):
            return 1.0, ""
        rep = rep or {}
        f = rep.get("full") or {}
        h1, h2 = rep.get("first_half") or {}, rep.get("second_half") or {}
        if not (f.get("ok") and h1.get("ok") and h2.get("ok")):
            return 1.0, ""
        if f.get("trades", 0) < tv("replay_brake_min_trades"):
            return 1.0, ""
        if h1["return_pct"] < 0 and h2["return_pct"] < 0:
            return (tv("replay_brake_mult"),
                    f"replay negative in both halves ({h1['return_pct']:+.2f}% / "
                    f"{h2['return_pct']:+.2f}%, {f['trades']} trades)")
        return 1.0, ""

    def _apply_replay_brake(self, rep, announce=True):
        mult, why = self._replay_brake_for(rep)
        was = getattr(risk, "replay_brake", 1.0)
        risk.replay_brake, risk.replay_brake_reason = mult, why
        if not announce or abs(mult - was) < 1e-9:
            return
        from .alerts import alert
        if mult < 1.0:
            msg = (f"New positions now sized at {mult:.0%} of normal risk: {why}. "
                   f"Lifts automatically when a replay is no longer negative in "
                   f"both halves.")
            db.log_event("risk", "🧯 REPLAY BRAKE ON — " + msg)
            alert("warning", "Replay brake ON", msg)
        else:
            msg = "Strategy replay no longer negative in both halves — normal position risk restored."
            db.log_event("risk", "Replay brake OFF — " + msg)
            alert("info", "Replay brake OFF", msg)

    # ------------------------------------------------ forward test
    def _ensure_forward_baseline(self):
        """Record, once, the equity and every universe coin's price at the
        first start of this version. Every later report compares the LIVE paper
        result since then against simply holding those coins — the only test
        that can't be overfit, because it's all out-of-sample."""
        import json
        from .config import PRODUCTS
        path = self._replay_path("forward_test_baseline.json")
        if os.path.exists(path) or not market.tickers:
            return
        prices = {p: market.price(p) for p in PRODUCTS}
        prices = {p: v for p, v in prices.items() if v}
        if len(prices) < 3:
            return
        base = {"ts": time.time(), "equity": self._total_equity(market),
                "prices": prices, "learn_version": 2}
        try:
            with open(path, "w") as f:
                json.dump(base, f, indent=1)
            db.log_event("system", f"Forward test started: equity "
                         f"${base['equity']:,.0f}, {len(prices)} coins benchmarked")
        except Exception as e:
            db.log_event("warn", f"Forward-test baseline not saved: {e}")

    def _filter_snapshot(self):
        try:
            from .learn.trade_filter import trade_filter
            return trade_filter.snapshot()
        except Exception:
            return None

    def _core_snapshot(self):
        try:
            from .strategies.core import core
            return core.snapshot(market)
        except Exception:
            return None

    def _safe_forward_test(self):
        try:
            return self.forward_test()
        except Exception:
            return None

    def scorecard(self):
        """One honest view: what the core runs and why, how every candidate
        did in backtest AND forward, the live result vs simply holding BTC /
        BTC+ETH since the forward baseline, and which gates are on."""
        import json
        from .engine import challengers as C
        from .learn import gate
        from .strategies.core import core
        from .strategies.exploration import manager as explore
        from .data import store
        rep = (C._load(C.STATE, {}).get("last_report") or {})
        name, cfg = C.champion()
        cands = rep.get("candidates") or {}
        try:
            with open(self._replay_path("forward_test_baseline.json")) as f:
                base = json.load(f)
        except Exception:
            base = None

        def held_return(coins):
            if not base:
                return None
            rs = [market.price(c) / base["prices"][c] - 1 for c in coins
                  if base["prices"].get(c) and market.price(c)]
            return round(100 * sum(rs) / len(rs), 2) if rs else None
        fwd = self.forward_test() or {}
        try:
            last_sync = os.path.getmtime(os.path.join(store.STORE, "ingest_log.jsonl"))
        except OSError:
            last_sync = None
        return {
            "champion": {"name": name, "config": cfg,
                         "backtest": (cands.get(name) or {}).get("backtest"),
                         "forward_days": (cands.get(name) or {}).get("forward_days")},
            "candidates": {n: {"sharpe": (c.get("backtest") or {}).get("full", {}).get("sharpe"),
                               "halves": [(c.get("backtest") or {}).get(h, {}).get("sharpe")
                                          for h in ("first_half", "second_half")],
                               "cagr_pct": (c.get("backtest") or {}).get("cagr_pct"),
                               "max_dd_pct": (c.get("backtest") or {}).get("full", {}).get(
                                   "max_drawdown_pct"),
                               "deflated_sharpe": (c.get("backtest") or {}).get("deflated_sharpe"),
                               "forward_days": c.get("forward_days"),
                               "forward_sharpe_vs_champion": c.get("forward_sharpe_vs_champion"),
                               "eligible": c.get("eligible_for_promotion")}
                           for n, c in cands.items()},
            "trials_counted": rep.get("trials"),
            "research_ran_at": rep.get("ran_at"),
            "live": {"since_ts": fwd.get("since_ts"), "days": fwd.get("days"),
                     "account_return_pct": fwd.get("return_pct"),
                     "btc_hold_pct": held_return(["BTC-USD"]),
                     "btc_eth_hold_pct": held_return(["BTC-USD", "ETH-USD"])},
            "exploration": explore.snapshot(market, broker) if explore.enabled() else {},
            "core": {k: v for k, v in core.snapshot(market).items()
                     if k in ("enabled", "allocation_pct", "selection", "sizing", "value",
                              "positions", "trend")},
            "gates": {"hourly_bot_mode": settings.get("hourly_bot_mode"),
                      "hourly_bot_active": self.hourly_bot_active(),
                      "hourly_replay_verdict": (self.last_replay or {}).get("verdict"),
                      "learners": {n: s["active"] for n, s in gate.status().items()}},
            "data": {"store_last_sync": last_sync,
                     "store_age_hours": round((time.time() - last_sync) / 3600, 1)
                     if last_sync else None},
        }

    def forward_test(self):
        """Live paper return since the baseline vs equal-weight buy & hold of
        the baseline coins over the same time. None before a baseline exists."""
        import json
        try:
            with open(self._replay_path("forward_test_baseline.json")) as f:
                base = json.load(f)
        except Exception:
            return None
        eq = self._total_equity(market)
        bh = [market.price(p) / px0 - 1 for p, px0 in base["prices"].items()
              if px0 and market.price(p)]
        return {"since_ts": base["ts"], "days": round((time.time() - base["ts"]) / 86400, 1),
                "return_pct": round((eq / base["equity"] - 1) * 100, 2) if base["equity"] else None,
                "buy_hold_pct": round(100 * sum(bh) / len(bh), 2) if bh else None,
                "start_equity": round(base["equity"], 2), "equity": round(eq, 2)}

    # ------------------------------------------------ weekly component audit
    AUDIT_EVERY_SEC = 7 * 86400
    AUDIT_MIN_EVIDENCE = 50       # effective learner observations before judging

    def component_audit(self):
        """Which parts of the system are earning their keep, judged by the
        learner's NET-of-cost evidence (never switches anything off itself)."""
        from .signals.engine import STRATEGIES
        from .config import DISABLED_STRATEGIES
        rows, keep, cut, unproven = {}, [], [], []
        for name in STRATEGIES:
            if name in DISABLED_STRATEGIES:
                continue
            n, mean, _ = learner.bandit._strategy_pool(name)
            w = learner.weights.get(name, 0.0)
            rows[name] = {"weight": round(w, 4), "evidence_n": round(n, 1),
                          "net_score": round(mean * 1e4, 2)}
            if n < self.AUDIT_MIN_EVIDENCE:
                unproven.append(name)
            elif mean < 0:
                cut.append(name)
            else:
                keep.append(name)
        extras = {k: bool(settings.get(k)) for k in
                  ("llm_advisor_enabled", "model_advisor_enabled", "researcher_enabled",
                   "polymarket_enabled", "invo_enabled", "crawl4ai_producer_enabled",
                   "ml_autotrain_enabled")}
        return {"ts": time.time(), "sleeves": rows, "earning": keep,
                "candidates_to_switch_off": cut, "not_enough_evidence": unproven,
                "optional_components_enabled": extras}

    def _maybe_component_audit(self):
        import json
        path = self._replay_path("component_audit_latest.json")
        try:
            with open(path) as f:
                last = json.load(f).get("ts", 0)
        except Exception:
            last = 0
        if last == 0:
            # first run after the learning reset: start the clock instead of
            # reporting a week of "no evidence"
            try:
                with open(path, "w") as f:
                    json.dump({"ts": time.time(), "note": "audit clock started"}, f)
            except Exception:
                pass
            return None
        if time.time() - last < self.AUDIT_EVERY_SEC:
            return None
        a = self.component_audit()
        try:
            with open(path, "w") as f:
                json.dump(a, f, indent=1)
        except Exception:
            pass
        from .alerts import alert
        lines = [f"Earning (net of costs): {', '.join(a['earning']) or 'none yet'}",
                 f"Losing — consider switching off: {', '.join(a['candidates_to_switch_off']) or 'none'}",
                 f"Not enough evidence yet: {', '.join(a['not_enough_evidence']) or 'none'}"]
        on = [k.replace('_enabled', '') for k, v in a["optional_components_enabled"].items() if v]
        lines.append("Optional components running: " + (", ".join(on) or "none"))
        db.log_event("learn", "Weekly component audit: " + " | ".join(lines), a)
        alert("info", "Weekly component audit", "\n".join(lines))
        return a

    def replay_snapshot(self):
        rep = getattr(self, "last_replay", None)
        if not rep:
            return None
        full = rep.get("full") or {}
        return {"ran_at": rep.get("ran_at"), "reason": rep.get("reason"),
                "verdict": rep.get("verdict"), "return_pct": full.get("return_pct"),
                "max_drawdown_pct": full.get("max_drawdown_pct"),
                "trades": full.get("trades"),
                "buy_hold_pct": full.get("buy_hold_equal_weight_pct"),
                "universe_size": len(rep.get("universe") or []),
                "risk_brake": getattr(risk, "replay_brake", 1.0),
                "chop_filter_active": self.chop_filter_active(),
                "chop_filter_blocking": self._chop_blocking,
                "risk_brake_reason": getattr(risk, "replay_brake_reason", ""),
                "new_products": rep.get("new_products")}

    async def crawl_producer_loop(self):
        """Run the crawl4ai producer INSIDE the app on a cadence, so the operator
        never has to run the CLI. Crawls/scores configured or auto-discovered
        sources in a worker thread (off the hot path) and pushes rows into the
        ingest seam. No-ops unless `crawl4ai_producer_enabled` is set. The producer
        is imported LAZILY and every failure is swallowed, so a missing crawl4ai
        (Apache-2.0, optional) or a dead source can never affect trading — the core
        still never hard-imports the scraper."""
        from . import settings
        await asyncio.sleep(120)     # let feeds warm up
        while True:
            try:
                if settings.get("crawl4ai_producer_enabled"):
                    def _run():
                        from tools.crawl4ai_signal.producer import run_once
                        return run_once()
                    rep = await asyncio.to_thread(_run)
                    if rep.get("ok") and rep.get("rows"):
                        db.log_event("data", "crawl4ai producer pushed "
                                     f"{rep['rows']} web-signal row(s) "
                                     f"(scored {rep.get('scored', 0)}, "
                                     f"auto-discovered {rep.get('autodiscovered', 0)})")
            except Exception as e:
                db.log_event("warn", f"crawl4ai producer loop error: {e}")
            try:
                interval = max(60, int(settings.get("crawl4ai_interval_sec")))
            except Exception:
                interval = 900
            await asyncio.sleep(interval)

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
                    await asyncio.to_thread(db.prune_decisions)
                    await asyncio.to_thread(db.prune_signals)
                    await asyncio.to_thread(db.prune_events)
            except Exception as e:
                db.log_event("error", f"reconcile loop failed: {e}")
            await asyncio.sleep(60)

    def _audit(self, sig, action, reason, size_pre=0.0, size_post=0.0):
        """Write one decision-audit row (NOFX "no position without a paper
        trail"). Never let a logging failure break the decision loop."""
        try:
            db.log_decision(
                self.tick_count, sig["product"], sig["direction"],
                sig.get("composite", 0.0), sig.get("confidence", 0.0),
                sig.get("ml_confidence", 1.0), action, reason,
                size_pre=size_pre, size_post=size_post,
                regime=sig.get("regime"),
                votes=engine.per_strategy.get(sig["product"], {}))
        except Exception:
            pass

    def _record_skip(self, sig):
        """Phase 3: hand a gated-out ACTIONABLE conviction signal to the learner
        as a counterfactual. The learner scores what the trade would have made
        net of cost and teaches the bandit — so a signal we were throttled out
        of (cooldown, max-positions, per-hour cap, funding, min-notional) still
        produces a learning sample. Never let it break the decision loop."""
        try:
            learner.on_entry_skipped(
                sig["product"], sig["direction"], sig.get("price"),
                sig.get("regime"), engine.per_strategy.get(sig["product"], {}))
        except Exception:
            pass

    def _entries_enabled(self):
        """New entries pause while the shared paper-capital ledger migration is
        pending (see app.portfolio.bootstrap); existing positions are still
        managed (stops/take-profit/exits) regardless."""
        return self.running and paper_portfolio.ready and self.hourly_bot_active()

    def hourly_bot_active(self):
        """Evidence gate for the hourly bot's NEW entries (`hourly_bot_mode`):
        auto = only while the latest daily replay is positive in both halves."""
        from .strategies.exploration import manager as explore
        if explore.enabled():
            # inside the exploration sleeve the bench rule governs it
            return explore.is_active("hourly_bot")
        mode = settings.get("hourly_bot_mode")
        if mode in ("on", "off"):
            return mode == "on"
        return (self.last_replay or {}).get("verdict") == "positive in both halves"

    def _bot_equity(self, market):
        """Equity the TRADING BOT sizes against (see core.bot_equity)."""
        from .strategies.core import core
        return core.bot_equity(self._total_equity(market), market)

    def _total_equity(self, market):
        """Account-level equity: shared cash plus every sleeve's marked
        positions once the ledger migration is confirmed; crypto-only
        beforehand (matches pre-migration behavior exactly). Used for the
        account-level kill-switch/drawdown tracker, equity logging and
        reporting -- NOT for crypto-only gates like max_gross_exposure or
        meme caps, which stay scoped to the crypto book (see risk.can_open)."""
        from .strategies.core import core
        from .strategies.exploration import manager as explore
        books = core.value(market) + explore.value(market)
        if not paper_portfolio.ready:
            return broker.equity(market) + books
        from .markets.polymarket.engine import engine as pm_engine
        return paper_portfolio.total_equity(market, pm_engine._mid_lookup) + books

    def _total_exposure(self, market):
        """Account-level committed exposure across both sleeves (reporting /
        equity-curve logging only; crypto-only gates are unaffected)."""
        from .strategies.core import core
        from .strategies.exploration import manager as explore
        if not paper_portfolio.ready:
            return broker.exposure(market) + core.value(market) + explore.value(market)
        from .markets.polymarket.engine import engine as pm_engine
        # the core book is real exposure too (it was left out: the dashboard
        # showed 0 exposure with ~$99k in BTC/ETH)
        return (paper_portfolio.total_exposure(market, pm_engine._mid_lookup)
                + core.value(market) + explore.value(market))

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
        equity = self._total_equity(market)
        self.last_risk_status = risk.update(equity, regime)
        # the bot sizes against its own sleeve, never the optional core holding
        from .strategies.core import core as _core
        bot_equity = _core.bot_equity(equity, market)

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

        # PROTECTIONS: refresh the time-boxed circuit breakers from the recent
        # trade history (cheap tail scan) so a stop-out cluster, a chronic
        # losing coin, or a temporary drawdown benches the relevant pairs before
        # the entry loop below consults risk.can_open.
        try:
            newly_locked_before = risk.protections.stats()["global_locked"]
            risk.protections.evaluate(broker.closed_trades, equity=equity)
            if risk.protections.stats()["global_locked"] and not newly_locked_before:
                db.log_event("risk", "🛑 PROTECTION engaged (all pairs): "
                             + risk.protections.stats()["global_reason"])
        except Exception as e:
            db.log_event("error", f"protections evaluate failed: {e}")

        # 5b. PREDICTIVE LOSS-CUT — after the hard stops/targets, ask the exit
        # advisor whether any losing position is expected to keep going against
        # us; if so, cut it early. This runs AFTER broker.manage so the hard
        # stop/take-profit/liquidation always take precedence.
        try:
            self._manage_exit_advisor(market, regime, signals)
        except Exception as e:
            db.log_event("error", f"exit advisor failed: {e}")

        # 5c. PATTERN-AWARE EXIT — a confirmed reversal chart pattern forming
        # against an open position tightens its stop, or cuts it when strong.
        # Runs AFTER the hard stops and the loss-cut advisor, so it only ever
        # ADDS protection; it never loosens a stop or overrides a hard exit.
        try:
            self._manage_pattern_exit(market)
        except Exception as e:
            db.log_event("error", f"pattern exit failed: {e}")

        # 6. exits on signal flip
        self._manage_signal_flip_exits(market, regime, signals)

        # 6b. resting limit entries: fill the ones price traded through,
        # cancel the expired ones
        try:
            self._process_pending_entries(market)
        except Exception as e:
            db.log_event("error", f"limit-order processing failed: {e}")

        # 7. entries — conviction trades at full size
        if self._entries_enabled() and not self._chop_blocks_entries():
            ranked = sorted((s for s in signals.values() if s["actionable"]),
                            key=lambda s: -s["confidence"])
            for sig in ranked:
                p = sig["product"]
                if p in self.pending_entries:
                    continue                  # a limit order is already working
                # GUARDIAN gate — safe mode + per-hour entry cap, checked BEFORE
                # per-product risk gates. A deterministic "no new risk" veto
                # independent of the learner (NOFX "runtime disposes").
                gok, gwhy = guardian.can_enter()
                if not gok:
                    self._audit(sig, "skip", gwhy)
                    self._record_skip(sig)     # counterfactual (Phase 3)
                    continue
                ok, why = risk.can_open(p, broker, market, market.healthy)
                if not ok:
                    self._audit(sig, "skip", why)
                    self._record_skip(sig)     # counterfactual (Phase 3)
                    continue
                ok, why = risk.funding_gate(p, sig["direction"])
                if not ok:
                    self._audit(sig, "skip", why)
                    self._record_skip(sig)     # counterfactual (Phase 3)
                    continue
                notional, stop, take = risk.size(
                    bot_equity, sig["price"], sig["atr"], sig["confidence"],
                    self.last_risk_status, direction=sig["direction"], product=p,
                    ml_confidence=sig.get("ml_confidence", 1.0))
                if notional < tv("min_notional"):
                    self._audit(sig, "skip",
                                f"notional ${notional:,.0f} below min "
                                f"${tv('min_notional')}", size_pre=notional)
                    self._record_skip(sig)     # counterfactual (Phase 3)
                    continue
                # LEARNED TRADE FILTER (meta-label): skip signals the model rates
                # below the historical base win rate (only while active).
                ok, why = self._trade_filter_ok(sig)
                if not ok:
                    self._audit(sig, "skip", why, size_pre=notional)
                    self._record_skip(sig)     # counterfactual (Phase 3)
                    continue
                # A cost-viable conviction candidate cleared the gate — this
                # interval WAS a real chance to trade, whether or not the fill
                # ultimately succeeds. That's what makes flat equity meaningful
                # (or not) to the RL agent.
                tradable = True
                from .execution.costs import maker_entries
                if maker_entries():
                    self._place_limit_entry(sig, notional, stop, take)
                    continue
                pos = broker.open(p, sig["direction"], notional, sig["price"],
                                  stop, take,
                                  f"composite={sig['composite']} regime={sig['regime']}",
                                  votes=engine.per_strategy.get(p, {}),
                                  regime_at_entry=sig["regime"],
                                  atr_at_entry=sig.get("atr"),
                                  mtf_at_entry=sig.get("mtf_align"))
                self._audit(sig, "enter" if pos is not None else "reject",
                            "opened" if pos is not None else "broker rejected fill",
                            size_pre=notional,
                            size_post=(pos["qty"] * pos["entry"]) if pos else 0.0)
                if pos is not None:
                    guardian.note_entry()
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
                    gok, gwhy = guardian.can_enter()
                    if not gok:
                        continue
                    ok, why = risk.can_open(p, broker, market, market.healthy)
                    if not ok:
                        continue
                    ok, why = risk.funding_gate(p, sig["direction"])
                    if not ok:
                        continue
                    notional, stop, take = risk.size(
                        bot_equity, sig["price"], sig["atr"], sig["confidence"],
                        self.last_risk_status, direction=sig["direction"], product=p,
                        ml_confidence=sig.get("ml_confidence", 1.0))
                    notional *= tv("explore_size_factor")
                    if notional < tv("min_notional"):
                        continue
                    tradable = True
                    pos = broker.open(p, sig["direction"], notional,
                                      sig["price"], stop, take,
                                      f"EXPLORE composite={sig['composite']} "
                                      f"regime={sig['regime']}",
                                      votes=engine.per_strategy.get(p, {}),
                                      regime_at_entry=sig["regime"],
                                      atr_at_entry=sig.get("atr"),
                                      mtf_at_entry=sig.get("mtf_align"))
                    self._audit(sig, "explore" if pos is not None else "reject",
                                "probe opened" if pos is not None else "broker rejected probe",
                                size_pre=notional,
                                size_post=(pos["qty"] * pos["entry"]) if pos else 0.0)
                    if pos is not None:
                        guardian.note_entry()
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
        if self.tick_count % 3 == 0:    # ~1/min is plenty for the equity curve
            db.log_equity(equity, broker.cash, self._total_exposure(market))
        # score the exit advisor's matured hold-vs-cut decisions against what
        # price actually did next (counterfactual learning), using the learner's
        # price history as the lookup.
        try:
            from .learn.exit_advisor import exit_advisor
            exit_advisor.score_pending(
                lambda prod, ts: learner._price_at(prod, ts, market))
        except Exception as e:
            db.log_event("error", f"exit advisor scoring failed: {e}")
        # score the exit THROTTLE's matured pattern_exit/signal-flip exits
        # against the realized counterfactual (same price-history lookup).
        try:
            from .learn.exit_throttle import exit_throttle
            exit_throttle.score_pending(
                lambda prod, ts: learner._price_at(prod, ts, market),
                horizon_sec=tv("exit_throttle_horizon_sec"))
        except Exception as e:
            db.log_event("error", f"exit throttle scoring failed: {e}")
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

    @staticmethod
    def _trend_bucket(side, regime):
        """Is this position WITH or AGAINST the current BTC-regime trend —
        the same bucket app.learn.exit_advisor uses, reused here so the exit
        throttle's hierarchical pooling lines up with proven state buckets."""
        trend_num = 1 if regime.get("trend") == "bull" else \
                    -1 if regime.get("trend") == "bear" else 0
        st = side * trend_num
        return "with" if st > 0 else "against" if st < 0 else "neutral"

    def _manage_signal_flip_exits(self, market, regime, signals):
        """Direction-aware exit: close a long on a confident bearish signal,
        close a short on a confident bullish one. The confidence bar is
        throttled by the learned, regime-conditioned edge of this mechanism
        (see app.learn.exit_throttle) — same scheme as pattern_exit; the hard
        stop/target are never touched by it."""
        from . import settings as app_settings
        from .learn.exit_throttle import exit_throttle
        throttle_on = app_settings.get("exit_throttle_enabled")
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
            # MIN-HOLD: don't let a signal flip churn a position out on the same
            # (or next) tick it opened — its hard stop/target still protect it.
            age = time.time() - broker.positions[p].get("opened", 0)
            if age < tv("min_hold_sec"):
                continue
            trend_bucket = self._trend_bucket(side, regime)
            factor = (exit_throttle.throttle_factor(
                        "signal_flip", regime["label"], trend_bucket)
                      if throttle_on else 1.0)
            conf_bar = min(0.99, 0.5 / factor)
            if sig["direction"] * side < 0 and sig["confidence"] > conf_bar:
                px = market.price(p)
                t = broker.sell(p, px, "signal flip")
                if t:
                    risk.on_trade_closed(t)
                    learner.on_trade_closed(t)
                    if throttle_on:
                        exit_throttle.record_exit("signal_flip", regime["label"],
                                                  trend_bucket, p, side, px)

    def _manage_pattern_exit(self, market):
        """Tighten or cut an open position when a confirmed reversal chart
        pattern forms against it. Additive protection only — layered on top of
        the hard stop / take-profit / trailing / loss-cut advisor.

        The cut/tighten trigger bar is throttled by the learned, regime-
        conditioned edge of pattern_exit itself (see app.learn.exit_throttle):
        a mechanism confirmed to cut into recoveries gets a harder bar; one
        confirmed to catch real reversals gets a little more rope. The hard
        stop/take-profit below this are never touched by that learning."""
        from .signals import patterns
        from . import settings as app_settings
        if not app_settings.get("pattern_exit_enabled"):
            return
        from .learn.exit_throttle import exit_throttle
        throttle_on = app_settings.get("exit_throttle_enabled")
        regime = market.regime()
        cut_th = tv("pattern_exit_cut")
        tighten_th = tv("pattern_exit_tighten")
        tighten_atr = tv("pattern_exit_tighten_atr")
        for p in list(broker.positions.keys()):
            pos = broker.positions[p]
            if pos.get("hedge"):
                continue
            if time.time() - pos.get("opened", 0) < tv("min_hold_sec"):
                continue
            px = market.price(p)
            f = market.features(p)
            if px is None or not f:
                continue
            rep = f.get("patterns")
            if not rep:
                continue
            side = pos.get("side", 1)
            trend_bucket = self._trend_bucket(side, regime)
            factor = (exit_throttle.throttle_factor(
                        "pattern_exit", regime["label"], trend_bucket)
                      if throttle_on else 1.0)
            eff_cut_th = cut_th / factor
            eff_tighten_th = tighten_th / factor
            threat, name = patterns.exit_threat(side, rep)
            if threat < eff_tighten_th:
                continue
            label = name or "reversal pattern"
            if threat >= eff_cut_th:
                t = broker.sell(p, px, f"pattern reversal ({label})")
                if t:
                    risk.on_trade_closed(t)
                    learner.on_trade_closed(t)
                    if throttle_on:
                        exit_throttle.record_exit("pattern_exit", regime["label"],
                                                  trend_bucket, p, side, px)
                    if self.shadow is not None and p in self.shadow.positions:
                        try:
                            self.shadow.mirror_close(p, t.get("exit", t["entry"]))
                        except Exception:
                            pass
                    db.log_event("risk", f"✂️ {p}: cut on {label} "
                                 f"(threat {threat:.2f} ≥ {eff_cut_th:.2f})")
                continue
            # tighten: pull the stop to `tighten_atr` swing-ATR from price, but
            # only ever CLOSER than the current stop (never loosen it).
            atr = f.get("atr_swing") or f.get("atr")
            if not atr:
                continue
            if side > 0:
                new_stop = px - tighten_atr * atr
                if new_stop > pos["stop"]:
                    pos["stop"] = new_stop
                    db.log_event("risk", f"⚠️ {p}: stop tightened to {new_stop:.4f} "
                                 f"on {label} (threat {threat:.2f})")
            else:
                new_stop = px + tighten_atr * atr
                if new_stop < pos["stop"]:
                    pos["stop"] = new_stop
                    db.log_event("risk", f"⚠️ {p}: stop tightened to {new_stop:.4f} "
                                 f"on {label} (threat {threat:.2f})")

    def _manage_exit_advisor(self, market, regime, signals):
        """Consult the self-learning exit advisor for every open (non-hedge)
        position and cut losers it confidently expects to keep bleeding.

        The advisor learns from the counterfactual next-horizon move, so each
        consultation is also RECORDED for scoring later. Nothing here overrides
        the hard stop/target (already applied in broker.manage above)."""
        from .learn.exit_advisor import exit_advisor
        from .learn.online_model import model, committee, build_x
        from .data.derivatives import derivatives
        from .nlp.sentiment import nlp
        if not exit_advisor._enabled():
            return
        for p in list(broker.positions.keys()):
            pos = broker.positions[p]
            if pos.get("hedge"):
                continue
            # respect the same min-hold that governs signal-flip exits
            if time.time() - pos.get("opened", 0) < tv("min_hold_sec"):
                continue
            px = market.price(p)
            f = market.features(p)
            if px is None or not f:
                continue
            side = pos.get("side", 1)
            entry = pos["entry"]
            unrealized_pct = side * (px / entry - 1)
            atr_pct = (f.get("atr_swing") or f.get("atr") or 0.0) / px if px else 0.0
            # model forward view, rotated into the POSITION frame
            ml_pos = 0.0
            ml_pos_hi = None
            if model.n_updates >= 40:
                acc = model.stats().get("directional_accuracy")
                if acc is not None and acc > 0.50:
                    asset_sent, _ = nlp.asset_score(p)
                    x = build_x(f, asset_sent, nlp.market_sentiment,
                                derivatives.features(p))
                    u = committee.predict_with_uncertainty(x)
                    ml_pos = side * u["mean"] * 0.004    # scale units -> ~return
                    # rotate the CALIBRATED interval into the position frame and
                    # take its optimistic (upper) edge; only trust it once the
                    # conformal calibrator actually carries a coverage guarantee.
                    if u.get("calibrated"):
                        lo = side * u["lo"] * 0.004
                        hi = side * u["hi"] * 0.004
                        ml_pos_hi = max(lo, hi)         # side<0 flips the ends
            # record context for counterfactual learning, then decide (both use
            # the SAME position-frame ml view so the learned state matches)
            exit_advisor.record(p, side, px, unrealized_pct, ml_pos,
                                regime, atr_pct)
            action, reason, _ = exit_advisor.decide(
                p, side, unrealized_pct, ml_pos, regime, atr_pct,
                ml_pos_hi=ml_pos_hi)
            if action == "cut":
                t = broker.sell(p, px, f"exit-advisor cut: {reason}")
                if t:
                    exit_advisor.note_cut()
                    risk.on_trade_closed(t)
                    learner.on_trade_closed(t)
                    db.log_event("learn", f"✂️ Loss-cut {p} ({reason})")
                    try:
                        db.log_decision(
                            self.tick_count, p, side, 0.0, 0.0, 1.0,
                            "loss_cut", reason,
                            size_pre=pos["qty"] * entry, size_post=0.0,
                            regime=regime.get("label"))
                    except Exception:
                        pass
                    if self.shadow is not None and p in self.shadow.positions:
                        try:
                            self.shadow.mirror_close(p, px)
                        except Exception:
                            pass

    @staticmethod
    def _exit_advisor_snapshot():
        try:
            from .learn.exit_advisor import exit_advisor
            s = exit_advisor.stats()
            return {"enabled": s.get("enabled"), "cuts": s.get("cuts"),
                    "states_learned": s.get("states_learned")}
        except Exception:
            return {}

    @staticmethod
    def _meme_snapshot():
        try:
            from .data.memes import memes
            s = memes.stats()
            held = [p for p in broker.positions if memes.is_meme(p)]
            return {"enabled": s.get("enabled"),
                    "active_in_universe": s.get("active_in_universe"),
                    "held": held, "known_count": s.get("known_count")}
        except Exception:
            return {}

    def snapshot(self):
        eq = self._total_equity(market)
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
            "cash": round(broker.cash, 2) + 0.0,     # + 0.0: never "-0.00"
            "exposure": round(self._total_exposure(market), 2),
            # unified paper-capital labeling (Task 5): "equity"/"cash"/
            # "exposure" above are the SHARED account total once the ledger
            # migration is confirmed (paper_portfolio.ready); these two extra
            # fields always show the crypto sleeve's own view so the operator
            # can see the breakdown, not just one combined number.
            "shared_ledger_ready": paper_portfolio.ready,
            "crypto_equity": round(broker.equity(market), 2),
            "crypto_exposure": round(broker.exposure(market), 2),
            "realized_pnl": round(broker.realized_pnl, 2),
            # unrealized = sum of per-position mark-to-market vs entry, correct
            # for BOTH longs and shorts (was a convoluted, short-wrong expr).
            "unrealized_pnl": round(sum(
                pos.get("side", 1) * ((market.price(p) or pos["entry"]) - pos["entry"])
                * pos["qty"] for p, pos in broker.positions.items()), 2),
            "risk": self.last_risk_status,
            "stance": stance.current(),
            "guardian": guardian.snapshot(),
            "exit_advisor": self._exit_advisor_snapshot(),
            "memes": self._meme_snapshot(),
            "hedge": hedger.snapshot(),
            "replay": self.replay_snapshot(),
            "core_holding": self._core_snapshot(),
            "exploration": self._exploration_snapshot(),
            "trade_filter": self._filter_snapshot(),
            "forward_test": self._safe_forward_test(),
            "pending_limit_orders": {p: {k: o[k] for k in ("limit", "direction", "notional", "ts")}
                                     for p, o in self.pending_entries.items()},
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
