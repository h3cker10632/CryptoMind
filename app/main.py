"""CryptoMind — FastAPI app: REST API + dashboard + background engines."""
import asyncio, os, sys, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from . import db
from .config import PRODUCTS
from .data.market import market
from .data.ws_market import WSMarket
from .data.research import research
from .data.derivatives import derivatives
from .data.universe import universe
from .data.memes import memes
from .data.calendar import calendar
from .nlp.sentiment import nlp
from .signals.engine import engine, STRATEGIES
from .execution.paper import broker
from .execution.shadow import ShadowBroker
from .risk.manager import risk
from .learn.loop import learner
from .orchestrator import orch
from .guardian import guardian
from .backtest.engine import full_report
from .backtest.composite import composite_report
from . import persistence
from . import settings as app_settings
from . import alerts
from . import security

app = FastAPI(title="CryptoMind", version="2.0")
app.middleware("http")(security.auth_middleware)   # protect state-changing routes
STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

# shadow execution: OMS-driven mirror account (NEVER trades real money) that
# runs in parallel to measure live-vs-paper divergence.
shadow = ShadowBroker(market)
ws_market = WSMarket(market)
orch.shadow = shadow


@app.on_event("startup")
async def startup():
    db.init()
    db.log_event("system", "CryptoMind starting (paper-trading mode; shadow OMS active)")
    # surface the API token location once so the operator can find it
    db.log_event("system", "Control API requires a token for mutations "
                           "(see api_token.txt or CRYPTOMIND_API_TOKEN); "
                           "loopback is allowed for local ops.")
    restored = persistence.load()
    if not restored:
        db.log_event("system", "No saved state found — starting fresh ($100k paper account)")
    from . import portfolio as portfolio_module
    portfolio_module.bootstrap(broker.cash)
    if portfolio_module.portfolio.ready:
        broker.bind_portfolio(portfolio_module.portfolio)
        from .markets.polymarket.broker import broker as pm_broker
        pm_broker.bind_portfolio(portfolio_module.portfolio)
    else:
        db.log_event("system", "Shared paper portfolio migration PENDING — "
                                "POST /api/portfolio/migration/confirm once legacy "
                                "Polymarket positions are confirmed settled/empty. "
                                "New entries are paused until then.")
    # Reconcile the alert/bot config (Telegram token, chat id, webhook) from
    # alerts.json BEFORE the worker tasks start, so the dashboard, /api/alerts
    # and the first outgoing alert all see the persisted credentials right away
    # (not a blank state until a worker happens to run _load_conf). This also
    # re-persists any env-seeded creds so they outlive the env var.
    alerts.load_conf()
    if alerts.status()["telegram_configured"]:
        db.log_event("system", "Telegram bot config restored from alerts.json")
    asyncio.create_task(market.run())
    asyncio.create_task(ws_market.run())
    asyncio.create_task(research.run())
    asyncio.create_task(derivatives.run())
    asyncio.create_task(universe.run())
    asyncio.create_task(memes.run())
    asyncio.create_task(calendar.run())
    asyncio.create_task(orch.decision_loop())
    asyncio.create_task(orch.watchdog())
    asyncio.create_task(orch.reconcile_loop())
    asyncio.create_task(orch.llm_advisor_loop())
    asyncio.create_task(orch.model_advisor_loop())   # crypto_ml_lab model vote
    asyncio.create_task(orch.ml_trainer_loop())      # autonomous metric-gated retrain
    asyncio.create_task(orch.researcher_loop())      # autonomous strategy discovery
    asyncio.create_task(orch.crawl_producer_loop())  # in-app crawl4ai producer
    asyncio.create_task(alerts.worker())
    asyncio.create_task(alerts.command_worker())   # two-way Telegram commands
    from .export import auto_export_loop
    asyncio.create_task(auto_export_loop())         # periodic full-state reports
    alerts.alert("info", "System started",
                 "CryptoMind is up (paper mode). State restored." )


@app.on_event("shutdown")
async def shutdown():
    persistence.save()
    alerts.persist()               # keep the Telegram bot config across restarts
    db.log_event("system", "State saved on shutdown")
    db.flush()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/status")
def status():
    return orch.snapshot()


@app.get("/api/portfolio/migration")
def portfolio_migration_status():
    from . import portfolio as portfolio_module
    return {"ready": portfolio_module.portfolio.ready,
            "cash": portfolio_module.portfolio.cash}


@app.post("/api/portfolio/migration/confirm")
def confirm_portfolio_migration():
    """Operator attestation that legacy (never-persisted) Polymarket positions
    are settled or empty. Required once before the shared paper ledger
    activates; a repeated call is a no-op and never adds cash."""
    from . import portfolio as portfolio_module
    was_ready = portfolio_module.portfolio.ready
    if not was_ready:
        portfolio_module.bootstrap(broker.cash, legacy_pm_confirmed=True)
    if portfolio_module.portfolio.ready:
        broker.bind_portfolio(portfolio_module.portfolio)
        from .markets.polymarket.broker import broker as pm_broker
        pm_broker.bind_portfolio(portfolio_module.portfolio)
    return {"ok": True, "ready": portfolio_module.portfolio.ready,
            "cash": portfolio_module.portfolio.cash,
            "already_migrated": was_ready}


@app.get("/api/signals")
def signals():
    return {"signals": engine.latest, "per_strategy": engine.per_strategy,
            "weights": learner.weights}


@app.get("/api/market")
def market_data():
    return {"tickers": market.tickers, "books": market.books,
            "features": {p: market.features(p) for p in PRODUCTS},
            "regime": market.regime(), "last_update": market.last_update,
            "ws": ws_market.stats()}


@app.get("/api/shadow")
def shadow_data():
    """Shadow-execution account (OMS-driven, no real money) + live-vs-paper
    divergence. This is the safe stand-in for the roadmap's Stage-2 shadow
    trading and Stage-3 divergence tracking."""
    return shadow.snapshot()


@app.get("/api/oms")
def oms_data():
    """Order-management state + append-only order/fill audit trail."""
    return {"stats": shadow.oms.stats(),
            "recent_order_events": db.order_events(limit=60)}


@app.get("/api/decisions")
def decisions_data(limit: int = 100, product: str = None, action: str = None):
    """Cycle-level decision audit trail — every candidate considered, its
    composite/confidence, the chosen action, the reason (e.g. the gate that
    blocked it) and the notional BEFORE and AFTER the risk cage clamped it.
    This is the "no position without a paper trail" record: it answers
    'why did nothing trade / why was it sized so small' from history."""
    return {"decisions": db.recent_decisions(min(limit, 500), product, action),
            "guardian": guardian.snapshot()}


@app.get("/api/security")
def security_info(request: Request):
    """Tells the operator whether their control API is protected. Never
    returns the token itself."""
    return {"auth_required_for_mutations": True,
            "loopback_allowed": os.environ.get("CRYPTOMIND_ALLOW_LOOPBACK", "1") == "1",
            "token_source": ("env CRYPTOMIND_API_TOKEN"
                             if os.environ.get("CRYPTOMIND_API_TOKEN")
                             else "api_token.txt"),
            "your_request_authorized": security.check(request)}


@app.get("/api/derivatives")
def derivatives_data():
    return {"metrics": derivatives.metrics,
            "features": {p: derivatives.features(p) for p in PRODUCTS},
            "healthy": derivatives.healthy,
            "last_update": derivatives.last_update}


@app.get("/api/universe")
def universe_data():
    return {**universe.stats(), "sources": research.source_stats()}


@app.get("/api/calendar")
def calendar_data():
    return calendar.stats()


@app.get("/api/alerts")
def alerts_status():
    return alerts.status()


@app.post("/api/alerts/config")
async def alerts_config(request: Request):
    """Configure channels: {telegram_bot_token, telegram_chat_id,
    webhook_url, push_level, push_trades}. Values persist to alerts.json.
    Setting telegram_bot_token + telegram_chat_id also activates the two-way
    command bot (/status, /positions, /resetkill, ...)."""
    changes = await request.json()
    st = alerts.save_conf(changes)
    db.log_event("system", "Alert config updated "
                 f"(telegram={'on' if st['telegram_configured'] else 'off'}, "
                 f"webhook={'on' if st['webhook_configured'] else 'off'})")
    return st


@app.post("/api/alerts/test")
async def alerts_test():
    return await alerts.send_test()


@app.get("/api/research")
def research_data():
    return {"documents": research.documents[:50],
            "macro_documents": [d for d in research.documents
                                if d.get("kind") == "macro"][:15],
            "fear_greed": research.fear_greed,
            "research_queue": research.research_queue,
            "narratives": nlp.narratives,
            "asset_sentiment": nlp.asset_sentiment,
            "market_sentiment": nlp.market_sentiment,
            "macro_sentiment": nlp.macro_sentiment,
            "macro_docs": nlp.macro_docs}


@app.get("/api/learning")
def learning():
    return learner.full_stats()


@app.get("/api/ml_live")
def ml_live(product: str = None):
    """Live introspection into the online neural net for the ML-learning tab:
    the feature row currently flowing in for `product`, the model's prediction +
    uncertainty on it, the rolling learning curves, and the recent
    predicted-vs-realized stream so the operator can watch it learn."""
    from .learn.online_model import committee, build_x
    from .nlp.sentiment import nlp
    # choose a product: explicit → open position → first in universe
    prods = list(PRODUCTS)
    if product not in prods:
        product = (next(iter(broker.positions), None)
                   or (prods[0] if prods else None))
    x = None
    f = market.features(product) if product else None
    if f:
        try:
            asset_sent, _ = nlp.asset_score(product)
            x = build_x(f, asset_sent, nlp.market_sentiment,
                        derivatives.features(product))
        except Exception:
            x = None
    state = committee.live_state(x)
    # a compact per-product prediction map ("data being transferred" per coin)
    per_product = {}
    for p in prods:
        pf = market.features(p)
        if not pf:
            continue
        try:
            a_s, _ = nlp.asset_score(p)
            px = build_x(pf, a_s, nlp.market_sentiment, derivatives.features(p))
            per_product[p] = {
                "prediction": round(committee.predict(px), 4),
                "held": p in broker.positions,
            }
        except Exception:
            continue
    state["product"] = product
    state["products"] = prods
    state["per_product"] = per_product
    return state


@app.get("/api/llm_live")
def llm_live():
    """Live introspection into the LLM advisor for the LLM tab: exactly what the
    model is shown per coin (the compact numeric context), the directional lean +
    reasoning it returned, how fresh that opinion is, and how much the bandit is
    weighting the 'llm' arm — i.e. what it's seeing AND how it's affecting trades."""
    from .learn.llm_advisor import advisor
    st = advisor.stats()
    prods = list(PRODUCTS)
    now = time.time()
    ttl = advisor._ttl()
    cadence = advisor._cadence()

    # bandit weight of the 'llm' arm vs the rest (how much say it has)
    weights = dict(getattr(learner, "weights", {}) or {})
    arm_w = weights.get("llm")
    others = [v for k, v in weights.items() if k != "llm"]
    avg_w = (sum(weights.values()) / len(weights)) if weights else None
    rank = None
    if arm_w is not None and weights:
        rank = 1 + sum(1 for v in weights.values() if v > arm_w)

    per_product = {}
    for p in prods:
        entry = advisor._leans.get(p) or {}
        lean = entry.get("lean")
        ts = entry.get("ts", 0.0)
        age = (now - ts) if ts else None
        expired = (age is None) or (age > ttl)
        try:
            ctx = advisor.build_context(p, market, nlp)
        except Exception:
            ctx = None
        raw = None
        try:
            raw = engine.per_strategy.get(p, {}).get("llm")
        except Exception:
            raw = None
        per_product[p] = {
            "context": ctx,
            "lean": lean,
            "why": entry.get("why", ""),
            "ts": ts,
            "age_sec": age,
            "expired": expired,
            "contributing": bool(st["configured"] and lean not in (None, 0.0)
                                 and not expired),
            "engine_raw": raw,
            "held": p in broker.positions,
        }
    return {
        **st,
        "now": now, "ttl_sec": ttl, "cadence_sec": cadence,
        "arm_weight": arm_w, "arm_weight_avg": avg_w,
        "arm_rank": rank, "n_arms": len(weights),
        "next_refresh_sec": max(0.0, cadence - (now - st["last_refresh"]))
                            if st["last_refresh"] else 0.0,
        "products": prods, "per_product": per_product,
    }


@app.post("/api/control/evolve")
def trigger_evolution(product: str = None):
    """Manually kick off a genetic-evolution run. With no product,
    rotates through the universe (also runs automatically every 20 min)."""
    if product is not None and product not in PRODUCTS:
        return JSONResponse({"error": f"unknown product {product}"}, 400)
    learner.last_evolution_start = 0.0
    learner.maybe_evolve(product)
    return {"started": True, "product": product or "auto-rotation"}


@app.post("/api/control/evolve_universe")
def trigger_universe_evolution():
    """Force a cross-sectional (universe-pooled) GA run now, regardless of the
    ga_cross_sectional tunable — the dashboard's 'Run cross-sectional now'."""
    if learner._evo_thread and learner._evo_thread.is_alive():
        return {"started": False, "reason": "an evolution run is already in progress"}
    learner.maybe_evolve(force_universe=True)
    return {"started": True, "product": "(universe)"}


@app.get("/api/trades")
def trades():
    return {"open": [dict(p) for p in broker.positions.values()],
            "closed": broker.closed_trades[-50:], "stats": broker.stats()}


_EQUITY_TF = {
    "20s": 20, "5m": 300, "10m": 600, "1h": 3600, "1d": 86400,
    "1w": 7 * 86400, "1m": 30 * 86400,
}


@app.get("/api/equity")
def equity(tf: str = "1d"):
    """Paper-account equity curve for a lookback window (20s / 5m / 1d / 1w / 1m)."""
    key = tf if tf in _EQUITY_TF else "1d"
    curve = db.equity_since(time.time() - _EQUITY_TF[key])
    return {"tf": key, "curve": curve}


@app.get("/api/events")
def events():
    return {"events": db.recent("events", 100)}


@app.get("/api/export")
def export_all(history: int = 1000, features: bool = True,
               analytics: bool = True, download: bool = True):
    """Aggregate EVERY data surface into ONE JSON document: meta/config, status,
    trades (full ledger), computed analytics (Sharpe/Sortino/profit-factor/
    streaks + per-product/per-exit-reason/per-strategy breakdowns), equity curves,
    signals, market (+features/closes), a consolidated per-product cross-section,
    derivatives, universe, memes, calendar, research/sentiment, the full learning
    stack, guardian/stance/risk/hedge/shadow, decision/order/trade/signal audit
    trails, settings, tunables, alerts, and security posture. `download=true`
    returns it as a timestamped file attachment; `features=false`/`analytics=false`
    shrink it; `history` caps rows per history section (max 5000)."""
    from .export import build_export
    data = build_export(history_limit=min(max(history, 1), 5000),
                        include_features=features,
                        include_analytics=analytics)
    if download:
        fname = time.strftime("cryptomind_export_%Y%m%d_%H%M%S.json", time.gmtime())
        return JSONResponse(data, headers={
            "Content-Disposition": f'attachment; filename="{fname}"'})
    return data


@app.get("/api/ml_dataset")
def ml_dataset(limit: int = 100000, download: bool = True):
    """Emit a TRAINING-GRADE dataset for offline ML (crypto_ml_lab). Unlike
    /api/export (a snapshot with mostly-unlabelled newest signals), this returns
    `events` (a leakage-free per-bar time series in the lab's contract) plus
    `labels` (only signals with a realized fwd_return, joined to decision-time
    context and trade outcomes). `download=true` returns it as a file attachment;
    `limit` caps labelled rows (max 500000). See docs/crypto_ml_lab_integration.md."""
    from .ml_export import build_ml_dataset
    data = build_ml_dataset(limit=min(max(limit, 1), 500000))
    if download:
        fname = time.strftime("cryptomind_ml_dataset_%Y%m%d_%H%M%S.json", time.gmtime())
        return JSONResponse(data, headers={
            "Content-Disposition": f'attachment; filename="{fname}"'})
    return data


@app.post("/api/control/pause")
def pause():
    from . import settings
    settings.update({"trading_paused": True})
    orch.running = False
    db.log_event("system", "Trading PAUSED by operator")
    return {"trading_enabled": False}


@app.post("/api/control/resume")
def resume():
    from . import settings
    # Resuming is meaningless while the kill switch is tripped: can_open()
    # blocks every entry and the status stays KILLED. Refuse clearly instead
    # of silently flipping trading_enabled with no visible effect.
    if risk.killed:
        return {"trading_enabled": False, "blocked": "kill_switch",
                "message": "Trading is KILLED — reset the kill switch first "
                           f"({risk.kill_reason or 'reason not recorded'})."}
    if risk.halted_today:
        return {"trading_enabled": False, "blocked": "daily_halt",
                "message": "Daily loss halt active — reset the kill switch to "
                           "clear it, or wait for the next UTC day."}
    settings.update({"trading_paused": False})
    orch.running = True
    db.log_event("system", "Trading RESUMED by operator")
    return {"trading_enabled": True}


@app.post("/api/control/kill")
def kill():
    risk.trip_kill("Manual: operator pressed the kill switch")
    orch.running = False
    flattened, stranded = [], []
    for p in list(broker.positions.keys()):
        px = market.price(p)
        if px is None:
            # feed down during a kill is EXACTLY when we must not fabricate a
            # flat exit at entry price — leave the position, halt, and alert.
            stranded.append(p)
            continue
        broker.sell(p, px, "KILL SWITCH")
        flattened.append(p)
    if stranded:
        db.log_event("risk", f"KILL SWITCH: flattened {flattened}; COULD NOT price "
                             f"{stranded} (feed down) — trading halted, positions held")
        from .alerts import alert
        alert("critical", "KILL SWITCH — positions stranded",
              f"Flattened {flattened}. NO price for {stranded}; they remain open. "
              f"Trading halted. Resolve manually when the feed recovers.")
    else:
        db.log_event("risk", "KILL SWITCH triggered by operator — all positions flattened")
    return {"kill_switch": True, "flattened": flattened, "stranded": stranded}


@app.post("/api/control/shutdown")
def shutdown_server():
    """Full server shutdown: pause trading, then (per settings) flatten or
    keep positions, save state, and terminate. State restored on relaunch."""
    import signal, threading

    orch.running = False
    flatten = app_settings.get("flatten_on_shutdown")
    if flatten:
        stranded = []
        for p in list(broker.positions.keys()):
            px = market.price(p)
            if px is None:          # don't fabricate exits when the feed is down
                stranded.append(p)
                continue
            broker.sell(p, px, "SERVER SHUTDOWN")
        note = ("Positions flattened and state saved."
                if not stranded else
                f"Flattened all but {stranded} (no price; kept & will be "
                f"restored/re-managed on restart). State saved.")
    else:
        note = (f"{len(broker.positions)} open position(s) KEPT — they will "
                f"be restored and re-managed on restart.")
    persistence.save()
    alerts.persist()               # keep the Telegram bot config across restarts
    # marker tells the supervisor loop (run.sh) to NOT relaunch
    open(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".shutdown"), "w").close()
    db.log_event("system", f"SERVER SHUTDOWN by operator — "
                           f"{'positions flattened' if flatten else 'positions kept'}, "
                           f"state saved, terminating")

    def _die():
        import time as _t
        _t.sleep(1.0)                      # let the HTTP response flush
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_die, daemon=True).start()
    return {"shutting_down": True, "state_saved": True,
            "positions_flattened": flatten,
            "positions_kept": 0 if flatten else len(broker.positions),
            "note": note + " Restart the server to resume."}


def _listen_addr():
    """Host/port this process was started with (uvicorn argv), else 8000."""
    host, port = "0.0.0.0", 8000
    argv = sys.argv
    for i, a in enumerate(argv):
        if a in ("--host", "-H") and i + 1 < len(argv):
            host = argv[i + 1]
        elif a.startswith("--host="):
            host = a.split("=", 1)[1]
        elif a in ("--port", "-p") and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith("--port="):
            try:
                port = int(a.split("=", 1)[1])
            except ValueError:
                pass
    return host, port


def _spawn_successor():
    """Launch relaunch.py detached so it can start uvicorn after we die.

    Skipped when CRYPTOMIND_SUPERVISED=1 (run.sh / run.ps1 already loop).
    """
    import subprocess
    if os.environ.get("CRYPTOMIND_SUPERVISED"):
        return False
    root = os.path.dirname(os.path.dirname(__file__))
    helper = os.path.join(root, "relaunch.py")
    host, port = _listen_addr()
    cmd = [sys.executable, helper, str(os.getpid()), host, str(port),
           root, sys.executable]
    kwargs = dict(cwd=root, stdin=subprocess.DEVNULL,
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                  close_fds=True)
    if os.name == "nt":
        # DETACHED + BREAKAWAY so we outlive uvicorn (job objects / IDE
        # terminals otherwise kill the helper when this process exits).
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        kwargs["creationflags"] = (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                                   | CREATE_BREAKAWAY_FROM_JOB)
        try:
            subprocess.Popen(cmd, **kwargs)
        except OSError:
            kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(cmd, **kwargs)
        return True
    kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)
    return True


@app.post("/api/control/restart")
def restart_server():
    """Graceful restart: save state, exit, relaunch with current code.

    If a supervisor (run.sh / run.ps1) is wrapping us, it relaunches.
    Otherwise we spawn relaunch.py ourselves so the Restart button works
    when uvicorn was started directly (the usual Windows case).
    """
    import signal, threading

    marker = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".shutdown")
    if os.path.exists(marker):
        os.remove(marker)
    persistence.save()
    alerts.persist()               # keep the Telegram bot config across restarts
    spawned = _spawn_successor()
    db.log_event("system", "RESTART requested by operator — state saved, "
                           + ("successor spawned" if spawned
                              else "supervisor will relaunch"))

    def _die():
        import time as _t
        _t.sleep(1.0)
        if os.name == "nt":
            os._exit(0)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_die, daemon=True).start()
    return {"restarting": True, "state_saved": True, "spawned": spawned,
            "note": "Server exiting; it relaunches with the latest code in "
                    "~3-10s. This page will reconnect automatically."}


@app.get("/api/settings")
def get_settings():
    return app_settings.public()


@app.post("/api/settings")
async def set_settings(request: Request):
    changes = await request.json()
    updated = app_settings.update(changes)
    db.log_event("system", f"Settings updated: {changes}", updated)
    return updated


@app.get("/api/invo/status")
def invo_status():
    from .data import invo
    return invo.collector.status()


@app.post("/api/invo/peek")
async def invo_peek():
    from .data import invo
    return await asyncio.to_thread(invo.peek_sync)


@app.post("/api/invo/collector/start")
async def invo_collector_start():
    from .data import invo
    return invo.collector.start()


@app.post("/api/invo/collector/stop")
async def invo_collector_stop():
    from .data import invo
    return invo.collector.stop()


@app.post("/api/invo/study/run")
async def invo_study_run():
    from .data import invo
    return await asyncio.to_thread(invo.run_study_sync)


@app.get("/api/ml/autotrain/status")
def ml_autotrain_status():
    from .learn.ml_trainer import trainer
    return trainer.status()


@app.post("/api/ml/autotrain/run")
async def ml_autotrain_run():
    """One-click 'run the full ML pipeline now' — forces a run regardless of the
    data-driven trigger (still metric-gated before it promotes anything)."""
    from .learn.ml_trainer import trainer
    return trainer.start_async(force=True)


@app.get("/api/polymarket/status")
def polymarket_status():
    from .markets.polymarket import engine
    return engine.snapshot()


@app.get("/api/polymarket/markets")
async def polymarket_markets():
    from .markets.polymarket import engine
    return await asyncio.to_thread(engine.peek)


@app.get("/api/polymarket/trades")
def polymarket_trades():
    from .markets.polymarket import engine
    return engine.trades()


@app.get("/api/polymarket/learning")
def polymarket_learning():
    from .markets.polymarket import engine
    return engine.snapshot()["learning"]


@app.get("/api/polymarket/forecasts")
def polymarket_forecasts(limit: int = 25):
    from .markets.polymarket import engine
    return engine.forecasts(limit)


@app.post("/api/polymarket/tick")
async def polymarket_tick():
    from .markets.polymarket import engine
    return await asyncio.to_thread(engine.tick)


@app.post("/api/polymarket/start")
async def polymarket_start():
    from .markets.polymarket import engine
    return engine.start()


@app.post("/api/polymarket/stop")
async def polymarket_stop():
    from .markets.polymarket import engine
    return engine.stop()


@app.post("/api/polymarket/reset")
def polymarket_reset():
    from .markets.polymarket import engine
    return engine.reset()


@app.get("/api/ingest/status")
def ingest_status():
    from .data import ingest
    return ingest.status()


@app.post("/api/ingest/push")
async def ingest_push(request: Request):
    """Accept external structured rows from ANY standalone producer (crawl4ai,
    Maxun, curl…). Body: a single row object or {"rows": [...]}. License-safe:
    this only ingests OUTPUT, never the producer's code."""
    from .data import ingest
    if not app_settings.get("ingest_enabled"):
        return {"accepted": 0, "rejected": 0,
                "error": "ingest_enabled is off — enable it in Settings"}
    body = await request.json()
    rows = body.get("rows", body) if isinstance(body, dict) else body
    return ingest.push(rows)


@app.get("/api/ingest/feature")
def ingest_feature(product: str):
    from .data import ingest
    return {"product": product, "feature": ingest.feature(product),
            "texts": ingest.latest_texts(product),
            "recent": ingest.recent(product, 10)}


@app.post("/api/ingest/study/run")
async def ingest_study_run(horizon_hours: float = 4.0):
    from .data import ingest
    return await asyncio.to_thread(ingest.study, horizon_hours)


@app.post("/api/ingest/clear")
def ingest_clear():
    from .data import ingest
    return ingest.clear()


@app.post("/api/ingest/crawl/run")
async def ingest_crawl_run():
    """Run the crawl4ai producer ONCE on demand (the dashboard 'crawl now' button).
    Crawls/scores configured or auto-discovered sources and pushes rows into the
    ingest seam. Imported lazily + off the hot path; never affects trading."""
    def _run():
        try:
            from tools.crawl4ai_signal.producer import run_once
            return run_once()
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return await asyncio.to_thread(_run)


# ---------------- Strategy Researcher (automated strategy discovery) ----------
@app.get("/api/research/status")
def research_status():
    from .learn.researcher import researcher
    return researcher.status()


@app.post("/api/research/run")
async def research_run(product: str = "", use_llm: bool | None = None):
    """Run one discovery pass. `product` empty = the whole universe. Gated on
    researcher_enabled. Runs off the hot path in a worker thread. Auto-promotes
    any candidate that clears the purged walk-forward + deflated-Sharpe gate."""
    from .learn.researcher import researcher
    if not app_settings.get("researcher_enabled"):
        return {"ok": False, "error": "researcher_enabled is off — enable it in Settings"}
    kw = {} if use_llm is None else {"use_llm": use_llm}
    if product:
        return await asyncio.to_thread(researcher.run_once, product, **kw)
    return await asyncio.to_thread(researcher.run_universe, None, **kw)


@app.get("/api/research/portfolio")
def research_portfolio():
    from .learn.researcher import researcher
    return researcher.status().get("portfolio", {})


@app.post("/api/research/clear")
def research_clear():
    from .learn.researcher import researcher
    return researcher.clear()


@app.get("/api/tunables")
def get_tunables():
    from . import tunables
    return {"meta": tunables.TUNABLES, "values": tunables.values()}


@app.post("/api/tunables")
async def set_tunables(request: Request):
    from . import tunables
    changes = await request.json()
    updated = tunables.update(changes)
    db.log_event("system", f"Tunables updated: {changes}")
    return {"values": updated}


@app.post("/api/tunables/reset")
def reset_tunables():
    from . import tunables
    db.log_event("system", "Tunables reset to defaults")
    return {"values": tunables.reset()}


@app.post("/api/control/reset-kill")
def reset_kill():
    risk.reset_kill()
    return {"kill_switch": False}


@app.post("/api/control/save")
def save_state():
    ok = persistence.save()
    return {"saved": ok, "path": persistence.STATE_PATH}


@app.post("/api/control/reset-account")
def reset_account():
    """Fresh paper account. Learned state (models, Q-table, bandit) is KEPT —
    only the trading account resets. Once the shared ledger is active this is
    a WHOLE-ACCOUNT reset (crypto + Polymarket share one balance), so it
    refuses while the crypto sleeve still holds open positions."""
    from .config import START_CASH
    from . import portfolio as portfolio_module
    if broker.positions:
        return {"ok": False,
                "error": "crypto sleeve has open positions; close them first"}
    if portfolio_module.portfolio.ready:
        event_id = f"account-reset-{uuid.uuid4().hex}"
        if not portfolio_module.portfolio.reset_account(START_CASH, event_id):
            return {"ok": False, "error": "shared portfolio reset failed"}
    else:
        broker.cash = START_CASH
    broker.positions = {}
    broker.closed_trades = []
    broker.realized_pnl = 0.0
    risk.reset_account_baselines(START_CASH)
    risk.killed = False
    risk.kill_reason = ""
    risk.kill_ts = None
    risk.halted_today = False
    risk.halt_reason = ""
    risk.consecutive_losses = 0
    risk.risk_scale = 1.0
    risk.cooldowns = {}
    persistence.save()
    db.log_event("system", "Paper account RESET to $100k (learned state kept)")
    return {"reset": True, "cash": START_CASH}


@app.get("/api/tearsheet")
def tearsheet(exclude_hedge: bool = False):
    """Honest risk-adjusted performance on the REAL closed-trade log — overall
    plus sliced by entry regime and exit reason (Sharpe/Sortino/Calmar/maxDD/
    profit-factor/win-rate-CI). This is the measured basis for regime/risk/exit
    tuning, not the in-sample backtest."""
    from .analytics.tearsheet import build_tearsheet
    from .config import START_CASH
    return build_tearsheet(broker.closed_trades, start_cash=START_CASH,
                           exclude_hedge=exclude_hedge)


@app.get("/api/backtest")
async def backtest(product: str = "BTC-USD", strategy: str = "trend"):
    if product not in PRODUCTS:
        return JSONResponse({"error": f"unknown product, use one of {PRODUCTS}"}, 400)
    if strategy not in ("trend", "meanrev", "breakout"):
        return JSONResponse({"error": "strategy must be trend|meanrev|breakout"}, 400)
    try:
        rep = await full_report(product, strategy)
        db.log_event("backtest", f"Backtest {strategy} on {product}",
                     {k: v for k, v in rep.items() if k not in ("full",)})
        return rep
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/backtest/composite")
async def backtest_composite(product: str = "BTC-USD"):
    """Validate the REAL ensemble (not 3 toy rules): composite vs buy-and-hold
    benchmark, walk-forward, Deflated Sharpe, PBO, and confidence intervals."""
    if product not in PRODUCTS:
        return JSONResponse({"error": f"unknown product, use one of {PRODUCTS}"}, 400)
    try:
        allow_shorts = app_settings.get("allow_shorts")
        rep = await asyncio.to_thread(_run_composite_sync, product, allow_shorts)
        db.log_event("backtest", f"Composite backtest on {product}",
                     {"beat_benchmark": rep.get("beat_benchmark"),
                      "validation": rep.get("validation", {}).get("gates")})
        return rep
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


def _run_composite_sync(product, allow_shorts):
    return asyncio.run(composite_report(product, weights=learner.weights,
                                        allow_shorts=allow_shorts))
