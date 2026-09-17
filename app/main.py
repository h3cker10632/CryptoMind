"""CryptoMind — FastAPI app: REST API + dashboard + background engines."""
import asyncio, os, sys, time
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from . import db
from .config import PRODUCTS
from .data.market import market
from .data.ws_market import WSMarket
from .data.research import research
from .data.derivatives import derivatives
from .data.universe import universe
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
    asyncio.create_task(market.run())
    asyncio.create_task(ws_market.run())
    asyncio.create_task(research.run())
    asyncio.create_task(derivatives.run())
    asyncio.create_task(universe.run())
    asyncio.create_task(calendar.run())
    asyncio.create_task(orch.decision_loop())
    asyncio.create_task(orch.watchdog())
    asyncio.create_task(orch.reconcile_loop())
    asyncio.create_task(orch.llm_advisor_loop())
    asyncio.create_task(alerts.worker())
    asyncio.create_task(alerts.command_worker())   # two-way Telegram commands
    alerts.alert("info", "System started",
                 "CryptoMind is up (paper mode). State restored." )


@app.on_event("shutdown")
async def shutdown():
    persistence.save()
    db.log_event("system", "State saved on shutdown")
    db.flush()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/status")
def status():
    return orch.snapshot()


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


@app.post("/api/control/evolve")
def trigger_evolution(product: str = None):
    """Manually kick off a genetic-evolution run. With no product,
    rotates through the universe (also runs automatically every 20 min)."""
    if product is not None and product not in PRODUCTS:
        return JSONResponse({"error": f"unknown product {product}"}, 400)
    learner.last_evolution_start = 0.0
    learner.maybe_evolve(product)
    return {"started": True, "product": product or "auto-rotation"}


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


@app.post("/api/control/pause")
def pause():
    orch.running = False
    db.log_event("system", "Trading PAUSED by operator")
    return {"trading_enabled": False}


@app.post("/api/control/resume")
def resume():
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
    return app_settings.load()


@app.post("/api/settings")
async def set_settings(request: Request):
    changes = await request.json()
    updated = app_settings.update(changes)
    db.log_event("system", f"Settings updated: {changes}", updated)
    return updated


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
    """Fresh $100k paper account. Learned state (models, Q-table, bandit)
    is KEPT — only the trading account resets."""
    from .config import START_CASH
    broker.cash = START_CASH
    broker.positions = {}
    broker.closed_trades = []
    broker.realized_pnl = 0.0
    risk.peak_equity = 0.0
    risk.day_start_equity = None
    risk.killed = False
    risk.halted_today = False
    risk.consecutive_losses = 0
    risk.cooldowns = {}
    persistence.save()
    db.log_event("system", "Paper account RESET to $100k (learned state kept)")
    return {"reset": True, "cash": START_CASH}


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
