"""Alerting — pushes critical events to Telegram (and/or a generic webhook)
and keeps an in-memory alert feed for the dashboard.

Setup (either or both):
  Telegram: set env vars or POST /api/settings/alerts
      TELEGRAM_BOT_TOKEN  — from @BotFather
      TELEGRAM_CHAT_ID    — your chat id (message @userinfobot to get it)
  Webhook: ALERT_WEBHOOK_URL — receives JSON {"level","title","message","ts"}
           (works with Discord webhooks, Slack, ntfy.sh, etc.)

Alert levels: critical (kill switch, crash, halt), warning (feed outages,
blackouts), info (trades, promotions). Only critical+warning are pushed
externally by default; everything appears in the dashboard feed.
"""
import asyncio, json, os, time, tempfile
import httpx
from . import db

CONF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alerts.json")

_state = {
    "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
    "webhook_url": os.environ.get("ALERT_WEBHOOK_URL", ""),
    "push_level": "warning",     # push warning+critical externally
    "push_trades": True,         # push every trade open/close to Telegram
}

recent_alerts = []               # newest first, capped
_queue: asyncio.Queue = None
_dedupe = {}                     # title -> last sent ts (anti-spam)
DEDUPE_SEC = 900
_commands_active = False         # set True once the command poller is running

LEVELS = {"info": 0, "warning": 1, "critical": 2}


def _load_conf():
    """Reconcile the in-memory alert/bot config with alerts.json.

    Disk is the source of truth once configured (via the API or the bot), but
    the Telegram bot token + chat id can also be seeded from environment
    variables on first boot. Those env-seeded credentials were previously NEVER
    written to disk, so a restart WITHOUT the env var silently lost the bot.
    Here we load disk values, then persist any credentials we ended up holding
    that the file doesn't already reflect — so the bot survives restart/shutdown
    regardless of how it was first configured.
    """
    saved = {}
    try:
        if os.path.exists(CONF_PATH):
            with open(CONF_PATH) as f:
                saved = json.load(f)
            for k in _state:
                # booleans must load even when False (e.g. push_trades off)
                if k == "push_trades":
                    if k in saved:
                        _state[k] = bool(saved[k])
                elif saved.get(k):
                    _state[k] = saved[k]
    except Exception:
        saved = {}
    # persist env-seeded (or otherwise not-yet-saved) credentials so they last
    creds = ("telegram_bot_token", "telegram_chat_id", "webhook_url")
    if any(_state.get(k) and not saved.get(k) for k in creds):
        try:
            save_conf({})        # writes the full current _state atomically
        except Exception:
            pass


def save_conf(changes: dict):
    for k, v in changes.items():
        if k in _state:
            _state[k] = v
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CONF_PATH))
        with os.fdopen(fd, "w") as f:
            json.dump(_state, f)
        os.replace(tmp, CONF_PATH)
    except Exception:
        pass
    return status()


def persist():
    """Flush the current alert/bot config (incl. the Telegram token) to disk.

    Called on startup (after loading) and on every shutdown/restart/stop path so
    the bot credentials ALWAYS survive, regardless of how they were set (env
    var, dashboard, or the bot itself) or how the process ends. Idempotent and
    exception-safe: it must never break shutdown.
    """
    try:
        return save_conf({})
    except Exception:
        return None


def load_conf():
    """Public entry point to reconcile config from disk. Safe to call at
    startup BEFORE the worker tasks run, so the dashboard/API and the very
    first alert see the persisted token immediately (not a blank _state)."""
    _load_conf()
    return status()


def status():
    return {
        "telegram_configured": bool(_state["telegram_bot_token"] and
                                    _state["telegram_chat_id"]),
        "webhook_configured": bool(_state["webhook_url"]),
        "push_level": _state["push_level"],
        "push_trades": _state["push_trades"],
        "commands_active": _commands_active,
        "recent": recent_alerts[:30],
    }


def alert(level: str, title: str, message: str = ""):
    """Fire an alert. Sync-safe: queues external push for the async worker."""
    entry = {"level": level, "title": title, "message": message,
             "ts": time.time()}
    recent_alerts.insert(0, entry)
    del recent_alerts[100:]

    if LEVELS.get(level, 0) >= LEVELS.get(_state["push_level"], 1):
        last = _dedupe.get(title, 0)
        if time.time() - last >= DEDUPE_SEC or level == "critical":
            _dedupe[title] = time.time()
            if _queue is not None:
                try:
                    _queue.put_nowait(entry)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Rich trade notifications
# ---------------------------------------------------------------------------
def _fmt_money(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


def notify_trade_open(pos):
    """Detailed Telegram/webhook message when a position is OPENED."""
    if not _state.get("push_trades", True):
        return
    side = "LONG 🟢" if pos.get("side", 1) > 0 else "SHORT 🔴"
    entry = pos.get("entry", 0.0)
    qty = pos.get("qty", 0.0)
    notional = qty * entry
    stop, take = pos.get("stop", 0.0), pos.get("take", 0.0)
    # risk/reward from the levels
    try:
        rr = abs(take - entry) / abs(entry - stop) if entry != stop else 0.0
    except Exception:
        rr = 0.0
    votes = pos.get("votes", {})
    votes_str = ", ".join(f"{k}:{v:+.2f}" for k, v in
                          sorted(votes.items(), key=lambda kv: -abs(kv[1]))[:5])
    lines = [
        f"📈 *OPENED {side}*  `{pos.get('product','?')}`",
        f"• Entry: `{entry:,.6f}`",
        f"• Qty: `{qty:,.6f}`  (notional {_fmt_money(notional)})",
        f"• Stop: `{stop:,.6f}`   TP: `{take:,.6f}`   R:R `{rr:.2f}`",
    ]
    if pos.get("regime_at_entry"):
        lines.append(f"• Regime: {pos['regime_at_entry']}")
    if votes_str:
        lines.append(f"• Strategy votes: {votes_str}")
    if pos.get("reason"):
        lines.append(f"• Reason: {pos['reason']}")
    _push_now("info", f"Opened {pos.get('product','?')}", "\n".join(lines),
              raw=True)


def notify_trade_close(trade):
    """Detailed Telegram/webhook message when a position is CLOSED."""
    if not _state.get("push_trades", True):
        return
    side = "LONG" if trade.get("side", 1) > 0 else "SHORT"
    entry = trade.get("entry", 0.0)
    exit_px = trade.get("exit", 0.0)
    qty = trade.get("qty", 0.0)
    pnl = trade.get("pnl", 0.0)
    win = pnl >= 0
    icon = "✅" if win else "❌"
    # pnl % on the position notional
    notional = qty * entry if entry else 0.0
    pnl_pct = (pnl / notional * 100) if notional else 0.0
    held = trade.get("closed", time.time()) - trade.get("opened", time.time())
    hrs = held / 3600.0
    held_str = f"{hrs:.1f}h" if hrs >= 1 else f"{held/60:.0f}m"
    lines = [
        f"{icon} *CLOSED {side}*  `{trade.get('product','?')}`",
        f"• Entry `{entry:,.6f}` → Exit `{exit_px:,.6f}`",
        f"• PnL: *{_fmt_money(pnl)}*  ({pnl_pct:+.2f}%)",
        f"• Qty: `{qty:,.6f}`   Held: {held_str}",
        f"• Reason: {trade.get('exit_reason','?')}",
    ]
    _push_now(("info" if win else "warning"),
              f"Closed {trade.get('product','?')}", "\n".join(lines), raw=True)


def _push_now(level, title, message, raw=False):
    """Record in the feed and queue for external push, bypassing the level
    gate (used for trade notifications, which the operator opts into via
    push_trades). `raw=True` sends `message` verbatim (already formatted)."""
    entry = {"level": level, "title": title, "message": message,
             "ts": time.time(), "raw": raw}
    recent_alerts.insert(0, entry)
    del recent_alerts[100:]
    if _queue is not None:
        try:
            _queue.put_nowait(entry)
        except Exception:
            pass


async def send_telegram(text, client=None, chat_id=None):
    """Low-level Telegram send (Markdown). Reused by the command bot for
    replies. Creates its own client if one isn't supplied."""
    tok = _state["telegram_bot_token"]
    chat = chat_id or _state["telegram_chat_id"]
    if not (tok and chat):
        return False
    own = client is None
    if own:
        client = httpx.AsyncClient()
    try:
        r = await client.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "Markdown",
                  "disable_web_page_preview": True},
            timeout=15)
        if r.status_code != 200:
            db.log_event("warn", f"Telegram send failed: {r.status_code} {r.text[:120]}")
            return False
        return True
    except Exception as e:
        db.log_event("warn", f"Telegram send error: {e}")
        return False
    finally:
        if own:
            await client.aclose()


async def send_telegram_photo(png_bytes, caption="", client=None, chat_id=None):
    """Send an image (PNG bytes) to Telegram via sendPhoto, with an optional
    Markdown caption. Used by the /chart command."""
    tok = _state["telegram_bot_token"]
    chat = chat_id or _state["telegram_chat_id"]
    if not (tok and chat):
        return False
    own = client is None
    if own:
        client = httpx.AsyncClient()
    try:
        r = await client.post(
            f"https://api.telegram.org/bot{tok}/sendPhoto",
            data={"chat_id": str(chat), "caption": caption,
                  "parse_mode": "Markdown"},
            files={"photo": ("equity.png", png_bytes, "image/png")},
            timeout=30)
        if r.status_code != 200:
            db.log_event("warn", f"Telegram photo failed: {r.status_code} {r.text[:120]}")
            return False
        return True
    except Exception as e:
        db.log_event("warn", f"Telegram photo error: {e}")
        return False
    finally:
        if own:
            await client.aclose()


def render_equity_png(tf="1d"):
    """Render an equity-curve PNG (bytes) for the given timeframe using Pillow.

    Returns (png_bytes, caption) or (None, message) if there isn't enough data.
    Pure-Pillow so it has no matplotlib dependency and is fast/thread-safe.
    """
    import time as _time
    from . import db
    windows = {"20s": 20, "5m": 300, "10m": 600, "1h": 3600, "1d": 86400,
               "1w": 7 * 86400, "1m": 30 * 86400}
    secs = windows.get(tf, 86400)
    curve = db.equity_since(_time.time() - secs)
    if not curve or len(curve) < 2:
        return None, "Not enough equity history yet to draw a chart."
    ys = [float(p["equity"]) for p in curve]
    xs = [float(p["ts"]) for p in curve]
    first, last = ys[0], ys[-1]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or (abs(hi) * 0.01 or 1.0)
    chg = last - first
    chg_pct = (chg / first * 100) if first else 0.0

    from PIL import Image, ImageDraw
    W, H = 900, 420
    ML, MR, MT, MB = 70, 20, 44, 40           # margins
    PW, PH = W - ML - MR, H - MT - MB
    bg = (17, 22, 33)
    grid = (38, 46, 62)
    up = (46, 204, 113)
    down = (231, 76, 60)
    txt = (150, 165, 190)
    line_col = up if chg >= 0 else down

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # title
    d.text((ML, 12), f"CryptoMind Equity — {tf}", fill=(220, 230, 245))
    # horizontal gridlines + y labels (5 divisions)
    for i in range(5):
        gy = MT + PH * i / 4
        d.line([(ML, gy), (ML + PW, gy)], fill=grid, width=1)
        val = hi - span * i / 4
        d.text((6, gy - 6), f"${val:,.0f}", fill=txt)

    n = len(ys)

    def px(idx):
        # evenly spaced by sample index (robust to clustered timestamps)
        return ML + PW * idx / (n - 1)

    def py(v):
        return MT + PH * (1 - (v - lo) / span)

    # baseline (starting equity) as a dashed reference
    by = py(first)
    for x0 in range(ML, ML + PW, 10):
        d.line([(x0, by), (x0 + 5, by)], fill=(90, 100, 120), width=1)

    # filled area under the curve + the line
    pts = [(px(i), py(ys[i])) for i in range(len(ys))]
    poly = pts + [(pts[-1][0], MT + PH), (pts[0][0], MT + PH)]
    fill = (up[0], up[1], up[2]) if chg >= 0 else (down[0], down[1], down[2])
    d.polygon(poly, fill=(fill[0] // 6 + bg[0], fill[1] // 6 + bg[1],
                          fill[2] // 6 + bg[2]))
    d.line(pts, fill=line_col, width=2)

    # last-point marker + value
    d.ellipse([pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3],
              fill=line_col)
    d.text((ML, H - 26),
           f"start ${first:,.0f}   now ${last:,.0f}   "
           f"{'+' if chg >= 0 else ''}{chg:,.0f} ({chg_pct:+.2f}%)",
           fill=txt)

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    caption = (f"*Equity — {tf}*\n"
               f"start `${first:,.0f}` → now `${last:,.0f}`  "
               f"({chg_pct:+.2f}%)")
    return buf.getvalue(), caption


async def _push_telegram(client, entry):
    tok, chat = _state["telegram_bot_token"], _state["telegram_chat_id"]
    if not (tok and chat):
        return
    if entry.get("raw"):
        text = entry["message"]           # already fully formatted
    else:
        icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(entry["level"], "")
        text = f"{icon} *CryptoMind — {entry['title']}*\n{entry['message']}"
    await send_telegram(text, client=client)


async def _push_webhook(client, entry):
    url = _state["webhook_url"]
    if not url:
        return
    try:
        # Discord-compatible payload with a generic JSON fallback
        payload = {"content": f"**CryptoMind — {entry['title']}**\n{entry['message']}",
                   **entry}
        await client.post(url, json=payload, timeout=15)
    except Exception as e:
        db.log_event("warn", f"Webhook push error: {e}")


async def worker():
    """Background task: drains the alert queue and pushes externally."""
    global _queue
    _queue = asyncio.Queue()
    _load_conf()
    async with httpx.AsyncClient() as client:
        while True:
            entry = await _queue.get()
            await _push_telegram(client, entry)
            await _push_webhook(client, entry)


async def send_test():
    """Send a test alert through all configured channels immediately."""
    entry = {"level": "critical", "title": "Test alert",
             "message": "If you can read this, alerting works.",
             "ts": time.time()}
    recent_alerts.insert(0, entry)
    async with httpx.AsyncClient() as client:
        await _push_telegram(client, entry)
        await _push_webhook(client, entry)
    return status()


# ===========================================================================
# Two-way Telegram command bot
# ===========================================================================
# A long-polling worker that reads messages from the operator's chat and runs
# commands: status, positions, trades, pnl, pause/resume, kill/resetkill,
# settings, tunables, etc. Only messages from the configured chat_id are
# honoured (everything else is ignored), so this is a private control channel.

HELP_TEXT = (
    "*CryptoMind — commands*\n"
    "/status — system + account snapshot\n"
    "/positions — open positions with live PnL\n"
    "/trades [n] — last n closed trades (default 5)\n"
    "/pnl — realized/unrealized PnL + win rate\n"
    "/balance — cash, equity, exposure\n"
    "/stats — trade statistics\n"
    "/brains — how much the system has learned (models/DB)\n"
    "/replay — latest strategy replay over the universe (/replay now = run it)\n"
    "/signals — current actionable signals\n"
    "/diag — per-coin: why each isn't trading (or would)\n"
    "/decisions [n] — recent decision audit trail (paper trail)\n"
    "/chart [tf] — equity chart image (tf: 1h,1d,1w,1m; default 1d)\n"
    "/risk — drawdown, daily loss, kill/halt state\n"
    "/why — why trading is killed/halted (if it is)\n"
    "/pause — stop opening new trades\n"
    "/resume — resume trading\n"
    "/close <product>|all — flatten one position (or all)\n"
    "/kill — 🛑 flatten all & stop (kill switch)\n"
    "/resetkill — ♻️ clear the kill switch + daily halt\n"
    "/shorts on|off — allow/deny short positions\n"
    "/llm on|off — optional LLM advisor vote (needs API key)\n"
    "/mode passive|auto|aggressive — trading stance\n"
    "/set <tunable> <value> — change a risk/cost knob\n"
    "/get <tunable> — read a tunable\n"
    "/tunables — list all tunables\n"
    "/notify on|off — toggle per-trade push messages\n"
    "/help — this list"
)


def _authorized(chat_id):
    want = str(_state.get("telegram_chat_id", "")).strip()
    return want and str(chat_id).strip() == want


def _cmd_status():
    from .orchestrator import orch
    s = orch.snapshot()
    trading = ("🛑 KILLED" if s["kill_switch"] else
               "⏸ HALTED" if s["halted_today"] else
               "▶️ ACTIVE" if s["trading_enabled"] else "⏸ PAUSED")
    up = int(s.get("uptime_sec", 0)); hrs = up // 3600; mins = (up % 3600) // 60
    lines = [
        "*CryptoMind status*",
        f"• Trading: {trading}",
        f"• Equity: {_fmt_money(s['equity'])}   Cash: {_fmt_money(s['cash'])}",
        f"• Exposure: {_fmt_money(s['exposure'])}",
        f"• Realized PnL: {_fmt_money(s['realized_pnl'])}   "
        f"Unrealized: {_fmt_money(s['unrealized_pnl'])}",
        f"• Open positions: {len(s.get('positions', []))}",
        f"• Regime: {s['regime']['label'] if s.get('regime') else '—'}",
        f"• Data feed: {'LIVE' if s['data_healthy'] else 'DOWN'}",
        f"• Uptime: {hrs}h {mins}m",
    ]
    if s["kill_switch"] and s.get("kill_reason"):
        lines.append(f"• ⚠️ Kill reason: {s['kill_reason']}")
    return "\n".join(lines)


def _cmd_positions():
    from .execution.paper import broker
    from .data.market import market
    if not broker.positions:
        return "No open positions."
    out = ["*Open positions*"]
    for p, pos in broker.positions.items():
        side = "LONG" if pos.get("side", 1) > 0 else "SHORT"
        px = market.price(p) or pos["entry"]
        upnl = pos.get("side", 1) * (px - pos["entry"]) * pos["qty"]
        pct = (upnl / (pos["qty"] * pos["entry"]) * 100) if pos["entry"] else 0
        out.append(
            f"`{p}` {side}\n"
            f"  entry `{pos['entry']:,.6f}` → now `{px:,.6f}`  ({pct:+.2f}%)\n"
            f"  uPnL {_fmt_money(upnl)}   stop `{pos.get('stop',0):,.6f}` "
            f"tp `{pos.get('take',0):,.6f}`")
    return "\n".join(out)


def _cmd_trades(n=5):
    from .execution.paper import broker
    trades = broker.closed_trades[-n:][::-1]
    if not trades:
        return "No closed trades yet."
    out = [f"*Last {len(trades)} trades*"]
    for t in trades:
        side = "LONG" if t.get("side", 1) > 0 else "SHORT"
        pnl = t.get("pnl", 0.0)
        icon = "✅" if pnl >= 0 else "❌"
        out.append(f"{icon} `{t['product']}` {side}  {_fmt_money(pnl)}  "
                   f"({t.get('exit_reason','?')})")
    return "\n".join(out)


def _cmd_pnl():
    from .execution.paper import broker
    from .data.market import market
    st = broker.stats()
    upnl = sum(pos.get("side", 1) * ((market.price(p) or pos["entry"]) - pos["entry"])
               * pos["qty"] for p, pos in broker.positions.items())
    wr = st["win_rate"]
    return ("*PnL*\n"
            f"• Realized: {_fmt_money(st['realized_pnl'])}\n"
            f"• Unrealized: {_fmt_money(upnl)}\n"
            f"• Trades: {st['trades']}   Win rate: "
            f"{(wr*100):.0f}%" if wr is not None else
            f"• Trades: {st['trades']}   Win rate: —")


def _cmd_balance():
    from .execution.paper import broker
    from .data.market import market
    eq = broker.equity(market)
    return ("*Balance*\n"
            f"• Cash: {_fmt_money(broker.cash)}\n"
            f"• Equity: {_fmt_money(eq)}\n"
            f"• Exposure: {_fmt_money(broker.exposure(market))}")


def _cmd_stats():
    from .execution.paper import broker
    st = broker.stats()
    wr = st["win_rate"]
    return ("*Trade stats*\n"
            f"• Trades: {st['trades']}\n"
            f"• Win rate: {(wr*100):.0f}%\n" if wr is not None else
            f"• Win rate: —\n") + (
            f"• Avg win: {_fmt_money(st['avg_win'])}\n"
            f"• Avg loss: {_fmt_money(st['avg_loss'])}\n"
            f"• Realized PnL: {_fmt_money(st['realized_pnl'])}")


def _cmd_brains():
    """How much has the system actually learned? Surfaces the live in-memory
    learning state (bandit arms, online model, RL Q-table, GA champions) plus
    the durable SQLite row counts, and where all of it is stored on disk."""
    from .learn.loop import learner
    from .learn.online_model import model, committee
    from .learn.rl_risk import agent as rl_agent
    from .learn.evolution import evolution
    from .learn.drift import page_hinkley
    from . import db

    arms = learner.bandit.arms
    regimes_seen = len({r for (r, _s) in arms})
    # v = (n, mean, M2); n is a decayed EFFECTIVE sample size (float), so round.
    total_obs = round(sum(v[0] for v in arms.values()), 1)
    mst = committee.stats()
    rst = rl_agent.stats()
    champs = evolution.champions or {}
    counts = db.learning_counts()

    acc = mst["directional_accuracy"]
    acc_str = "—" if acc is None else f"{acc*100:.0f}%"

    out = ["*🧠 CryptoMind — learning state*", ""]
    out.append("*Bandit (strategy allocation)*")
    out.append(f"• Arms populated: {len(arms)}  ({regimes_seen} regimes)")
    out.append(f"• Total observations: {total_obs}")
    out.append(f"• Trade attributions: {learner.trade_attributions}")
    out.append("")
    out.append("*Online model (committee of TinyMLPs)*")
    out.append(f"• Members: {mst.get('committee_members', 1)}   "
               f"updates: {mst['n_updates']}  "
               f"({'warm' if mst['warmed_up'] else 'cold'})")
    out.append(f"• Directional acc: {acc_str}   replay: {mst['replay_buffer']}")
    out.append("")
    out.append("*RL risk agent (Q-learning)*")
    out.append(f"• Updates: {rst['n_updates']}   states: {rst['states_visited']}")
    out.append(f"• Epsilon: {rst['epsilon']}   last reward: {rst['last_reward']}")
    out.append("")
    out.append("*Concept drift*")
    ph = getattr(page_hinkley, "stats", lambda: {})() or {}
    out.append(f"• Feature drift (PSI): {'yes' if learner.drift_state.get('drifting') else 'no'}"
               f"   error drift (Page-Hinkley): {ph.get('concept_drift_events', 0)} events")
    out.append("")
    out.append("*Loss-cut exit advisor*")
    try:
        est = learner._exit_advisor_stats()
        out.append(f"• States learned: {est.get('states_learned', 0)}   "
                   f"cuts made: {est.get('cuts', 0)}   "
                   f"updates: {est.get('updates', 0)}")
        worst = est.get("worst_hold_states") or []
        if worst:
            w = worst[0]
            out.append(f"• Worst hold state: {w['state']} "
                       f"(next {w['mean_next_return']*100:+.2f}%, n={w['n']})")
    except Exception:
        out.append("• (initialising)")
    out.append("")
    out.append("*Direction learner (long vs short)*")
    try:
        dst = learner._direction_stats()
        out.append(f"• Flips: {dst.get('flips', 0)}   "
                   f"HTF vetoes: {dst.get('vetoes', 0)}   "
                   f"updates: {dst.get('updates', 0)}")
        for reg, sides in list((dst.get("regime_edge") or {}).items())[:3]:
            parts = [f"{k} {v['mean_net']*100:+.2f}%(n{v['n']})"
                     for k, v in sides.items()]
            out.append(f"• {reg}: " + ", ".join(parts))
    except Exception:
        out.append("• (initialising)")
    out.append("")
    out.append("*Meme sleeve*")
    try:
        from .data.memes import memes
        ms = memes.stats()
        out.append(f"• Enabled: {'yes' if ms.get('enabled') else 'no'}   "
                   f"known memes: {ms.get('known_count', 0)}")
        active = ms.get("active_in_universe") or []
        out.append("• Trading: " + (", ".join(active) if active else "none in universe yet"))
    except Exception:
        out.append("• (initialising)")
    out.append("")
    out.append("*Chart patterns*")
    try:
        from .data.market import market
        from .execution.paper import broker
        held = list(broker.positions.keys())
        # held positions first (patterns that help or hurt live risk), then a
        # couple of other active names for context.
        others = [p for p in list(market.tickers)[:6] if p not in held]
        shown = 0
        for p in held + others:
            if shown >= 5:
                break
            f = market.features(p)
            rep = f.get("patterns") if f else None
            if not rep or not rep.get("detected"):
                continue
            side = broker.positions.get(p, {}).get("side", 0)
            top = rep["detected"][0]
            tag = "⚠️ vs open" if (side and (
                (side > 0 and top["direction"] == "bearish") or
                (side < 0 and top["direction"] == "bullish"))) else \
                ("✓ open" if side else "")
            out.append(f"• {p}: {rep['structure']} — {top['name']} "
                       f"({top['direction']}) {tag}".rstrip())
            shown += 1
        if shown == 0:
            out.append("• No notable patterns right now")
    except Exception:
        out.append("• (initialising)")
    out.append("")
    out.append("*GA evolution*")
    ports = evolution.champion_portfolios or {}
    if champs:
        out.append(f"• Champions: {len(champs)} → " +
                   ", ".join(sorted(champs.keys())))
        if ports:
            out.append("• Portfolios (top-k): " +
                       ", ".join(f"{p}×{len(g)}" for p, g in sorted(ports.items())))
    else:
        out.append("• Champions: none promoted yet")
    out.append("")
    out.append("*Durable history (SQLite)*")
    out.append(f"• Signals: {counts['signals_scored']} scored / "
               f"{counts['signals_pending']} pending")
    out.append(f"• Trades: {counts['trades']}   equity pts: {counts['equity']}")
    out.append(f"• Events: {counts['events']}   order events: {counts['order_events']}")
    out.append("")
    out.append("_Stored in: `cryptomind.db` (history) + `state.json` "
               "(model snapshot, saved ~1/min)._")
    return "\n".join(out)


def _cmd_signals():
    from .signals.engine import engine
    sigs = [s for s in engine.latest.values() if s.get("actionable")]
    if not sigs:
        return "No actionable signals right now."
    sigs.sort(key=lambda s: -s["confidence"])
    out = ["*Actionable signals*"]
    for s in sigs[:8]:
        d = "LONG" if s["direction"] > 0 else "SHORT"
        out.append(f"`{s['product']}` {d}  conf {s['confidence']:.2f}  "
                   f"composite {s['composite']:+.2f}  ({s['regime']})")
    return "\n".join(out)


def _cmd_diag():
    """Run the full entry-gate chain per product and report the single
    blocking reason for each — a one-command answer to 'why am I not trading?'.
    Mirrors exactly the checks the orchestrator runs before opening a trade."""
    from .orchestrator import orch
    from .risk.manager import risk
    from .execution.paper import broker
    from .data.market import market
    from .signals.engine import engine
    from .tunables import tv
    from .config import PRODUCTS

    lines = ["*Trade diagnostics — why each coin isn't trading*"]

    # ---- global blocks first (affect everything) ----
    if not orch.running:
        lines.append("⏸ GLOBAL: trading is paused — use /resume")
    if risk.killed:
        lines.append(f"🛑 GLOBAL: KILLED — {risk.kill_reason or 'reason not recorded'} "
                     "(use /resetkill)")
    if risk.halted_today:
        lines.append(f"⏸ GLOBAL: daily loss halt — {risk.halt_reason or ''} "
                     "(use /resetkill)")
    if not market.healthy:
        lines.append("📵 GLOBAL: market data feed is DOWN — entries blocked")
    try:
        from .data.calendar import calendar
        bl, ev = calendar.blackout()
        if bl:
            lines.append(f"🗓 GLOBAL: macro blackout — {ev['title']}")
    except Exception:
        pass

    equity = broker.equity(market)
    rs = orch.last_risk_status or {"effective_risk_scale": 1.0}
    lines.append("")

    for p in PRODUCTS:
        sig = engine.latest.get(p)
        # 1) risk gates (kill/halt/feed/blackout/inpos/maxpos/exposure/cooldown/price)
        ok, why = risk.can_open(p, broker, market, market.healthy)
        if not ok:
            lines.append(f"`{p}` ⛔ {why}")
            continue
        # 2) need a signal at all
        if not sig:
            lines.append(f"`{p}` … no signal yet (warming up / not enough data)")
            continue
        # 3) actionable? (confidence gate + direction allowed)
        if not sig.get("actionable"):
            lines.append(f"`{p}` … signal not actionable "
                         f"(conf {sig['confidence']:.2f}, need higher; "
                         f"composite {sig['composite']:+.2f})")
            continue
        # 4) funding gate
        ok, why = risk.funding_gate(p, sig["direction"])
        if not ok:
            lines.append(f"`{p}` ⛔ {why}")
            continue
        # 5) sizing / cost-viability gate
        notional, stop, take = risk.size(
            equity, sig["price"], sig["atr"], sig["confidence"], rs,
            direction=sig["direction"], product=p,
            ml_confidence=sig.get("ml_confidence", 1.0))
        if notional <= 0:
            lines.append(f"`{p}` ⛔ cost gate: ATR target can't clear fees "
                         "(unprofitable — try lower fee_rate/cost_multiple)")
            continue
        if notional < tv("min_notional"):
            lines.append(f"`{p}` ⛔ notional ${notional:,.0f} < min "
                         f"${tv('min_notional')}")
            continue
        d = "LONG" if sig["direction"] > 0 else "SHORT"
        lines.append(f"`{p}` ✅ WOULD TRADE {d}  conf {sig['confidence']:.2f}  "
                     f"~${notional:,.0f}")

    # helpful hints if literally nothing is tradable
    tradable = any("✅" in ln for ln in lines)
    if not tradable:
        lines.append("")
        lines.append("_Tips:_ /set min_confidence 0.35 (lower entry bar) · "
                     "/mode aggressive · /get fee_rate (0.5% is strict — "
                     "/set fee_rate 0.001) · /get cost_multiple")
    return "\n".join(lines)


def _cmd_risk():
    from .risk.manager import risk
    from .orchestrator import orch
    s = orch.snapshot()
    r = s.get("risk") or {}
    lines = [
        "*Risk*",
        f"• Drawdown: {(r.get('drawdown',0)*100):.1f}%  "
        f"(kill at {tv_pct('max_drawdown_kill')})",
        f"• Daily loss: {(r.get('day_loss',0)*100):.1f}%  "
        f"(halt at {tv_pct('daily_loss_limit')})",
        f"• Kill switch: {'ON 🛑' if risk.killed else 'off'}",
        f"• Daily halt: {'ON ⏸' if risk.halted_today else 'off'}",
        f"• Loss streak: {risk.consecutive_losses}",
    ]
    if risk.killed and risk.kill_reason:
        lines.append(f"• Kill reason: {risk.kill_reason}")
    return "\n".join(lines)


def tv_pct(key):
    from .tunables import tv
    return f"{tv(key)*100:.0f}%"


def _cmd_decisions(n=6):
    """Recent decision-audit rows — the cycle-level paper trail: what each
    candidate's composite/confidence was, the action taken, and why."""
    from . import db
    from .guardian import guardian
    rows = db.recent_decisions(min(n, 15))
    g = guardian.snapshot()
    head = "*Recent decisions*"
    if g.get("safe_mode"):
        head += f"\n🟡 SAFE MODE: {g.get('safe_reason', '')}"
    head += (f"\nEntries last hour: {g.get('entries_last_hour', 0)}"
             f"/{g.get('max_entries_per_hour', 0)}")
    if not rows:
        return head + "\n_(no decisions recorded yet — feeds still warming)_"
    icon = {"enter": "✅", "explore": "🔬", "skip": "⏭", "reject": "⛔"}
    lines = [head]
    for r in rows:
        sym = r["product"].split("-")[0]
        d = "long" if r["direction"] > 0 else "short"
        i = icon.get(r["action"], "•")
        line = (f"{i} {sym} {d} conf={r['confidence']:.2f} "
                f"cmp={r['composite']:+.2f} — {r['action']}")
        if r.get("reason"):
            line += f" ({r['reason']})"
        if r.get("size_post"):
            line += f" ${r['size_post']:,.0f}"
        lines.append(line)
    return "\n".join(lines)


def _cmd_why():
    from .risk.manager import risk
    if risk.killed:
        return f"🛑 KILLED — {risk.kill_reason or 'reason not recorded'}\n" \
               "Use /resetkill to clear it and resume."
    if risk.halted_today:
        return f"⏸ HALTED — {risk.halt_reason or 'daily loss halt'}\n" \
               "Use /resetkill to clear it, or wait for the next UTC day."
    return "✅ Trading is not killed or halted."


def _cmd_kill():
    from .risk.manager import risk
    from .execution.paper import broker
    from .data.market import market
    from .orchestrator import orch
    risk.trip_kill("Manual: /kill from Telegram")
    orch.running = False
    flat = []
    for p in list(broker.positions.keys()):
        px = market.price(p)
        if px is not None:
            broker.sell(p, px, "KILL SWITCH (telegram)")
            flat.append(p)
    return f"🛑 Kill switch ON. Flattened: {', '.join(flat) or 'none'}.\n" \
           "Trading stopped until /resetkill."


def _resolve_product(arg):
    """Map a loose user token to a real open-position product.
    Accepts 'btc', 'BTC', 'btc-usd', 'BTC-USD' (case-insensitive)."""
    from .execution.paper import broker
    if not arg:
        return None
    a = arg.strip().upper()
    if a in broker.positions:
        return a
    if (a + "-USD") in broker.positions:
        return a + "-USD"
    # match on the base symbol before the dash
    for p in broker.positions:
        if p.split("-")[0] == a:
            return p
    return None


def _cmd_close(args):
    from .execution.paper import broker
    from .data.market import market
    from .risk.manager import risk
    from .learn.loop import learner
    if not args:
        if not broker.positions:
            return "No open positions to close."
        return ("Usage: /close <product>  (e.g. /close BTC)\nOpen: "
                + ", ".join(broker.positions.keys()))
    arg = args[0]
    # /close all — flatten everything
    if arg.lower() == "all":
        if not broker.positions:
            return "No open positions to close."
        closed, stranded = [], []
        for p in list(broker.positions.keys()):
            px = market.price(p)
            if px is None:
                stranded.append(p)
                continue
            t = broker.sell(p, px, "manual /close all (telegram)")
            if t:
                risk.on_trade_closed(t)
                learner.on_trade_closed(t)
                closed.append(f"{p} ({t['pnl']:+,.2f})")
        msg = f"✅ Closed: {', '.join(closed) or 'none'}."
        if stranded:
            msg += f"\n⚠️ No price for: {', '.join(stranded)} (kept open)."
        return msg
    # single product
    p = _resolve_product(arg)
    if not p:
        from .execution.paper import broker as b
        return (f"No open position matching `{arg}`.\n"
                + ("Open: " + ", ".join(b.positions.keys()) if b.positions
                   else "No open positions."))
    px = market.price(p)
    if px is None:
        return f"⚠️ Can't price `{p}` right now (feed issue) — not closed."
    pos = broker.positions[p]
    side = "LONG" if pos.get("side", 1) > 0 else "SHORT"
    t = broker.sell(p, px, "manual /close (telegram)")
    if not t:
        return f"Failed to close `{p}`."
    risk.on_trade_closed(t)
    learner.on_trade_closed(t)
    # keep the shadow OMS in sync, mirroring the orchestrator's close path
    try:
        sh = getattr(__import__("app.orchestrator", fromlist=["orch"]).orch,
                     "shadow", None)
        if sh is not None and p in getattr(sh, "positions", {}):
            sh.mirror_close(p, t.get("exit", px))
    except Exception:
        pass
    return (f"✅ Closed {side} `{p}` @ `{t['exit']:,.6f}`\n"
            f"• PnL: *{t['pnl']:+,.2f}*  ({t.get('exit_reason','manual')})")


def _cmd_resetkill():
    from .risk.manager import risk
    from .orchestrator import orch
    was = risk.killed or risk.halted_today
    risk.reset_kill()
    orch.running = True
    return ("♻️ Kill switch + daily halt cleared. Trading resumed."
            if was else "Nothing to reset — trading was already live. Resumed.")


def _cmd_pause():
    from .orchestrator import orch
    orch.running = False
    db.log_event("system", "Trading PAUSED via Telegram")
    return "⏸ Trading paused (no new entries). Open positions still managed."


def _cmd_resume():
    from .orchestrator import orch
    from .risk.manager import risk
    if risk.killed:
        return f"Can't resume — KILLED ({risk.kill_reason or 'reason not recorded'}).\n" \
               "Use /resetkill first."
    if risk.halted_today:
        return "Can't resume — daily loss halt active. Use /resetkill or wait for next day."
    orch.running = True
    db.log_event("system", "Trading RESUMED via Telegram")
    return "▶️ Trading resumed."


def _cmd_shorts(arg):
    from . import settings as app_settings
    if arg not in ("on", "off"):
        return "Usage: /shorts on|off"
    app_settings.update({"allow_shorts": arg == "on"})
    return f"Shorts {'ENABLED' if arg == 'on' else 'DISABLED'}."


def _cmd_llm(arg):
    from . import settings as app_settings
    from .learn.llm_advisor import advisor
    if arg not in ("on", "off"):
        st = advisor.stats()
        return (f"LLM advisor: {'ON' if st['enabled'] else 'off'}"
                f"{'' if st['configured'] else ' (no API key — inert)'}\n"
                "Usage: /llm on|off")
    app_settings.update({"llm_advisor_enabled": arg == "on"})
    if arg == "on" and not advisor.configured():
        return ("LLM advisor ENABLED — but no API key is set, so it stays "
                "inert. Put a Gemini key in llm_key.txt, or set "
                "CRYPTOMIND_LLM_KEY / GEMINI_API_KEY.")
    return f"LLM advisor {'ENABLED' if arg == 'on' else 'DISABLED'}."


def _cmd_mode(arg):
    from . import settings as app_settings
    if arg not in ("passive", "auto", "aggressive"):
        return "Usage: /mode passive|auto|aggressive"
    app_settings.update({"trade_mode": arg})
    return f"Trade mode set to *{arg}*."


def _cmd_notify(arg):
    if arg not in ("on", "off"):
        return "Usage: /notify on|off"
    save_conf({"push_trades": arg == "on"})
    return f"Per-trade notifications {'ON' if arg == 'on' else 'OFF'}."


def _cmd_set(args):
    from . import tunables
    if len(args) < 2:
        return "Usage: /set <tunable> <value>   (see /tunables)"
    key, val = args[0], args[1]
    if key not in tunables.TUNABLES:
        return f"Unknown tunable `{key}`. Use /tunables to list them."
    try:
        tunables.update({key: float(val)})
    except ValueError:
        return f"`{val}` is not a number."
    return f"Set `{key}` = {tunables.tv(key)}"


def _cmd_get(args):
    from . import tunables
    if not args or args[0] not in tunables.TUNABLES:
        return "Usage: /get <tunable>   (see /tunables)"
    k = args[0]
    m = tunables.TUNABLES[k]
    return f"`{k}` = {tunables.tv(k)}  (min {m['min']}, max {m['max']}) — {m['label']}"


def _cmd_tunables():
    from . import tunables
    out = ["*Tunables* (use /set <name> <value>)"]
    for k in tunables.TUNABLES:
        out.append(f"`{k}` = {tunables.tv(k)}")
    return "\n".join(out)


def _cmd_replay(args):
    """Latest automatic replay (per-coin best/worst), or start one with 'now'."""
    import asyncio
    from .orchestrator import orch
    if args and args[0].lower() == "now":
        try:
            asyncio.get_running_loop().create_task(orch.run_replay_now(reason="manual (Telegram)"))
            return "Replay started over the whole universe — results arrive as an alert in ~1 min."
        except RuntimeError:
            return "Can't start a replay from here (no running event loop)."
    rep = getattr(orch, "last_replay", None)
    if not rep or not (rep.get("full") or {}).get("ok"):
        return "No strategy replay yet. Send /replay now to run one."
    f, h1, h2 = rep["full"], rep.get("first_half") or {}, rep.get("second_half") or {}
    out = [f"*Strategy replay* — {rep.get('verdict')}",
           f"{len(rep.get('universe') or [])} coins, {f['days']} days, {rep.get('reason')}",
           f"Return {f['return_pct']:+.2f}% · max DD {f['max_drawdown_pct']}% · "
           f"{f['trades']} trades · win {f['win_rate_pct']}%",
           f"Halves {h1.get('return_pct')}% / {h2.get('return_pct')}% · "
           f"buy&hold {f.get('buy_hold_equal_weight_pct')}%"]
    try:
        ft = orch.forward_test()
    except Exception:
        ft = None
    if ft and ft.get("return_pct") is not None:
        out.append(f"Live since restart ({ft['days']} d): {ft['return_pct']:+.2f}% "
                   f"vs buy&hold {ft['buy_hold_pct']:+.2f}%")
    from .risk.manager import risk
    brake = getattr(risk, "replay_brake", 1.0)
    out.append(f"Risk brake: ON — new positions at {brake:.0%} size" if brake < 1.0
               else "Risk brake: off")
    try:
        from .learn.trade_filter import trade_filter
        snap = trade_filter.snapshot()
        if snap["trained"]:
            out.append(f"Trade filter: {'ACTIVE' if snap['active'] else 'inactive'} "
                       f"({snap['backend']}, {snap['train_samples']:,} samples, "
                       f"base win rate {snap['base_win_rate']:.0%})")
    except Exception:
        pass
    pp = sorted(((p, r) for p, r in (rep.get("per_product") or {}).items() if r["trades"]),
                key=lambda kv: kv[1]["net_usd"])
    if pp:
        out.append("Worst: " + ", ".join(f"{p} ${r['net_usd']:+,.0f}" for p, r in pp[:3]))
        out.append("Best: " + ", ".join(f"{p} ${r['net_usd']:+,.0f}" for p, r in pp[-3:][::-1]))
    return "\n".join(out)


def handle_command(text):
    """Parse and execute one command; return the reply text."""
    parts = text.strip().split()
    if not parts:
        return None
    cmd = parts[0].lstrip("/").lower()
    # strip @BotName suffix Telegram adds in groups
    cmd = cmd.split("@")[0]
    args = parts[1:]
    try:
        if cmd in ("start", "help"):
            return HELP_TEXT
        if cmd == "status":
            return _cmd_status()
        if cmd == "positions" or cmd == "pos":
            return _cmd_positions()
        if cmd == "trades":
            n = int(args[0]) if args and args[0].isdigit() else 5
            return _cmd_trades(min(n, 20))
        if cmd == "pnl":
            return _cmd_pnl()
        if cmd == "balance" or cmd == "bal":
            return _cmd_balance()
        if cmd == "stats":
            return _cmd_stats()
        if cmd in ("brains", "learning", "learn"):
            return _cmd_brains()
        if cmd in ("replay", "backtest"):
            return _cmd_replay(args)
        if cmd == "signals":
            return _cmd_signals()
        if cmd in ("diag", "diagnose", "why_not"):
            return _cmd_diag()
        if cmd in ("decisions", "audit", "trail"):
            n = int(args[0]) if args and args[0].isdigit() else 6
            return _cmd_decisions(n)
        if cmd == "chart":
            # handled asynchronously in command_worker (image send); this path
            # is only hit if called synchronously — return a hint.
            return "📊 Generating chart… (if you don't get an image, ensure " \
                   "there's equity history and the bot token is set)."
        if cmd == "risk":
            return _cmd_risk()
        if cmd == "why":
            return _cmd_why()
        if cmd == "pause":
            return _cmd_pause()
        if cmd == "resume":
            return _cmd_resume()
        if cmd == "kill":
            return _cmd_kill()
        if cmd in ("close", "flatten"):
            return _cmd_close(args)
        if cmd in ("resetkill", "reset_kill", "unkill"):
            return _cmd_resetkill()
        if cmd == "shorts":
            return _cmd_shorts(args[0] if args else "")
        if cmd == "llm":
            return _cmd_llm(args[0] if args else "")
        if cmd == "mode":
            return _cmd_mode(args[0] if args else "")
        if cmd == "notify":
            return _cmd_notify(args[0] if args else "")
        if cmd == "set":
            return _cmd_set(args)
        if cmd == "get":
            return _cmd_get(args)
        if cmd == "tunables":
            return _cmd_tunables()
        return f"Unknown command `/{cmd}`. Send /help for the list."
    except Exception as e:
        db.log_event("warn", f"Telegram command error ({cmd}): {e}")
        return f"⚠️ Error running /{cmd}: {e}"


async def command_worker():
    """Long-poll Telegram for operator commands (private to the chat_id)."""
    global _commands_active
    _load_conf()
    offset = None
    async with httpx.AsyncClient() as client:
        while True:
            tok = _state.get("telegram_bot_token")
            chat = _state.get("telegram_chat_id")
            if not (tok and chat):
                await asyncio.sleep(10)      # not configured yet; wait & retry
                continue
            _commands_active = True
            try:
                params = {"timeout": 50}
                if offset is not None:
                    params["offset"] = offset
                r = await client.get(
                    f"https://api.telegram.org/bot{tok}/getUpdates",
                    params=params, timeout=60)
                data = r.json()
                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message") or upd.get("edited_message")
                    if not msg:
                        continue
                    frm = msg.get("chat", {}).get("id")
                    txt = msg.get("text", "")
                    if not _authorized(frm):
                        # ignore strangers, but let them know it's private
                        await send_telegram(
                            "⛔ Unauthorized. This is a private control bot.",
                            client=client, chat_id=frm)
                        continue
                    if not txt:
                        continue
                    # /chart needs an async image send; handle it inline
                    cmd0 = txt.strip().split()[0].lstrip("/").lower().split("@")[0]
                    if cmd0 == "chart":
                        parts = txt.strip().split()
                        tf = parts[1] if len(parts) > 1 else "1d"
                        png, cap = render_equity_png(tf)
                        if png:
                            await send_telegram_photo(png, cap, client=client,
                                                      chat_id=frm)
                        else:
                            await send_telegram(cap, client=client, chat_id=frm)
                        continue
                    reply = handle_command(txt)
                    if reply:
                        await send_telegram(reply, client=client, chat_id=frm)
            except Exception as e:
                db.log_event("warn", f"Telegram command poll error: {e}")
                await asyncio.sleep(5)
