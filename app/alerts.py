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
    "/signals — current actionable signals\n"
    "/risk — drawdown, daily loss, kill/halt state\n"
    "/why — why trading is killed/halted (if it is)\n"
    "/pause — stop opening new trades\n"
    "/resume — resume trading\n"
    "/kill — 🛑 flatten all & stop (kill switch)\n"
    "/resetkill — ♻️ clear the kill switch + daily halt\n"
    "/shorts on|off — allow/deny short positions\n"
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
        if cmd == "signals":
            return _cmd_signals()
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
        if cmd in ("resetkill", "reset_kill", "unkill"):
            return _cmd_resetkill()
        if cmd == "shorts":
            return _cmd_shorts(args[0] if args else "")
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
                    reply = handle_command(txt)
                    if reply:
                        await send_telegram(reply, client=client, chat_id=frm)
            except Exception as e:
                db.log_event("warn", f"Telegram command poll error: {e}")
                await asyncio.sleep(5)
