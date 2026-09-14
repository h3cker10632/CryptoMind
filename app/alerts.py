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
}

recent_alerts = []               # newest first, capped
_queue: asyncio.Queue = None
_dedupe = {}                     # title -> last sent ts (anti-spam)
DEDUPE_SEC = 900

LEVELS = {"info": 0, "warning": 1, "critical": 2}


def _load_conf():
    try:
        if os.path.exists(CONF_PATH):
            with open(CONF_PATH) as f:
                saved = json.load(f)
            for k in _state:
                if saved.get(k):
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


async def _push_telegram(client, entry):
    tok, chat = _state["telegram_bot_token"], _state["telegram_chat_id"]
    if not (tok and chat):
        return
    icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(entry["level"], "")
    text = f"{icon} *CryptoMind — {entry['title']}*\n{entry['message']}"
    try:
        r = await client.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "Markdown"},
            timeout=15)
        if r.status_code != 200:
            db.log_event("warn", f"Telegram push failed: {r.status_code} {r.text[:120]}")
    except Exception as e:
        db.log_event("warn", f"Telegram push error: {e}")


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
