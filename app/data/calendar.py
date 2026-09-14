"""Macro Economic Calendar — ForexFactory public JSON (key-free).

Implements the blueprint's "news blackout periods": around high-impact USD
macro events (CPI, FOMC, NFP, PPI, GDP...), new entries are blocked and the
system reports a blackout state. Positions already open keep their normal
stop/target management — the gate only stops NEW risk being added.

Blackout window: 30 min before → 45 min after each event (FOMC gets a wider
window: 45 min before → 90 min after, statement + press conference).
"""
import asyncio, time
from datetime import datetime
import httpx
from .. import db

URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
POLL_SEC = 3600 * 6         # calendar changes rarely; refresh every 6h
RETRY_SEC = 600             # retry sooner while unhealthy (e.g. after a 429)

PRE_SEC = 30 * 60
POST_SEC = 45 * 60
FOMC_PRE_SEC = 45 * 60
FOMC_POST_SEC = 90 * 60

FOMC_KEYWORDS = ("fomc", "federal funds rate", "fed chair", "fed press")


def _parse_ts(datestr):
    """Parse ISO timestamps like 2026-09-11T08:30:00-04:00."""
    try:
        return datetime.fromisoformat(datestr).timestamp()
    except Exception:
        return None


class MacroCalendar:
    def __init__(self):
        self.events = []          # [{title, ts, impact, is_fomc}]
        self.last_update = 0.0
        self.healthy = False

    async def refresh(self, client):
        events = []
        for url in URLS:
            try:
                r = await client.get(url, timeout=20)
                r.raise_for_status()
                for e in r.json():
                    if e.get("country") != "USD":
                        continue
                    if e.get("impact") not in ("High",):
                        continue
                    ts = _parse_ts(e.get("date", ""))
                    if not ts:
                        continue
                    title = e.get("title", "")
                    events.append({
                        "title": title,
                        "ts": ts,
                        "impact": e.get("impact"),
                        "is_fomc": any(k in title.lower() for k in FOMC_KEYWORDS),
                    })
            except httpx.HTTPStatusError as ex:
                code = ex.response.status_code
                if code == 404:
                    pass          # nextweek.json often doesn't exist — fine
                elif code == 429:
                    db.log_event("warn", "Macro calendar rate-limited (429) — "
                                         f"will retry in {RETRY_SEC // 60} min")
                else:
                    db.log_event("warn", f"Macro calendar fetch failed ({code})")
            except Exception as ex:
                db.log_event("warn", f"Macro calendar fetch failed: {ex}")
        if events:
            # dedupe same title+ts
            seen, out = set(), []
            for e in sorted(events, key=lambda x: x["ts"]):
                k = (e["title"], e["ts"])
                if k not in seen:
                    seen.add(k)
                    out.append(e)
            self.events = out
            self.healthy = True
            self.last_update = time.time()
            upcoming = [e for e in out if e["ts"] > time.time()]
            db.log_event("data", f"Macro calendar loaded: {len(out)} high-impact "
                                 f"USD events this+next week ({len(upcoming)} upcoming)")

    async def run(self):
        async with httpx.AsyncClient(headers={"User-Agent": "CryptoMind/1.0"}) as client:
            while True:
                await self.refresh(client)
                await asyncio.sleep(POLL_SEC if self.healthy else RETRY_SEC)

    # ---------------- blackout logic ----------------
    def blackout(self):
        """Return (active, event) — active if now is inside any event window."""
        now = time.time()
        for e in self.events:
            pre = FOMC_PRE_SEC if e["is_fomc"] else PRE_SEC
            post = FOMC_POST_SEC if e["is_fomc"] else POST_SEC
            if e["ts"] - pre <= now <= e["ts"] + post:
                return True, e
        return False, None

    def next_event(self):
        now = time.time()
        for e in self.events:
            if e["ts"] > now:
                return e
        return None

    def stats(self):
        active, ev = self.blackout()
        nxt = self.next_event()
        return {
            "blackout_active": active,
            "blackout_event": ev,
            "next_event": nxt,
            "next_event_in_min": round((nxt["ts"] - time.time()) / 60) if nxt else None,
            "upcoming": [e for e in self.events if e["ts"] > time.time()][:8],
            "healthy": self.healthy,
            "last_update": self.last_update,
        }


calendar = MacroCalendar()
