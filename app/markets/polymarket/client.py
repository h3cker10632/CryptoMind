"""Polymarket read-only market-data client (Gamma + CLOB).

This is the ONLY place that talks to Polymarket's public HTTP APIs, and it is
strictly READ-ONLY — quotes, order books and resolution status. No signing, no
orders, no credentials. (Order placement lives behind an explicit, disabled-by-
default seam in `execution.py`.)

Two upstreams, both public and keyless:
  * Gamma  (https://gamma-api.polymarket.com) — market metadata + last prices,
    liquidity, volume, resolution flags. One call returns everything we need to
    price and rank a batch of markets, which keeps us well under any rate limit.
  * CLOB   (https://clob.polymarket.com) — live top-of-book / midpoint for a
    single outcome token, used to refine a quote right before a (paper) fill.

Every network path fails SOFT: on any error we return an empty result and stash
the reason on `last_error`, so a dropped connection degrades the dashboard to
"no markets" instead of crashing the trading loop. All floats are coerced and
NaN/Inf are scrubbed at the boundary (Starlette serializes with allow_nan=False).
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import httpx

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
_UA = {"User-Agent": "CryptoMind/1.0 (+paper-research)"}
_TIMEOUT = 15.0


def _f(v, default=0.0):
    """Coerce to a finite float, else `default`."""
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _jload(v, default):
    """Gamma returns several array fields as JSON-encoded strings."""
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str) and v:
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return default
    return default


def _ttl_hours(end_iso: str) -> float | None:
    """Hours from now until the market's scheduled end (None if unparseable)."""
    if not end_iso:
        return None
    s = str(end_iso).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - datetime.now(timezone.utc)).total_seconds() / 3600.0


def _category(m: dict) -> str:
    """Best-effort category label from the market's parent event."""
    evs = m.get("events") or []
    if evs and isinstance(evs, list):
        ev = evs[0] or {}
        for k in ("category", "seriesSlug", "ticker", "slug"):
            v = ev.get(k)
            if v:
                # 'nfl-phi-chi-2026-09-29' -> 'nfl'
                return str(v).split("-")[0].lower()
    slug = m.get("slug") or ""
    return str(slug).split("-")[0].lower() if slug else "other"


def normalize_market(m: dict) -> dict | None:
    """Fold one raw Gamma market into the compact shape the engine trades on.

    Returns None for markets we can never trade on paper (not binary, no CLOB
    token ids, or order book disabled).
    """
    outcomes = _jload(m.get("outcomes"), [])
    tokens = _jload(m.get("clobTokenIds"), [])
    prices = _jload(m.get("outcomePrices"), [])
    if len(outcomes) != 2 or len(tokens) != 2:
        return None                       # MVP: binary YES/NO markets only
    prices = [_f(p, 0.5) for p in prices] or [0.5, 0.5]
    if len(prices) != 2:
        prices = [0.5, 0.5]
    ttl = _ttl_hours(m.get("endDate") or m.get("endDateIso") or "")
    return {
        "condition_id": m.get("conditionId") or "",
        "question": m.get("question") or "",
        "slug": m.get("slug") or "",
        "category": _category(m),
        "outcomes": [str(o) for o in outcomes],
        "token_ids": [str(t) for t in tokens],
        "prices": prices,                 # implied prob per outcome (sum ~1)
        "best_bid": _f(m.get("bestBid"), prices[0]),
        "best_ask": _f(m.get("bestAsk"), prices[0]),
        "spread": _f(m.get("spread"), 0.0),
        "last_trade_price": _f(m.get("lastTradePrice"), prices[0]),
        "liquidity": _f(m.get("liquidityNum") or m.get("liquidity"), 0.0),
        "volume_24h": _f(m.get("volume24hr"), 0.0),
        "tick_size": _f(m.get("orderPriceMinTickSize"), 0.01) or 0.01,
        "min_order_size": _f(m.get("orderMinSize"), 5.0) or 5.0,
        "mom_1h": _f(m.get("oneHourPriceChange"), 0.0),
        "mom_1d": _f(m.get("oneDayPriceChange"), 0.0),
        "ttl_hours": ttl,
        "end_date": m.get("endDate") or "",
        "active": bool(m.get("active")),
        "closed": bool(m.get("closed")),
        "accepting_orders": bool(m.get("acceptingOrders")),
        "fee_type": m.get("feeType") or "",
        "enable_order_book": bool(m.get("enableOrderBook")),
    }


class PolymarketClient:
    def __init__(self):
        self.last_error = ""
        self.last_fetch_ts = 0.0

    # ---------------------------- Gamma ----------------------------
    def _get(self, base, path, params=None):
        with httpx.Client(timeout=_TIMEOUT, headers=_UA) as c:
            r = c.get(f"{base}{path}", params=params)
            r.raise_for_status()
            return r.json()

    def fetch_markets(self, limit=40, min_liquidity=0.0,
                      order="volume24hr") -> list[dict]:
        """Top active, open, order-book-enabled binary markets, most-liquid first.

        Only markets currently `accepting_orders` are returned — a market that
        has stopped accepting orders can't be (paper-)traded honestly.
        """
        try:
            raw = self._get(GAMMA_BASE, "/markets", {
                "limit": max(1, min(int(limit) * 3, 300)),
                "active": "true", "closed": "false",
                "order": order, "ascending": "false",
            })
            self.last_error = ""
            self.last_fetch_ts = time.time()
        except Exception as e:                       # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            return []
        out = []
        for m in raw if isinstance(raw, list) else []:
            nm = normalize_market(m)
            if not nm:
                continue
            if not (nm["enable_order_book"] and nm["accepting_orders"]):
                continue
            if nm["liquidity"] < min_liquidity:
                continue
            out.append(nm)
            if len(out) >= int(limit):
                break
        return out

    def fetch_by_condition(self, condition_id: str) -> dict | None:
        """One market by conditionId — used to check resolution of a held bet."""
        if not condition_id:
            return None
        try:
            raw = self._get(GAMMA_BASE, "/markets",
                            {"condition_ids": condition_id})
        except Exception as e:                       # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        rows = raw if isinstance(raw, list) else [raw]
        for m in rows:
            nm = normalize_market(m)
            if nm and nm["condition_id"] == condition_id:
                return nm
        return None

    def resolution(self, condition_id: str) -> dict | None:
        """Resolution status for a market.

        Returns {resolved: bool, winning_index: int|None, prices: [..]} — a
        binary market is resolved once it is `closed` and one outcome's price
        has collapsed to ~1.0 (the winner) / ~0.0 (the loser).
        """
        nm = self.fetch_by_condition(condition_id)
        if not nm:
            return None
        prices = nm["prices"]
        resolved = nm["closed"] and (max(prices) >= 0.99 or min(prices) <= 0.01)
        win = None
        if resolved:
            win = 0 if prices[0] >= prices[1] else 1
        return {"resolved": resolved, "winning_index": win,
                "prices": prices, "closed": nm["closed"]}

    # ---------------------------- CLOB -----------------------------
    def clob_price(self, token_id: str, side: str) -> float | None:
        """Live best price to trade one outcome token: side='buy' (ask) or
        'sell' (bid). Returns None on any error (caller falls back to Gamma)."""
        try:
            d = self._get(CLOB_BASE, "/price",
                          {"token_id": token_id, "side": side})
            return _f(d.get("price"), None)
        except Exception:                            # noqa: BLE001
            return None

    def clob_midpoint(self, token_id: str) -> float | None:
        try:
            d = self._get(CLOB_BASE, "/midpoint", {"token_id": token_id})
            return _f(d.get("mid"), None)
        except Exception:                            # noqa: BLE001
            return None


client = PolymarketClient()
