"""WebSocket market feed — Coinbase Exchange public `ticker` channel.

The critique flagged 30s REST polling as unable to manage live stops. This
adds a real-time price stream that keeps `market.tickers` fresh between REST
refreshes, with the lifecycle discipline a production feed needs:

  * heartbeat / staleness detection
  * reconnect with exponential backoff and re-subscribe on every reconnect
  * REST resync is handled by the existing `market.run()` loop (authoritative)

It updates the SAME `market` object the rest of the system reads, so stops,
signals and equity all see sub-second prices. REST remains the source of
truth for candles and order-book depth.
"""
from __future__ import annotations
import asyncio
import json
import time
import contextlib
from ..config import PRODUCTS
from .. import db

WS_URL = "wss://ws-feed.exchange.coinbase.com"


class WSMarket:
    def __init__(self, market):
        self.market = market
        self.connected = False
        self.last_msg = 0.0
        self.reconnects = 0
        self._subscribed = []

    async def run(self):
        try:
            import websockets  # optional dep; degrade gracefully if absent
        except Exception:
            db.log_event("data", "websockets package not installed — "
                                 "real-time stream disabled, REST polling only")
            return
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                        WS_URL, ping_interval=20, ping_timeout=15,
                        max_queue=256) as ws:
                    await self._subscribe(ws)
                    self.connected = True
                    self.reconnects += 1 if self.reconnects or True else 0
                    backoff = 1.0
                    db.log_event("data", "WebSocket ticker stream connected")
                    async for raw in ws:
                        self._on_message(raw)
                        # re-subscribe if the universe changed
                        if set(self._subscribed) != set(PRODUCTS):
                            await self._subscribe(ws)
            except Exception as e:
                self.connected = False
                db.log_event("data", f"WebSocket stream dropped: {str(e)[:120]} "
                                     f"— reconnecting in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    async def _subscribe(self, ws):
        self._subscribed = list(PRODUCTS)
        await ws.send(json.dumps({
            "type": "subscribe",
            "product_ids": self._subscribed,
            "channels": ["ticker", "heartbeat"],
        }))

    def _on_message(self, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        self.last_msg = time.time()
        if msg.get("type") != "ticker":
            return
        p = msg.get("product_id")
        price = msg.get("price")
        if not p or price is None or p not in PRODUCTS:   # left the universe: no ticks
            return
        try:
            px = float(price)
        except (TypeError, ValueError):
            return
        t = self.market.tickers.get(p, {})
        t["price"] = px
        if msg.get("best_bid"):
            t["bid"] = float(msg["best_bid"])
        if msg.get("best_ask"):
            t["ask"] = float(msg["best_ask"])
        t["ts"] = time.time()
        t["src"] = "ws"
        self.market.tickers[p] = t

    def stats(self):
        stale = time.time() - self.last_msg if self.last_msg else None
        return {"connected": self.connected, "reconnects": self.reconnects,
                "last_msg_age_sec": round(stale, 1) if stale is not None else None,
                "subscribed": self._subscribed}
