"""Market Data Feed — live candles, tickers and order-book stats from
Coinbase Exchange public REST API (no keys required)."""
import asyncio, time, math, statistics
import httpx
from ..config import PRODUCTS, CANDLE_GRANULARITY, CANDLE_HISTORY, MARKET_POLL_SEC
from .. import db

BASE = "https://api.exchange.coinbase.com"


class MarketData:
    def __init__(self):
        self.candles = {p: [] for p in PRODUCTS}   # [ts, low, high, open, close, volume]
        self.tickers = {}                          # product -> {price, bid, ask, ts}
        self.books = {}                            # product -> {bid_depth, ask_depth, imbalance, spread_bps}
        self.last_update = 0.0
        self.healthy = False

    # ---------- fetch helpers ----------
    async def _get(self, client, path, params=None):
        r = await client.get(BASE + path, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    async def refresh_product(self, client, p):
        candles = await self._get(client, f"/products/{p}/candles",
                                  {"granularity": CANDLE_GRANULARITY})
        candles.sort(key=lambda c: c[0])
        self.candles[p] = candles[-CANDLE_HISTORY:]

        t = await self._get(client, f"/products/{p}/ticker")
        self.tickers[p] = {
            "price": float(t["price"]), "bid": float(t["bid"]),
            "ask": float(t["ask"]), "ts": time.time(),
        }

        book = await self._get(client, f"/products/{p}/book", {"level": 2})
        bids = book.get("bids", [])[:25]
        asks = book.get("asks", [])[:25]
        bid_depth = sum(float(b[0]) * float(b[1]) for b in bids)
        ask_depth = sum(float(a[0]) * float(a[1]) for a in asks)
        tot = bid_depth + ask_depth
        mid = (float(bids[0][0]) + float(asks[0][0])) / 2 if bids and asks else 0
        spread_bps = ((float(asks[0][0]) - float(bids[0][0])) / mid * 1e4) if mid else 0
        self.books[p] = {
            "bid_depth": bid_depth, "ask_depth": ask_depth,
            "imbalance": (bid_depth - ask_depth) / tot if tot else 0.0,
            "spread_bps": spread_bps,
        }

    async def run(self):
        async with httpx.AsyncClient(headers={"User-Agent": "CryptoMind/1.0"}) as client:
            while True:
                try:
                    for p in list(PRODUCTS):          # dynamic universe
                        if p not in self.candles:
                            self.candles[p] = []
                            db.log_event("data", f"Market feed tracking new asset: {p}")
                        await self.refresh_product(client, p)
                        await asyncio.sleep(0.25)  # be polite / rate limits
                    # drop assets pruned from the universe (keep no stale state)
                    for p in list(self.candles):
                        if p not in PRODUCTS:
                            self.candles.pop(p, None)
                            self.tickers.pop(p, None)
                            self.books.pop(p, None)
                    self.last_update = time.time()
                    if not self.healthy:
                        db.log_event("data", "Market data feed healthy")
                    self.healthy = True
                except Exception as e:
                    was_healthy = self.healthy
                    self.healthy = False
                    db.log_event("error", f"Market data error: {e}")
                    if was_healthy:
                        from ..alerts import alert
                        alert("warning", "Market data feed DOWN",
                              f"Coinbase feed failing: {str(e)[:150]}. "
                              f"New entries blocked until recovery.")
                await asyncio.sleep(MARKET_POLL_SEC)

    # ---------- derived features ----------
    def price(self, p):
        t = self.tickers.get(p)
        return t["price"] if t else None

    def closes(self, p):
        return [c[4] for c in self.candles.get(p, [])]

    def features(self, p):
        """Compute technical features for one product."""
        cs = self.candles.get(p, [])
        if len(cs) < 60:
            return None
        closes = [c[4] for c in cs]
        highs = [c[2] for c in cs]
        lows = [c[1] for c in cs]
        vols = [c[5] for c in cs]
        price = closes[-1]

        def sma(xs, n): return sum(xs[-n:]) / n
        def ema(xs, n):
            k = 2 / (n + 1); e = xs[-n]
            for x in xs[-n + 1:]: e = x * k + e * (1 - k)
            return e

        # RSI(14)
        gains, losses = [], []
        for a, b in zip(closes[-15:-1], closes[-14:]):
            d = b - a
            gains.append(max(d, 0)); losses.append(max(-d, 0))
        avg_g, avg_l = sum(gains) / 14, sum(losses) / 14
        rsi = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)

        # ATR(14)
        trs = []
        for i in range(-14, 0):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        atr = sum(trs) / 14

        # MACD
        macd = ema(closes, 12) - ema(closes, 26)
        macd_prev = ema(closes[:-1], 12) - ema(closes[:-1], 26)

        rets = [math.log(b / a) for a, b in zip(closes[-61:-1], closes[-60:])]
        vol = statistics.stdev(rets) if len(rets) > 2 else 0.0

        hi20, lo20 = max(highs[-20:]), min(lows[-20:])
        vol_ratio = vols[-1] / (sum(vols[-20:]) / 20) if sum(vols[-20:]) else 1.0
        book = self.books.get(p, {})

        return {
            "price": price, "rsi": rsi, "atr": atr,
            "sma20": sma(closes, 20), "sma50": sma(closes, 50),
            "ema12": ema(closes, 12), "ema26": ema(closes, 26),
            "macd": macd, "macd_delta": macd - macd_prev,
            "volatility": vol, "hi20": hi20, "lo20": lo20,
            "vol_ratio": vol_ratio,
            "mom_1h": price / closes[-13] - 1 if len(closes) >= 13 else 0,
            "mom_4h": price / closes[-49] - 1 if len(closes) >= 49 else 0,
            "imbalance": book.get("imbalance", 0.0),
            "spread_bps": book.get("spread_bps", 0.0),
        }

    def regime(self):
        """Simple market regime detector from BTC."""
        f = self.features("BTC-USD")
        if not f:
            return {"label": "unknown", "vol": 0}
        trend = "bull" if f["sma20"] > f["sma50"] * 1.002 else \
                "bear" if f["sma20"] < f["sma50"] * 0.998 else "sideways"
        vol_state = "high-vol" if f["volatility"] > 0.004 else "normal"
        return {"label": f"{trend}/{vol_state}", "trend": trend,
                "vol_state": vol_state, "vol": f["volatility"]}


market = MarketData()
