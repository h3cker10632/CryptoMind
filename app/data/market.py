"""Market Data Feed — live candles, tickers and order-book stats from
Coinbase Exchange public REST API (no keys required)."""
import asyncio, time
import httpx
from ..config import PRODUCTS, CANDLE_GRANULARITY, CANDLE_HISTORY, MARKET_POLL_SEC
from .. import db
from .features import features_from_ohlcv

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
        """Compute technical features for one product.

        The OHLCV-derivable core comes from the SHARED helper
        (app/data/features.features_from_ohlcv) so the live feed and the
        backtester compute identical arithmetic (train/live parity). This method
        then layers the LIVE-only additions on top: the higher-timeframe swing
        ATR used for sizing, the multi-timeframe trend context, and the
        order-book imbalance/spread.
        """
        cs = self.candles.get(p, [])
        if len(cs) < 60:
            return None
        closes = [c[4] for c in cs]
        highs = [c[2] for c in cs]
        lows = [c[1] for c in cs]
        vols = [c[5] for c in cs]

        feat = features_from_ohlcv(closes, highs, lows, vols)
        if feat is None:
            return None

        # ATR(14) on an aggregated HIGHER-TIMEFRAME bar (default 1h = 12x5m).
        # Stops / targets / trailing are sized off THIS so a swing trade can
        # clear round-trip costs at its natural horizon. Sizing off the 5m ATR
        # made every trade a scalp whose target could never honestly beat fees;
        # the fix is the HORIZON, not looser fees. Configurable via the
        # `swing_atr_bars` tunable (12 = 1h, 3 = 15m, 1 = native 5m).
        feat["atr_swing"] = self._swing_atr(highs, lows, closes)

        # order-book imbalance/spread (live only; 0.0 placeholders in backtest)
        book = self.books.get(p, {})
        feat["imbalance"] = book.get("imbalance", 0.0)
        feat["spread_bps"] = book.get("spread_bps", 0.0)

        # MULTI-TIMEFRAME context (Tier-2): aggregate the native 5m bars into
        # 15m / 1h / 4h bars and read the trend + RSI on each. A trade with
        # several timeframes aligned is a far cleaner signal than one 5m read;
        # `mtf_align` in [-1,1] is the mean trend agreement across timeframes,
        # exposed both to the strategies (trend confirmation) and the ML model.
        mtf = self._mtf(highs, lows, closes)
        feat["mtf_align"] = mtf["align"]
        feat["mtf_trend_15m"] = mtf["t15"]
        feat["mtf_trend_1h"] = mtf["t1h"]
        feat["mtf_trend_4h"] = mtf["t4h"]
        feat["mtf_rsi_1h"] = mtf["rsi_1h"]

        # CHART-PATTERN analysis (live-only, like atr_swing/mtf_*): recognise
        # market structure, S/R, reversal/continuation patterns, candlesticks
        # and RSI divergence. Feeds the `pattern` strategy sleeve and the ML
        # model (build_x reads feat["patterns"]["features"]). Best-effort.
        opens = [c[3] for c in cs]
        try:
            from ..signals import patterns
            feat["patterns"] = patterns.analyze(highs, lows, closes, vols, opens)
        except Exception:
            feat["patterns"] = None
        return feat

    @staticmethod
    def _mtf(highs, lows, closes):
        """Multi-timeframe trend/RSI from aggregated 5m bars.

        For each timeframe (15m=3, 1h=12, 4h=48 native 5m bars) we fold the
        trailing bars into higher-TF closes, then read the trend as the sign of
        a fast-vs-slow EMA cross on that series. `align` is the mean of the
        per-timeframe trend signs, in [-1, 1]: +1 = every timeframe bullish,
        -1 = every timeframe bearish, ~0 = conflicted. Degrades gracefully to
        whatever timeframes there is history for.
        """
        def fold_closes(n):
            # one higher-TF close per group of n native bars (group's last close)
            if n <= 1:
                return list(closes)
            out = []
            # align groups to the most recent bar
            total = len(closes)
            start = total % n
            i = start
            while i + n <= total:
                out.append(closes[i + n - 1])
                i += n
            return out

        def ema(xs, n):
            if len(xs) < n:
                return None
            k = 2 / (n + 1); e = xs[-n]
            for x in xs[-n + 1:]:
                e = x * k + e * (1 - k)
            return e

        def trend_sign(n, fast=5, slow=12):
            cs = fold_closes(n)
            ef, es = ema(cs, fast), ema(cs, slow)
            if ef is None or es is None:
                return 0.0
            if ef > es * 1.001:
                return 1.0
            if ef < es * 0.999:
                return -1.0
            return 0.0

        def rsi_tf(n, period=14):
            cs = fold_closes(n)
            if len(cs) < period + 1:
                return 50.0
            gains = losses = 0.0
            for a, b in zip(cs[-period - 1:-1], cs[-period:]):
                d = b - a
                gains += max(d, 0.0); losses += max(-d, 0.0)
            ag, al = gains / period, losses / period
            return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)

        t15, t1h, t4h = trend_sign(3), trend_sign(12), trend_sign(48)
        align = (t15 + t1h + t4h) / 3.0
        return {"t15": t15, "t1h": t1h, "t4h": t4h, "align": align,
                "rsi_1h": rsi_tf(12)}

    @staticmethod
    def _swing_atr(highs, lows, closes, period=14):
        """ATR(14) computed on a HIGHER-TIMEFRAME bar aggregated from the native
        5m candles, so stop/target distances reflect a swing horizon (e.g. 1h)
        rather than a 5m scalp. `swing_atr_bars` 5m candles are folded into one
        swing bar (high=max, low=min, close=last); ATR is the mean true range
        over the last `period` swing bars. Falls back to the native 5m ATR when
        there isn't enough history yet.
        """
        from ..tunables import tv
        try:
            n = int(tv("swing_atr_bars"))
        except Exception:
            n = 12
        n = max(1, n)
        # native 5m ATR fallback (also the n==1 case)
        def _native():
            trs = []
            for i in range(-period, 0):
                trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                               abs(lows[i] - closes[i - 1])))
            return sum(trs) / period
        if n == 1:
            return _native()
        # fold the trailing 5m bars into (period+1) higher-timeframe bars
        need = (period + 1) * n
        if len(closes) < need:
            return _native()
        H, L, C = [], [], []
        for j in range(period + 1):
            seg_hi = highs[-(j + 1) * n: len(highs) - j * n or None]
            seg_lo = lows[-(j + 1) * n: len(lows) - j * n or None]
            seg_cl = closes[-(j + 1) * n: len(closes) - j * n or None]
            if not seg_hi:
                return _native()
            H.append(max(seg_hi)); L.append(min(seg_lo)); C.append(seg_cl[-1])
        H.reverse(); L.reverse(); C.reverse()      # oldest -> newest
        trs = []
        for i in range(1, len(C)):
            trs.append(max(H[i] - L[i], abs(H[i] - C[i - 1]),
                           abs(L[i] - C[i - 1])))
        return sum(trs) / len(trs) if trs else _native()

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
