"""Backtesting & Validation Suite — event-driven bar simulation over real
historical candles fetched from Coinbase, with fees/slippage, per-strategy
walk-forward split and overfitting sanity metrics.

Historical candles are CACHED (in-memory TTL + optional on-disk) because the
same OHLCV window is re-fetched constantly: every /api/backtest call, the
composite report, and the GA feed thread all pull the same bars. Hourly candles
only change once an hour, so a short TTL removes almost all of the 3x HTTP
round-trips per report and shrinks our Coinbase rate-limit exposure. (pybroker
caches downloaded data, indicators, and models for the same reason.)
"""
import math, statistics, time, os, json, threading
import httpx
from ..tunables import tv
from . import stats as st

BASE = "https://api.exchange.coinbase.com"

# ---- history cache -------------------------------------------------------
CACHE_TTL_SEC = 300          # hourly bars change at most once/hour; 5 min is safe
_CACHE = {}                  # key -> (fetched_at, candles)
_CACHE_LOCK = threading.Lock()   # GA runs fetch_history from a worker thread
_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), ".cache", "history")


def _cache_key(product, granularity, chunks):
    return f"{product}:{granularity}:{chunks}"


def _disk_path(key):
    return os.path.join(_DISK_CACHE_DIR, key.replace(":", "_") + ".json")


def _disk_read(key, ttl):
    """Return cached candles from disk if fresh, else None. Never raises."""
    try:
        p = _disk_path(key)
        if not os.path.exists(p):
            return None
        if time.time() - os.path.getmtime(p) > ttl:
            return None
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _disk_write(key, candles):
    """Persist candles to disk (best-effort; failures are swallowed)."""
    try:
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        tmp = _disk_path(key) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(candles, f)
        os.replace(tmp, _disk_path(key))
    except Exception:
        pass


def clear_history_cache():
    """Drop the in-memory history cache (used by tests / manual refresh)."""
    with _CACHE_LOCK:
        _CACHE.clear()


async def _fetch_history_raw(product, granularity=3600, chunks=3):
    """Fetch up to ~900 hourly bars (Coinbase returns 300 per call). No cache."""
    out = []
    async with httpx.AsyncClient(headers={"User-Agent": "CryptoMind/1.0"}) as c:
        end = None
        for _ in range(chunks):
            params = {"granularity": granularity}
            if end:
                params["end"] = end
                params["start"] = end - granularity * 300
            r = await c.get(f"{BASE}/products/{product}/candles", params=params, timeout=20)
            r.raise_for_status()
            batch = sorted(r.json(), key=lambda x: x[0])
            if not batch:
                break
            out = batch + out
            end = batch[0][0]
    # dedupe
    seen, res = set(), []
    for b in out:
        if b[0] not in seen:
            seen.add(b[0]); res.append(b)
    return sorted(res, key=lambda x: x[0])


async def fetch_history(product, granularity=3600, chunks=3,
                        use_cache=True, ttl=CACHE_TTL_SEC):
    """Cached wrapper around _fetch_history_raw.

    Lookup order: in-memory (fast, per-process) -> on-disk (survives restart) ->
    network. A successful network fetch backfills both layers. A network FAILURE
    falls back to any stale cached copy rather than propagating, so a transient
    Coinbase outage degrades gracefully instead of killing a backtest.
    """
    if not use_cache:
        return await _fetch_history_raw(product, granularity, chunks)

    key = _cache_key(product, granularity, chunks)
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] <= ttl:
            return hit[1]

    disk = _disk_read(key, ttl)
    if disk is not None:
        with _CACHE_LOCK:
            _CACHE[key] = (now, disk)
        return disk

    try:
        candles = await _fetch_history_raw(product, granularity, chunks)
    except Exception:
        # network failed — serve a stale copy if we have one anywhere
        with _CACHE_LOCK:
            stale = _CACHE.get(key)
        if stale is not None:
            return stale[1]
        stale_disk = _disk_read(key, ttl=math.inf)
        if stale_disk is not None:
            return stale_disk
        raise

    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), candles)
    _disk_write(key, candles)
    return candles


def _atr(candles, i, n=14):
    trs = []
    for j in range(i - n + 1, i + 1):
        hi, lo, pc = candles[j][2], candles[j][1], candles[j - 1][4]
        trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    return sum(trs) / n


def _ema(xs, n):
    k = 2 / (n + 1); e = xs[0]
    for x in xs[1:]:
        e = x * k + e * (1 - k)
    return e


def run_backtest(candles, strategy="trend", start_cash=10_000.0):
    """Long-only event-driven simulation on [ts, low, high, open, close, vol] bars."""
    cash, qty, entry, stop, take = start_cash, 0.0, 0.0, 0.0, 0.0
    equity_curve, trades = [], []
    bars_total = bars_in_market = 0        # exposure / time-in-market
    # read LIVE-tuned costs/risk so the backtester matches the running system
    FEE_RATE = tv("fee_rate")
    SLIPPAGE_BPS = tv("slippage_bps")
    STOP_ATR_MULT = tv("stop_atr_mult")
    TAKE_PROFIT_ATR_MULT = tv("take_profit_atr_mult")
    slip = SLIPPAGE_BPS / 1e4

    for i in range(60, len(candles)):
        ts, lo, hi, op, cl, vol = candles[i]
        closes = [c[4] for c in candles[i - 59:i + 1]]
        atr = _atr(candles, i)

        # --- exit checks first (intra-bar stop/target via low/high) ---
        if qty > 0:
            if lo <= stop:
                px = stop * (1 - slip)
                cash += qty * px * (1 - FEE_RATE)
                trades.append(px / entry - 1)
                qty = 0.0
            elif hi >= take:
                px = take * (1 - slip)
                cash += qty * px * (1 - FEE_RATE)
                trades.append(px / entry - 1)
                qty = 0.0

        # --- signal ---
        long_sig = False
        if strategy == "trend":
            long_sig = _ema(closes[-24:], 12) > _ema(closes[-48:], 26) and closes[-1] > closes[-13]
        elif strategy == "meanrev":
            gains = [max(b - a, 0) for a, b in zip(closes[-15:-1], closes[-14:])]
            losses = [max(a - b, 0) for a, b in zip(closes[-15:-1], closes[-14:])]
            ag, al = sum(gains) / 14, sum(losses) / 14
            rsi = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
            long_sig = rsi < 30
        elif strategy == "breakout":
            long_sig = cl >= max(c[2] for c in candles[i - 20:i]) and \
                       vol > sum(c[5] for c in candles[i - 20:i]) / 20 * 1.2

        if qty == 0 and long_sig and atr > 0:
            px = cl * (1 + slip)
            notional = cash * 0.95
            qty = notional / px
            cash -= notional * (1 + FEE_RATE)
            entry, stop, take = px, px - STOP_ATR_MULT * atr, px + TAKE_PROFIT_ATR_MULT * atr

        bars_total += 1
        if qty > 0:
            bars_in_market += 1
        equity_curve.append((ts, cash + qty * cl))

    if qty > 0:
        cash += qty * candles[-1][4] * (1 - FEE_RATE)
        trades.append(candles[-1][4] / entry - 1)

    eq = [e for _, e in equity_curve]
    rets = [b / a - 1 for a, b in zip(eq[:-1], eq[1:]) if a > 0]
    total = eq[-1] / start_cash - 1 if eq else 0.0
    PPY = 24 * 365
    wins = [t for t in trades if t > 0]
    tq = st.trade_quality(trades)
    ann = st.annualized_return(eq, PPY)
    cal = st.calmar(eq, PPY)
    return {
        "total_return": round(total, 4),
        "sharpe_annualized": round(st.annualized_sharpe(rets, PPY), 2),
        "sortino_annualized": round(st.sortino(rets) * math.sqrt(PPY), 2),
        "max_drawdown": round(st.max_drawdown(eq), 4),
        "max_drawdown_bars": st.max_drawdown_duration(eq),
        "annualized_return": round(ann, 4) if ann is not None else None,
        "calmar": round(cal, 3) if cal is not None else None,
        "profit_factor": tq["profit_factor"],
        "expectancy": tq["expectancy"],
        "avg_win": tq["avg_win"], "avg_loss": tq["avg_loss"],
        "win_loss_ratio": tq["win_loss_ratio"],
        "exposure": round(bars_in_market / bars_total, 4) if bars_total else 0.0,
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3) if trades else None,
        "final_equity": round(eq[-1], 2) if eq else start_cash,
        "equity_curve": [(t, round(e, 2)) for t, e in equity_curve[::max(1, len(equity_curve)//200)]],
    }


async def full_report(product, strategy):
    candles = await fetch_history(product)
    if len(candles) < 200:
        return {"error": f"insufficient history for {product}"}
    split = int(len(candles) * 0.6)
    in_sample = run_backtest(candles[:split], strategy)
    out_sample = run_backtest(candles[split - 60:], strategy)
    overfit = None
    if in_sample["n_trades"] and out_sample["n_trades"]:
        overfit = bool(in_sample["total_return"] > 0 > out_sample["total_return"])
    return {
        "product": product, "strategy": strategy,
        "bars": len(candles), "granularity": "1h",
        "in_sample": {k: v for k, v in in_sample.items() if k != "equity_curve"},
        "out_of_sample": {k: v for k, v in out_sample.items() if k != "equity_curve"},
        "full": run_backtest(candles, strategy),
        "walk_forward_degradation_flag": overfit,
    }
