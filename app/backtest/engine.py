"""Backtesting & Validation Suite — event-driven bar simulation over real
historical candles fetched from Coinbase, with fees/slippage, per-strategy
walk-forward split and overfitting sanity metrics."""
import math, statistics
import httpx
from ..tunables import tv

BASE = "https://api.exchange.coinbase.com"


async def fetch_history(product, granularity=3600, chunks=3):
    """Fetch up to ~900 hourly bars (Coinbase returns 300 per call)."""
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

        equity_curve.append((ts, cash + qty * cl))

    if qty > 0:
        cash += qty * candles[-1][4] * (1 - FEE_RATE)
        trades.append(candles[-1][4] / entry - 1)

    eq = [e for _, e in equity_curve]
    rets = [b / a - 1 for a, b in zip(eq[:-1], eq[1:]) if a > 0]
    total = eq[-1] / start_cash - 1 if eq else 0.0
    sharpe = (statistics.mean(rets) / statistics.stdev(rets) * math.sqrt(24 * 365)
              if len(rets) > 2 and statistics.stdev(rets) > 0 else 0.0)
    peak, maxdd = 0.0, 0.0
    for e in eq:
        peak = max(peak, e)
        maxdd = max(maxdd, 1 - e / peak)
    wins = [t for t in trades if t > 0]
    return {
        "total_return": round(total, 4),
        "sharpe_annualized": round(sharpe, 2),
        "max_drawdown": round(maxdd, 4),
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
