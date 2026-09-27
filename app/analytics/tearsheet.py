"""Honest performance tearsheet over REAL closed trades.

This does NO simulation and invents NO data: it reads the paper broker's realized
closed-trade log and reports the risk-adjusted truth of what actually happened —
overall, and sliced by the regime the trade was opened in and by how it exited.
That per-slice view is what turns "increase bear-regime weights" / "the exit
advisor is too aggressive" from vibes into measured decisions.

All the hard math (Sharpe, Sortino, profit factor, expectancy, max drawdown,
Calmar, Wilson win-rate bounds) is reused from app.backtest.stats — the same
vetted implementations the GA gates on — so the tearsheet and the promotion gate
speak the same statistical language.

Per-trade return basis: pnl / entry notional (qty*entry). Each closed trade is
one return observation; the realized equity curve (start cash + cumulative pnl,
ordered by close time) drives the drawdown/Calmar figures. Everything is net of
fees because the broker books pnl net of fees.
"""
import math
import time

from ..backtest import stats

_YEAR_SEC = 365.25 * 24 * 3600


def _trade_return(t):
    """Fractional return of one closed trade on its entry notional, or None if
    the notional can't be reconstructed."""
    entry = t.get("entry")
    qty = t.get("qty")
    if not entry or not qty:
        return None
    notional = abs(qty * entry)
    if notional <= 0:
        return None
    return t.get("pnl", 0.0) / notional


def _round(x, n=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None if x is None else (x if math.isinf(x) else None)
    return round(x, n)


def _block(trades, start_cash):
    """Full statistics block for a list of closed-trade dicts."""
    n = len(trades)
    if n == 0:
        return {"n": 0}
    ordered = sorted(trades, key=lambda t: t.get("closed", 0.0))
    pnls = [t.get("pnl", 0.0) for t in ordered]
    rets = [r for r in (_trade_return(t) for t in ordered) if r is not None]
    wins = [p for p in pnls if p > 0]

    # realized equity curve (cumulative pnl on the account), for drawdown/Calmar
    equity = [start_cash]
    for p in pnls:
        equity.append(equity[-1] + p)
    mdd = stats.max_drawdown(equity)

    # time span -> annualization factors from the REAL clock, not assumed bars
    first, last = ordered[0].get("closed", 0.0), ordered[-1].get("closed", 0.0)
    years = (last - first) / _YEAR_SEC if last > first else 0.0
    ann_ret = None
    if years > 0 and equity[0] > 0 and equity[-1] > 0:
        try:
            ann_ret = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0
        except (OverflowError, ValueError):
            ann_ret = None
    calmar = (ann_ret / mdd) if (ann_ret is not None and mdd > 0) else None
    trades_per_year = (n / years) if years > 0 else 0.0
    ann_sharpe = (stats.annualized_sharpe(rets, trades_per_year)
                  if trades_per_year > 0 and len(rets) > 1 else None)

    p, lo, hi = stats.wilson_interval(len(wins), n)
    tq = stats.trade_quality(rets)   # profit_factor/expectancy/win-loss shape on returns

    return {
        "n": n,
        "net_pnl": _round(sum(pnls), 2),
        "win_rate": p,
        "win_rate_ci95": [lo, hi],
        "profit_factor": tq["profit_factor"],
        "expectancy_pct": tq["expectancy"],           # mean per-trade return
        "avg_win_pct": tq["avg_win"],
        "avg_loss_pct": tq["avg_loss"],
        "win_loss_ratio": tq["win_loss_ratio"],
        "sharpe_per_trade": _round(stats.sharpe(rets)),
        "sortino_per_trade": _round(stats.sortino(rets)),
        "ann_sharpe": _round(ann_sharpe),
        "max_drawdown": _round(mdd),
        "max_dd_duration_trades": stats.max_drawdown_duration(equity),
        "calmar": _round(calmar),
        "annualized_return": _round(ann_ret),
        "best_trade_pnl": _round(max(pnls), 2),
        "worst_trade_pnl": _round(min(pnls), 2),
        "gross_entry_fees": _round(sum(t.get("fees", 0.0) for t in ordered), 2),
        "span_days": _round((last - first) / 86400.0, 1) if last > first else 0.0,
    }


def _mini_block(trades, start_cash):
    """Lighter block for the per-regime / per-exit-reason breakdowns."""
    n = len(trades)
    if n == 0:
        return {"n": 0}
    pnls = [t.get("pnl", 0.0) for t in trades]
    rets = [r for r in (_trade_return(t) for t in trades) if r is not None]
    wins = [p for p in pnls if p > 0]
    pf = stats.profit_factor(rets)
    return {
        "n": n,
        "net_pnl": _round(sum(pnls), 2),
        "win_rate": _round(len(wins) / n),
        "expectancy_pct": _round(stats.expectancy(rets)) if rets else None,
        "profit_factor": (round(pf, 4) if pf not in (None, math.inf) else pf),
        "sharpe_per_trade": _round(stats.sharpe(rets)),
        "avg_pnl": _round(sum(pnls) / n, 2),
    }


def _group_by(trades, key, start_cash):
    groups = {}
    for t in trades:
        groups.setdefault(t.get(key) or "unknown", []).append(t)
    # sort slices by net pnl (worst first surfaces the bleed)
    out = {k: _mini_block(v, start_cash) for k, v in groups.items()}
    return dict(sorted(out.items(), key=lambda kv: kv[1].get("net_pnl", 0.0)))


def _json_safe(o):
    """Recursively neutralize non-JSON-compliant floats (inf/nan -> None) so the
    endpoint can't 500 under Starlette's allow_nan=False (the same failure mode
    that broke the Invo study endpoint). inf profit_factor => 'no losing trades',
    which the small-sample caveat already flags."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def build_tearsheet(closed_trades, start_cash=100_000.0, exclude_hedge=False):
    """Assemble the full tearsheet dict from a list of closed-trade dicts.

    exclude_hedge: drop market-neutral hedge legs so the directional book's edge
    isn't blurred by the hedge sleeve (they're reported separately by count).
    """
    trades = list(closed_trades or [])
    n_hedge = sum(1 for t in trades if t.get("hedge"))
    if exclude_hedge:
        trades = [t for t in trades if not t.get("hedge")]

    return _json_safe({
        "generated_at": time.time(),
        "source": "paper broker realized closed trades (current session)",
        "basis": "net of fees; per-trade return = pnl / entry notional; "
                 "drawdown/Calmar on realized equity curve",
        "caveat": "session-only (in-memory closed trades reset on restart); "
                  "small samples are noisy — read the win-rate CI and trade count.",
        "start_cash": start_cash,
        "n_trades": len(trades),
        "n_hedge_legs": n_hedge,
        "overall": _block(trades, start_cash),
        "by_regime": _group_by(trades, "regime_at_entry", start_cash),
        "by_exit_reason": _group_by(trades, "exit_reason", start_cash),
    })
