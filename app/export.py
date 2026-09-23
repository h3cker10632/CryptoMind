"""Full-state export — aggregate EVERY data surface into ONE JSON document.

Two ways to get it:
  * live, from the running server:  GET /api/export           (in-memory state)
  * offline, from the CLI:          python -m app.export out.json
    (rebuilds state from the saved snapshot + SQLite, no server needed)

The dashboard's ~24 read endpoints each expose one slice; this stitches them all
into a single object so the operator can grab the WHOLE picture in one file. On
top of the raw subsystem snapshots this module also COMPUTES derived analytics
that no single endpoint offers: portfolio risk/return stats (Sharpe, Sortino,
profit factor, expectancy, max drawdown, streaks), a per-product trade breakdown,
a per-strategy scoring/attribution breakdown, and a consolidated per-product view
that fuses price/features/derivatives/sentiment/universe into one object.

Every section is isolated in a try/except: a failing subsystem records an "error"
string for that key instead of breaking the whole dump. History depth is capped
per section (and overridable) so the file stays a reasonable size, but by default
the export is EXHAUSTIVE.
"""
import asyncio
import json
import math
import os
import time

# Where periodic auto-exports are written. Kept out of git (see .gitignore).
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


# ------------------------------------------------------------------ utilities
def _safe(fn, default=None):
    """Run a section producer; never let one failure sink the whole export."""
    try:
        return fn()
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"{type(e).__name__}: {e}"}


def _stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


# ------------------------------------------------------------ derived analytics
def _trade_analytics(closed):
    """Rich performance stats computed from the closed-trade ledger — the kind
    of summary no single endpoint returns."""
    n = len(closed)
    if not n:
        return {"trades": 0, "note": "no closed trades yet"}
    pnls = [t.get("pnl", 0.0) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    total = sum(pnls)

    # holding period (seconds) where both stamps exist
    holds = [t["closed"] - t["opened"] for t in closed
             if t.get("closed") and t.get("opened")]

    # win/loss streaks + max drawdown over the realized-PnL equity path
    cum, peak, max_dd = 0.0, 0.0, 0.0
    cur_streak = best_win_streak = worst_loss_streak = 0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        if p > 0:
            cur_streak = cur_streak + 1 if cur_streak > 0 else 1
            best_win_streak = max(best_win_streak, cur_streak)
        else:
            cur_streak = cur_streak - 1 if cur_streak < 0 else -1
            worst_loss_streak = min(worst_loss_streak, cur_streak)

    sd = _stdev(pnls)
    downside = _stdev([min(0.0, p) for p in pnls])
    avg = total / n
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n, 4),
        "total_pnl": round(total, 2),
        "gross_profit": round(gross_win, 2),
        "gross_loss": round(-gross_loss, 2),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else None,
        "expectancy": round(avg, 2),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "largest_win": round(max(pnls), 2),
        "largest_loss": round(min(pnls), 2),
        "payoff_ratio": round((gross_win / len(wins)) / (gross_loss / len(losses)), 3)
                        if wins and losses and gross_loss else None,
        "pnl_stdev": round(sd, 2),
        "sharpe_per_trade": round(avg / sd, 3) if sd else None,
        "sortino_per_trade": round(avg / downside, 3) if downside else None,
        "max_drawdown_realized": round(max_dd, 2),
        "best_win_streak": best_win_streak,
        "worst_loss_streak": abs(worst_loss_streak),
        "avg_hold_sec": round(sum(holds) / len(holds), 1) if holds else None,
        "avg_hold_human": _human_dur(sum(holds) / len(holds)) if holds else None,
    }


def _human_dur(sec):
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60}m"
    if sec < 86400:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"


def _per_product_trades(closed):
    """Break the closed-trade ledger down per product."""
    by = {}
    for t in closed:
        p = t.get("product", "?")
        b = by.setdefault(p, {"trades": 0, "wins": 0, "pnl": 0.0,
                              "long": 0, "short": 0})
        b["trades"] += 1
        b["pnl"] += t.get("pnl", 0.0)
        if t.get("pnl", 0.0) > 0:
            b["wins"] += 1
        if t.get("side", 1) >= 0:
            b["long"] += 1
        else:
            b["short"] += 1
    for p, b in by.items():
        b["pnl"] = round(b["pnl"], 2)
        b["win_rate"] = round(b["wins"] / b["trades"], 3) if b["trades"] else None
    return dict(sorted(by.items(), key=lambda kv: -kv[1]["pnl"]))


def _per_exit_reason(closed):
    """How trades ended (stop / take / trail / signal-flip / giveback / …)."""
    by = {}
    for t in closed:
        r = t.get("exit_reason", "?")
        b = by.setdefault(r, {"count": 0, "pnl": 0.0})
        b["count"] += 1
        b["pnl"] += t.get("pnl", 0.0)
    for b in by.values():
        b["pnl"] = round(b["pnl"], 2)
    return dict(sorted(by.items(), key=lambda kv: -kv[1]["count"]))


def _strategy_scoring(db, lookback):
    """Per-strategy hit-rate / mean forward-return from scored signals."""
    rows = db.strategy_scores(lookback)
    by = {}
    for r in rows:
        s = r["strategy"]
        b = by.setdefault(s, {"n": 0, "hits": 0, "sum_fwd": 0.0})
        b["n"] += 1
        fwd = r.get("fwd_return") or 0.0
        direction = r.get("direction") or 0.0
        b["sum_fwd"] += fwd
        # a "hit" = signal direction agreed with realized forward return
        if (direction > 0 and fwd > 0) or (direction < 0 and fwd < 0):
            b["hits"] += 1
    for b in by.values():
        b["hit_rate"] = round(b["hits"] / b["n"], 3) if b["n"] else None
        b["mean_fwd_return"] = round(b["sum_fwd"] / b["n"], 5) if b["n"] else None
        del b["sum_fwd"]
    return by


def _per_product_view(PRODUCTS, market, derivatives, nlp, universe_stats):
    """One consolidated object per product fusing price / features / derivatives
    / sentiment / universe membership — the single most useful cross-section."""
    out = {}
    core = set(universe_stats.get("core", [])) if isinstance(universe_stats, dict) else set()
    disc = set(universe_stats.get("discovered", [])) if isinstance(universe_stats, dict) else set()
    for p in PRODUCTS:
        entry = {"price": _safe(lambda p=p: market.price(p))}
        entry["features"] = _safe(lambda p=p: market.features(p))
        entry["derivatives"] = _safe(lambda p=p: derivatives.features(p))
        entry["book"] = _safe(lambda p=p: market.books.get(p))
        entry["sentiment"] = _safe(lambda p=p: nlp.asset_sentiment.get(p))
        entry["in_core"] = p in core
        entry["in_discovered"] = p in disc
        out[p] = entry
    return out


# ------------------------------------------------------------------ main build
def build_export(history_limit=1000, include_features=True, include_analytics=True):
    """Assemble the full-state dict.

    history_limit    — max rows per history section (default 1000; the export
                       aims to be EXHAUSTIVE).
    include_features — per-product computed features + consolidated per-product
                       view (the bulk of the size).
    include_analytics— derived performance/breakdown analytics.
    """
    out = {
        "schema": "cryptomind.export.v2",
        "generated_at": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "history_limit": history_limit,
    }

    # ---- live singletons (import lazily so the CLI path can prime them first)
    from . import db, config
    from .config import PRODUCTS
    from .data.market import market
    from .data.research import research
    from .data.derivatives import derivatives
    from .data.universe import universe
    from .data.memes import memes
    from .data.calendar import calendar
    from .nlp.sentiment import nlp
    from .signals.engine import engine, STRATEGIES
    from .execution.paper import broker
    from .risk.manager import risk
    from .risk.stance import stance
    from .learn.loop import learner
    from .orchestrator import orch
    from .guardian import guardian
    from .strategies.hedge import hedger
    from . import settings as app_settings
    from . import tunables
    from . import alerts
    from . import persistence

    closed = list(broker.closed_trades)

    # ---- meta / environment (constants + what this bot IS)
    out["meta"] = _safe(lambda: {
        "app": "CryptoMind", "version": "2.0", "mode": "paper",
        "products": list(PRODUCTS),
        "strategies": list(STRATEGIES),
        "config": {
            "start_cash": config.START_CASH,
            "fee_rate": config.FEE_RATE, "slippage_bps": config.SLIPPAGE_BPS,
            "risk_per_trade": config.RISK_PER_TRADE,
            "max_position_pct": config.MAX_POSITION_PCT,
            "max_open_positions": config.MAX_OPEN_POSITIONS,
            "max_gross_exposure": config.MAX_GROSS_EXPOSURE,
            "stop_atr_mult": config.STOP_ATR_MULT,
            "take_profit_atr_mult": config.TAKE_PROFIT_ATR_MULT,
            "trail_atr_mult": config.TRAIL_ATR_MULT,
            "max_drawdown_kill": config.MAX_DRAWDOWN_KILL,
            "daily_loss_limit": config.DAILY_LOSS_LIMIT,
            "min_confidence": config.MIN_CONFIDENCE,
            "cooldown_sec": config.COOLDOWN_SEC,
            "allow_shorts": config.ALLOW_SHORTS,
            "candle_granularity": config.CANDLE_GRANULARITY,
            "tick_sec": config.TICK_SEC,
            "market_poll_sec": config.MARKET_POLL_SEC,
            "news_poll_sec": config.NEWS_POLL_SEC,
        },
        "state_path": persistence.STATE_PATH,
        "db_path": config.DB_PATH,
    })

    # ---- account / system status (the master snapshot)
    out["status"] = _safe(orch.snapshot)

    # ---- trades: open positions (full dicts), FULL closed ledger, stats
    out["trades"] = _safe(lambda: {
        "open": [dict(p) for p in broker.positions.values()],
        "open_count": len(broker.positions),
        "closed": closed[-history_limit:],
        "closed_count": len(closed),
        "stats": broker.stats(),
    })

    # ---- derived performance analytics (computed here, not from any endpoint)
    if include_analytics:
        out["analytics"] = _safe(lambda: {
            "performance": _trade_analytics(closed),
            "per_product": _per_product_trades(closed),
            "per_exit_reason": _per_exit_reason(closed),
            "per_strategy_scoring": _strategy_scoring(db, history_limit),
        })

    # ---- equity curve (multiple windows, so the file is self-contained)
    def _equity():
        wins = {"1h": 3600, "6h": 6 * 3600, "1d": 86400, "1w": 7 * 86400,
                "1m": 30 * 86400}
        now = time.time()
        return {tf: db.equity_since(now - sec) for tf, sec in wins.items()}
    out["equity"] = _safe(_equity)

    # ---- signals / strategy votes / weights
    out["signals"] = _safe(lambda: {
        "signals": engine.latest,
        "per_strategy": engine.per_strategy,
        "weights": learner.weights,
        "strategy_list": list(STRATEGIES),
    })

    # ---- market data (features optional — they're the bulk of the size)
    def _market():
        m = {"tickers": market.tickers, "books": market.books,
             "regime": market.regime(), "healthy": market.healthy,
             "last_update": market.last_update}
        if include_features:
            m["features"] = {p: market.features(p) for p in PRODUCTS}
            m["closes"] = {p: market.closes(p) for p in PRODUCTS}
        return m
    out["market"] = _safe(_market)

    # ---- consolidated per-product cross-section (price+features+derivs+sentiment)
    if include_features:
        out["per_product"] = _safe(lambda: _per_product_view(
            PRODUCTS, market, derivatives, nlp, universe.stats()))

    # ---- derivatives (funding / OI / crowding)
    out["derivatives"] = _safe(lambda: {
        "metrics": derivatives.metrics,
        "features": {p: derivatives.features(p) for p in PRODUCTS},
        "no_swap": sorted(derivatives.no_swap),
        "healthy": derivatives.healthy,
        "last_update": derivatives.last_update,
    })

    # ---- dynamic universe (incl. per-coin performance table)
    out["universe"] = _safe(lambda: {**universe.stats(),
                                      "sources": research.source_stats()})

    # ---- meme sleeve
    out["memes"] = _safe(memes.stats)

    # ---- macro-event calendar / blackout
    out["calendar"] = _safe(calendar.stats)

    # ---- research / sentiment (full document set within the history cap)
    out["research"] = _safe(lambda: {
        "documents": research.documents[:history_limit],
        "document_count": len(research.documents),
        "fear_greed": research.fear_greed,
        "research_queue": research.research_queue,
        "narratives": nlp.narratives,
        "asset_sentiment": nlp.asset_sentiment,
        "market_sentiment": nlp.market_sentiment,
        "macro_sentiment": nlp.macro_sentiment,
        "macro_docs": nlp.macro_docs,
        "sentiment_last_update": nlp.last_update,
    })

    # ---- full learning stack (bandit / online model / drift / RL / GA / ...)
    out["learning"] = _safe(learner.full_stats)

    # ---- guardian (decider health / safe mode) + stance + risk detail
    out["guardian"] = _safe(guardian.snapshot)
    out["stance"] = _safe(stance.current)
    out["risk"] = _safe(lambda: {
        "killed": risk.killed, "kill_reason": risk.kill_reason,
        "kill_ts": risk.kill_ts, "halted_today": risk.halted_today,
        "halt_reason": risk.halt_reason,
        "peak_equity": round(risk.peak_equity, 2),
        "kill_arm_peak": round(risk.kill_arm_peak, 2),
        "day_start_equity": risk.day_start_equity,
        "risk_scale": risk.risk_scale,
        "consecutive_losses": risk.consecutive_losses,
        "cooldowns": risk.cooldowns,
        "stop_calibration": risk.stop_calibrator.stats(),
        "protections": risk.protections.stats(),
    })

    # ---- hedger (market-neutral pair sleeve)
    out["hedge"] = _safe(hedger.snapshot)

    # ---- shadow execution (live-vs-paper divergence) + OMS
    if orch.shadow is not None:
        out["shadow"] = _safe(orch.shadow.snapshot)

    # ---- decision + order + trade + signal audit trails (from SQLite)
    out["decisions"] = _safe(lambda: db.recent_decisions(history_limit))
    out["order_events"] = _safe(lambda: db.order_events(limit=history_limit))
    out["events"] = _safe(lambda: db.recent("events", history_limit))
    out["db_trades"] = _safe(lambda: db.recent("trades", history_limit))
    out["signal_scores"] = _safe(lambda: db.recent("signal_scores", history_limit))
    out["db_counts"] = _safe(db.learning_counts)

    # ---- config surfaces
    out["settings"] = _safe(app_settings.load)
    out["tunables"] = _safe(lambda: {"meta": tunables.TUNABLES,
                                     "values": tunables.values()})
    out["alerts"] = _safe(alerts.status)

    # ---- security posture (NEVER the token itself — just whether one is set)
    out["security"] = _safe(lambda: {
        "auth_token_configured": bool(__import__(
            "app.security", fromlist=["token"]).token()),
        "note": "token value intentionally omitted from exports",
    })

    return out


def write_report(directory=None, history_limit=1000, include_features=True,
                 include_analytics=True, compact=False):
    """Build a full-state export and write it to a timestamped file in
    `directory` (default: reports/). Returns the path written."""
    directory = directory or REPORTS_DIR
    os.makedirs(directory, exist_ok=True)
    data = build_export(history_limit=history_limit,
                        include_features=include_features,
                        include_analytics=include_analytics)
    fname = time.strftime("cryptomind_report_%Y%m%d_%H%M%S.json", time.gmtime())
    path = os.path.join(directory, fname)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        if compact:
            json.dump(data, f, separators=(",", ":"), default=str)
        else:
            json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)   # atomic — a reader never sees a half-written file
    return path


def prune_reports(directory=None, keep=288):
    """Keep only the most recent `keep` report files (default 288 = 24h at the
    5-minute default cadence) so the folder doesn't grow without bound."""
    directory = directory or REPORTS_DIR
    try:
        files = sorted(
            (os.path.join(directory, f) for f in os.listdir(directory)
             if f.startswith("cryptomind_report_") and f.endswith(".json")),
            key=os.path.getmtime)
    except FileNotFoundError:
        return 0
    removed = 0
    for old in files[:-keep] if keep > 0 else []:
        try:
            os.remove(old)
            removed += 1
        except OSError:
            pass
    return removed


async def auto_export_loop():
    """Background task: periodically write a full-state report to reports/.

    The cadence is read fresh from settings EVERY cycle, so moving the
    dashboard slider changes how often reports are written without a restart.
    When the feature is toggled off the loop idles (polling once a minute) and
    resumes cleanly when re-enabled. Errors never kill the loop.
    """
    from . import db, settings as app_settings
    db.log_event("system", "Auto-export loop started (periodic full-state reports)")
    # small initial delay so the first report reflects a warmed-up system
    await asyncio.sleep(15)
    while True:
        try:
            if not app_settings.get("auto_export_enabled"):
                await asyncio.sleep(60)     # idle poll while disabled
                continue
            path = await asyncio.to_thread(write_report)
            await asyncio.to_thread(prune_reports)
            db.log_event("export", f"Auto-export wrote {os.path.basename(path)}")
            interval = int(app_settings.get("auto_export_interval_sec"))
        except Exception as e:   # pragma: no cover - defensive; never die
            interval = 300
            try:
                db.log_event("export", f"Auto-export error: {e}")
            except Exception:
                pass
        # Sleep in short slices, re-reading the interval each slice so a slider
        # change (shorter OR longer) is honoured promptly instead of only on the
        # next cycle. Also bail early if the feature is toggled off mid-wait.
        waited = 0
        while True:
            await asyncio.sleep(10)
            waited += 10
            try:
                if not app_settings.get("auto_export_enabled"):
                    break
                interval = max(30, int(app_settings.get("auto_export_interval_sec")))
            except Exception:
                interval = max(30, interval)
            if waited >= interval:
                break


def _prime_from_snapshot():
    """CLI path: load the saved state into the singletons so the export reflects
    the last persisted run without needing the live server."""
    from . import db, persistence
    db.init()
    try:
        persistence.load()
    except Exception as e:  # pragma: no cover - defensive
        print(f"[export] warning: could not load saved state: {e}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Export CryptoMind's full state to one JSON file.")
    parser.add_argument("path", nargs="?", default="cryptomind_export.json",
                        help="output file (default: cryptomind_export.json)")
    parser.add_argument("--history", type=int, default=1000,
                        help="max rows per history section (default: 1000)")
    parser.add_argument("--no-features", action="store_true",
                        help="omit per-product features/closes (smaller file)")
    parser.add_argument("--no-analytics", action="store_true",
                        help="omit computed performance analytics")
    parser.add_argument("--compact", action="store_true",
                        help="minified JSON instead of indented")
    args = parser.parse_args(argv)

    _prime_from_snapshot()
    data = build_export(history_limit=args.history,
                        include_features=not args.no_features,
                        include_analytics=not args.no_analytics)
    with open(args.path, "w") as f:
        if args.compact:
            json.dump(data, f, separators=(",", ":"), default=str)
        else:
            json.dump(data, f, indent=2, default=str)
    n_sections = sum(1 for v in data.values() if isinstance(v, (dict, list)))
    print(f"[export] wrote {args.path} ({n_sections} sections, "
          f"history<= {args.history} rows/section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
