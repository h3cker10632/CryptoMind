"""Full-state export — aggregate every data surface into ONE JSON document.

Two ways to get it:
  * live, from the running server:  GET /api/export           (in-memory state)
  * offline, from the CLI:          python -m app.export out.json
    (rebuilds state from the saved snapshot + SQLite, no server needed)

The dashboard's ~24 read endpoints each expose one slice; this stitches them all
into a single object so the operator can grab the WHOLE picture in one file. Every
section is isolated in a try/except: a failing subsystem records an "error" string
for that key instead of breaking the whole dump. History depth is capped per
section (and overridable) so the file stays a reasonable size.
"""
import json
import time


def _safe(fn, default=None):
    """Run a section producer; never let one failure sink the whole export."""
    try:
        return fn()
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"{type(e).__name__}: {e}"}


def build_export(history_limit=200, include_features=True):
    """Assemble the full-state dict. `history_limit` caps per-table row counts."""
    out = {
        "schema": "cryptomind.export.v1",
        "generated_at": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ---- live singletons (import lazily so the CLI path can prime them first)
    from . import db
    from .config import PRODUCTS
    from .data.market import market
    from .data.research import research
    from .data.derivatives import derivatives
    from .data.universe import universe
    from .data.calendar import calendar
    from .nlp.sentiment import nlp
    from .signals.engine import engine
    from .execution.paper import broker
    from .risk.manager import risk
    from .learn.loop import learner
    from .orchestrator import orch
    from .guardian import guardian
    from . import settings as app_settings
    from . import tunables
    from . import alerts

    # ---- account / system status (the master snapshot)
    out["status"] = _safe(orch.snapshot)

    # ---- trades + performance
    out["trades"] = _safe(lambda: {
        "open": [dict(p) for p in broker.positions.values()],
        "closed": broker.closed_trades[-history_limit:],
        "stats": broker.stats(),
    })

    # ---- equity curve (multiple windows, so the file is self-contained)
    def _equity():
        wins = {"1h": 3600, "1d": 86400, "1w": 7 * 86400, "1m": 30 * 86400}
        now = time.time()
        return {tf: db.equity_since(now - sec) for tf, sec in wins.items()}
    out["equity"] = _safe(_equity)

    # ---- signals / strategy votes / weights
    out["signals"] = _safe(lambda: {
        "signals": engine.latest,
        "per_strategy": engine.per_strategy,
        "weights": learner.weights,
    })

    # ---- market data (features optional — they're the bulk of the size)
    def _market():
        m = {"tickers": market.tickers, "books": market.books,
             "regime": market.regime(), "last_update": market.last_update}
        if include_features:
            m["features"] = {p: market.features(p) for p in PRODUCTS}
        return m
    out["market"] = _safe(_market)

    # ---- derivatives (funding / OI / crowding)
    out["derivatives"] = _safe(lambda: {
        "metrics": derivatives.metrics,
        "features": {p: derivatives.features(p) for p in PRODUCTS},
        "healthy": derivatives.healthy,
        "last_update": derivatives.last_update,
    })

    # ---- dynamic universe (incl. per-coin performance table)
    out["universe"] = _safe(lambda: {**universe.stats(),
                                      "sources": research.source_stats()})

    # ---- macro-event calendar / blackout
    out["calendar"] = _safe(calendar.stats)

    # ---- research / sentiment
    out["research"] = _safe(lambda: {
        "documents": research.documents[:50],
        "fear_greed": research.fear_greed,
        "research_queue": research.research_queue,
        "narratives": nlp.narratives,
        "asset_sentiment": nlp.asset_sentiment,
        "market_sentiment": nlp.market_sentiment,
        "macro_sentiment": nlp.macro_sentiment,
    })

    # ---- full learning stack (bandit / online model / drift / RL / GA / ...)
    out["learning"] = _safe(learner.full_stats)

    # ---- guardian (decider health / safe mode)
    out["guardian"] = _safe(guardian.snapshot)

    # ---- decision + order audit trails (from SQLite)
    out["decisions"] = _safe(lambda: db.recent_decisions(history_limit))
    out["order_events"] = _safe(lambda: db.order_events(limit=history_limit))
    out["events"] = _safe(lambda: db.recent("events", history_limit))
    out["db_counts"] = _safe(db.learning_counts)

    # ---- config surfaces
    out["settings"] = _safe(app_settings.load)
    out["tunables"] = _safe(lambda: {"meta": tunables.TUNABLES,
                                     "values": tunables.values()})
    out["alerts"] = _safe(alerts.status)

    return out


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
    parser.add_argument("--history", type=int, default=200,
                        help="max rows per history section (default: 200)")
    parser.add_argument("--no-features", action="store_true",
                        help="omit per-product computed features (smaller file)")
    args = parser.parse_args(argv)

    _prime_from_snapshot()
    data = build_export(history_limit=args.history,
                        include_features=not args.no_features)
    with open(args.path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    n_sections = sum(1 for v in data.values() if isinstance(v, (dict, list)))
    print(f"[export] wrote {args.path} ({n_sections} sections, "
          f"history<= {args.history} rows/section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
