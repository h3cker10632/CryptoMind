"""Manual strategy replay — the same code the bot runs automatically.

Replays the LIVE signal engine (app/backtest/replay.py) over the hourly history
the bot caches in .cache/history for every coin it has seen, using your current
tunables.json / settings.json (read-only: this never writes either file).

    python tools/replay_backtest.py                    # full report (both halves + per coin)
    python tools/replay_backtest.py --start 0 --end 0.5     # one window only
    python tools/replay_backtest.py --set fee_rate=0.001    # what-if overrides
    python tools/replay_backtest.py --no-shorts
    python tools/replay_backtest.py --taker            # market-order entries
    python tools/replay_backtest.py --chop             # with the chop filter

The bot itself re-runs this daily and whenever a coin joins the universe
(setting `replay_enabled`); see reports/replay_latest.json, the /replay
Telegram command, or GET /api/backtest/replay.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=float, default=None, help="window start (0-1)")
    ap.add_argument("--end", type=float, default=None, help="window end (0-1)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a tunable for this run only (repeatable)")
    ap.add_argument("--no-shorts", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--maker", action="store_true", help="limit-order entries")
    g.add_argument("--taker", action="store_true", help="market-order entries")
    ap.add_argument("--chop", action="store_true", help="apply the chop filter")
    ap.add_argument("--per-coin", action="store_true", help="print the per-coin table")
    a = ap.parse_args()

    from app.backtest import replay as rp
    overrides = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = float(v)
    shorts = False if a.no_shorts else None
    maker = True if a.maker else False if a.taker else None
    candles = rp.load_cached_history(ROOT)

    if a.start is not None or a.end is not None:
        r = rp.run_replay(candles, a.start or 0.0, 1.0 if a.end is None else a.end,
                          overrides=overrides, shorts=shorts, maker=maker, chop=a.chop)
        trades = r.pop("_trades", [])
        print(json.dumps(r, indent=1))
        if a.per_coin:
            print(json.dumps(rp.per_product(trades, candles), indent=1))
        return
    if maker is not None:
        from app import settings
        settings.load()["entry_order_type"] = "maker" if maker else "taker"  # this run only
    rep = rp.run_report(candles, overrides=overrides, shorts=shorts, chop=a.chop)
    pp = rep.pop("per_product")
    print(json.dumps(rep, indent=1))
    if a.per_coin:
        print(json.dumps(pp, indent=1))


if __name__ == "__main__":
    main()
