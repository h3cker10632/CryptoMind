"""Daily lab — test core selection (trend / momentum / rank model) and sizing
(equal / inverse-vol / vol-target) on years of daily history, then train the
live rank model. Method: app/backtest/daily_lab.py. Writes
reports/daily_lab_latest.json (read by the core in "auto" mode) and
reports/daily_rank_model.pkl.

    python tools/daily_lab.py              # fetch/refresh daily history, run, train
    python tools/daily_lab.py --no-fetch   # use the cached history only
"""
import argparse
import asyncio
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
REPORT = os.path.join(ROOT, "reports", "daily_lab_latest.json")
MODEL = os.path.join(ROOT, "reports", "daily_rank_model.pkl")


def load_daily(root=ROOT, products=None):
    """Daily candles from the data store (all history) for `products` —
    default: the coins the bot has traded. Legacy JSON cache as fallback."""
    from app.data import store
    from app.backtest.replay import _legacy_products
    data, _ = store.load_candles(86400, products or _legacy_products(root, 3600) or None)
    if data:
        return data
    out = {}
    for f in glob.glob(os.path.join(root, ".cache", "history", "*_86400_*.json")):
        p = os.path.basename(f).split("_")[0]
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        if len(rows) > len(out.get(p, [])):
            out[p] = rows
    return out


def main():
    from app.backtest import daily_lab as dl
    from app import settings
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    if not a.no_fetch:
        from tools.data_sync import sync
        asyncio.run(sync("daily", say=say))
    candles = load_daily()
    try:
        sma = int(settings.get("core_sma_days"))
    except Exception:
        sma = 100
    rep = dl.run_lab(candles, sma_days=sma, progress=say)
    if not rep.get("ok"):
        print(json.dumps(rep, indent=1))
        sys.exit(1)
    say("training the live rank model on all history")
    model = dl.train_live_model(candles)
    if model is not None:
        dl.save_model(model, MODEL)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT + ".tmp", "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(REPORT + ".tmp", REPORT)

    print(f"\n{rep['coins']} coins, {rep['oos_days']} out-of-sample days "
          f"(Sharpe = return per unit of risk, annualized)")
    head = f"{'variant':24s} {'return %':>9} {'Sharpe':>6} {'max DD %':>8}   " \
           f"{'1st half Sharpe':>15} {'2nd half Sharpe':>15} {'invested':>8}"
    print(head + "\n" + "-" * len(head))
    rows = list(rep["variants"].items()) + [("buy & hold (equal)", rep["buy_hold_equal"])]
    for k, v in rows:
        print(f"{k:24s} {v['full']['return_pct']:>9} {v['full']['sharpe']:>6} "
              f"{v['full']['max_drawdown_pct']:>8}   {v['first_half']['sharpe']:>15} "
              f"{v['second_half']['sharpe']:>15} "
              f"{str(v.get('avg_invested_pct', 100)) + '%':>8}")
    d = rep["decision"]
    print(f"\nDecision for core 'auto': {d['variant']} ({d['why']}); passed: {d['passed'] or 'none'}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
