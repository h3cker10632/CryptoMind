"""Backtest a portfolio strategy on the data store and log it to the registry.

    python tools/backtest.py --selection momentum --sizing vol_target
    python tools/backtest.py --universe all --start 2021-01-01 --top-k 8
    python tools/backtest.py --list            # registry: every trial so far

Every run is logged (app/engine/registry.py) with its data version, so the
deflated Sharpe it prints counts EVERY variant ever tried in the family — the
honest correction for picking the best of many backtests.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from app.engine import panel as P, strategies as S, backtest as B, registry as R
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", default="trend", choices=("trend", "momentum"))
    ap.add_argument("--sizing", default="equal", choices=("equal", "inverse_vol", "vol_target"))
    ap.add_argument("--sma", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--vol-target", type=float, default=0.5)
    ap.add_argument("--tranches", type=int, default=7, help="weekly-pick slices (1 = none)")
    ap.add_argument("--cost", type=float, default=0.006, help="cost per side")
    ap.add_argument("--universe", default="bot", help="bot (coins the bot trades) | all")
    ap.add_argument("--start", default="2020-01-23", help="first day scored (YYYY-MM-DD)")
    ap.add_argument("--family", default="core")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for e in sorted(R.entries(), key=lambda e: e["ts"]):
            m = e["metrics"].get("full", {})
            print(f"{e['family']:8s} {e['name']:28s} Sharpe {m.get('sharpe')} "
                  f"data {e['data_version']} {json.dumps(e['params'])}")
        return
    t0 = time.time()
    products = None
    if a.universe == "bot":
        from app.backtest.replay import _legacy_products
        products = _legacy_products(ROOT, 3600) or None
    p = P.from_store(products)
    start_day = int(dt.datetime.strptime(a.start, "%Y-%m-%d").replace(
        tzinfo=dt.UTC).timestamp()) // 86400
    s0 = int((p.days < start_day).sum())
    params = {"selection": a.selection, "sizing": a.sizing, "sma": a.sma, "top_k": a.top_k,
              "vol_target": a.vol_target, "tranches": a.tranches, "cost": a.cost,
              "universe": a.universe,
              "start": a.start}
    W = S.trend_portfolio(p, a.selection, a.sizing, sma=a.sma, top_k=a.top_k,
                          vol_target=a.vol_target, start=s0, tranches=a.tranches)
    rets, invested = B.simulate(W, p.returns(), a.cost, start=s0)
    name = f"{a.selection}+{a.sizing}"
    R.log(a.family, name, params, p.data_version, rets, {}, start_day=start_day)
    rep = B.report(rets, invested, n_trials=R.n_trials(a.family),
                   trial_sr_std=R.trial_sharpe_std(a.family))
    R.log(a.family, name, params, p.data_version, rets, rep, start_day=start_day)
    print(json.dumps({"variant": name, "coins": p.N, "data_version": p.data_version,
                      "trials_in_family": R.n_trials(a.family), **rep,
                      "elapsed_sec": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
