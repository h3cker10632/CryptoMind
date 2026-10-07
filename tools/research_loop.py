"""Champion / challenger research loop (app/engine/challengers.py).

    python tools/research_loop.py            # backtest + forward-track + promote
    python tools/research_loop.py --status   # print the last report only

The bot runs this weekly in a background process. A challenger replaces the
champion only after >= 90 days of better FORWARD performance (data that did
not exist when its config was frozen) plus a backtest that beats the champion
in both halves with a deflated Sharpe >= 0.95.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def show(rep):
    print(f"champion: {rep['champion']}   trials counted for deflated Sharpe: {rep['trials']}"
          + (f"   PROMOTED: {rep['promoted']}" if rep.get("promoted") else ""))
    print(f"{'candidate':22s} {'Sharpe':>6} {'1st/2nd half':>13} {'CAGR%':>7} {'maxDD%':>7} "
          f"{'DSR':>5} {'turn/yr':>7} {'fwd days':>8} {'beats champ':>11} {'fwd test':>10}")
    for name, c in rep["candidates"].items():
        b = c["backtest"]
        ft = (c.get("forward_test") or {}).get("decision", "-")
        print(f"{name:22s} {b['full']['sharpe']:>6} "
              f"{str(b['first_half']['sharpe']) + '/' + str(b['second_half']['sharpe']):>13} "
              f"{b['cagr_pct']:>7} {b['full']['max_drawdown_pct']:>7} {b['deflated_sharpe']:>5} "
              f"{c['turnover_per_year']:>7} {c['forward_days']:>8} "
              f"{str(c['beats_champion_backtest_both_halves']):>11} {ft:>10}")
    ct = rep.get("costs_taxes") or {}
    if ct.get("scenarios"):
        print("\nchampion under other costs / taxes (CAGR %, Sharpe):")
        for k, v in ct["scenarios"].items():
            print(f"  {k:34s} {v.get('cagr_pct')!s:>7} {v.get('sharpe')!s:>6}")


def main():
    from app.engine import challengers as C
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        st = C._load(C.STATE, {})
        if st.get("last_report"):
            show(st["last_report"])
        return
    t0 = time.time()
    from app import settings
    rep = C.run(min_forward_days=int(settings.get("research_min_forward_days")),
                min_dsr=float(settings.get("research_min_dsr")),
                max_forward_days=int(settings.get("research_max_forward_days")),
                alpha=float(settings.get("research_forward_alpha")),
                tau=float(settings.get("research_forward_tau")),
                progress=lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True))
    show(rep)


if __name__ == "__main__":
    main()
