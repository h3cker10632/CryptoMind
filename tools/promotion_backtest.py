"""Backtest the research loop's promotion process itself
(app/engine/promotion_backtest.py): replay the champion / challenger rule
over history with only the data known at each evaluation date and compare
following it with never switching, holding BTC+ETH and the best candidate in
hindsight. It replays the BUILT-IN candidates only: queued ones exist because
of a gate run on today's history, so they could not have competed earlier.
The built-in set itself was also chosen with hindsight, which flatters the
process somewhat.

    python tools/promotion_backtest.py
    python tools/promotion_backtest.py --replay-from 2021-06-01 --every 30
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from app.engine import panel as P, challengers as C, promotion_backtest as PB
    from app.engine import registry as Rg
    from app import settings
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-from", default=None)
    ap.add_argument("--every", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    rep = PB.run(P.from_store(), C.CANDIDATES, C.DEFAULT_CHAMPION,
                 replay_from=a.replay_from, eval_every=a.every,
                 min_dsr=float(settings.get("research_min_dsr")),
                 min_forward_days=int(settings.get("research_min_forward_days")),
                 max_forward_days=int(settings.get("research_max_forward_days")),
                 alpha=float(settings.get("research_forward_alpha")),
                 tau=float(settings.get("research_forward_tau")),
                 n_trials=max(len(C.CANDIDATES),     # as live: every variant ever logged
                              sum(Rg.n_trials(f, floor=0) for f in C.RESEARCH_FAMILIES)),
                 trial_sr_std=Rg.trial_sharpe_std(C.FAMILY),
                 progress=lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True))
    rep["note"] = ("built-in candidates only; that set was itself chosen with hindsight, "
                   "so the process result is somewhat flattered")
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "promotion_backtest_latest.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("from", "days", "n_trials", "verdict", "promotions",
                                          "note")}, indent=1))
    for k in ("process", "never_switch", "hold_btc_eth", "best_in_hindsight"):
        if k in rep:
            f_ = rep[k]["full"]
            print(f"{k:18s} Sharpe {f_['sharpe']!s:>5}  return {f_['return_pct']!s:>8}%  "
                  f"maxDD {f_['max_drawdown_pct']!s:>6}%")


if __name__ == "__main__":
    main()
