"""Screen candidate signals before building strategies on them
(app/engine/screen.py).

    python tools/signal_screen.py                  # built-in features, h = 7 and 30 days
    python tools/signal_screen.py --top 30 --h 7   # universe size / horizon

Cross-sectional features are scored on the point-in-time liquid universe
(top N coins by trailing dollar volume, delisted included); market-wide
series from the external series store (app/data/series.py) are scored
against BTC. Holm-adjusted across everything screened in the run. The report
goes to reports/signal_screen_latest.json; every (feature, horizon) is logged
to the experiment registry (family "screen").
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    import numpy as np
    from app.engine import panel as P, screen as S, universe as U, registry as Rg
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--h", type=int, nargs="*", default=[7, 30])
    ap.add_argument("--from-day", default="2019-01-01")
    a = ap.parse_args()
    t0 = time.time()
    panel = P.from_store()
    import datetime as dt
    start = int((panel.days < (dt.date.fromisoformat(a.from_day) - dt.date(1970, 1, 1)).days).sum())
    mask = U.liquid_mask(panel, a.top)
    res = {}
    for name, X in S.builtin_features(panel).items():
        for h in a.h:
            res[f"{name} (h={h})"] = S.cross_sectional(X, panel.close, h, mask, start=start)
    try:
        from app.data import series as SR
        if "BTC-USD" in panel.coins:
            btc = panel.close[:, panel.coins.index("BTC-USD")]
            for sname in SR.names():
                x = SR.daily_array(sname, panel.days)
                if np.isfinite(x).sum() < 200:
                    continue
                for h in a.h:
                    res[f"{sname} vs BTC (h={h})"] = S.time_series(x, btc, h, start=start)
    except ImportError:
        pass
    S.holm(res)
    for k, v in res.items():
        Rg.log("screen", k, {"top": a.top}, panel.data_version, [v.get("mean") or v.get("corr") or 0.0],
               {"t": v["t"], "passes": v["passes"]})
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "signal_screen_latest.json"), "w") as f:
        json.dump({"ran_at": time.time(), "data_version": panel.data_version,
                   "universe_top": a.top, "results": res}, f, indent=1)
    print(f"{'signal':40s} {'IC/corr':>8} {'t':>6} {'halves':>16} {'p_holm':>7} pass")
    for k, v in sorted(res.items(), key=lambda kv: -abs(kv[1]["t"] or 0)):
        m = v.get("mean", v.get("corr"))
        print(f"{k:40s} {m!s:>8} {v['t']!s:>6} {str(v['halves_mean']):>16} "
              f"{v['p_holm']!s:>7} {'YES' if v['passes'] else ''}")
    print(f"\n{len(res)} screened in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
