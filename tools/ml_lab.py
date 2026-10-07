"""Pooled cross-sectional ML lab (app/ml/): walk-forward models over every
liquid coin's history, judged against simple baselines.

    python tools/ml_lab.py                       # rank model + trend meta-model
    python tools/ml_lab.py --top 20 --h 7 --model ensemble
    python tools/ml_lab.py --with-series         # + external series features (report only)

Rank model: out-of-sample daily rank IC vs the 30-day-momentum baseline,
Hansen-Hodrick t, both halves; calibration by prediction decile.
Trend meta-model: Brier of P(a coin in its trend gains over h days) vs the
base rate, both halves.

A model that passes its gate is QUEUED as a research-loop candidate
(app/engine/challengers.register) — it then has to beat the champion in both
backtest halves, clear the deflated Sharpe over every trial and win the
paired forward test like anything else. Report: reports/ml_lab_latest.json.
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
    from app.engine import panel as P, universe as U, challengers as C, registry as Rg
    from app.engine import features as F
    from app.ml import dataset as D, models as Mo, evaluate as Ev, strategies as MLS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--h", type=int, default=7)
    ap.add_argument("--model", default="ensemble", choices=["ridge", "gbt", "ensemble"])
    ap.add_argument("--with-series", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    panel = P.from_store()
    mask = U.liquid_mask(panel, a.top)
    extra = {}
    if a.with_series:
        from app.data import series as SR
        for name in SR.names():
            if not name.startswith("ext:"):
                extra[name] = D.series_z(SR.daily_array(name, panel.days))
    X, y, w, names, _ = D.build(panel, mask, h=a.h, extra_market=extra)
    say(f"dataset: {int((w > 0).sum())} samples, {len(names)} features")
    pred = Mo.walk_forward(X, y, w, model=a.model, h=a.h)
    base = F.cross_rank(F.ret_n(panel.close, 30))
    rank = Ev.ranking(pred, y, mask, base, h=a.h)
    rank["calibration_by_decile"] = Ev.calibration(pred, y, mask)
    say(f"rank model: {rank}")
    rep = {"ran_at": time.time(), "data_version": panel.data_version, "features": names,
           "universe_top": a.top, "horizon": a.h, "model": a.model,
           "with_series": sorted(extra), "rank_model": rank}
    meta_cfg = {"assets": ["BTC-USD", "ETH-USD"], "selection": "trend_meta", "sizing": "equal",
                "sma": 125, "hysteresis": 0.02, "model": "ridge", "horizon": a.h,
                "train_top": 30}
    prob = MLS.meta_probabilities(meta_cfg, panel)
    gap = F.sma_gap(panel.close, meta_cfg["sma"])
    held = ~np.isnan(panel.close) & (np.nan_to_num(gap, nan=-1) > 0)
    lab = D.meta_labels(panel, held & U.liquid_mask(panel, 30), h=a.h)
    rep["trend_meta_model"] = Ev.meta(prob, lab)
    say(f"trend meta-model: {rep['trend_meta_model']}")
    Rg.log("ml_lab", f"rank_{a.model}_top{a.top}_h{a.h}", {"series": sorted(extra)},
           panel.data_version, [rank.get("ic") or 0.0], {"passes": rank["passes"]})
    rep["registered"] = []
    if not a.no_register and not extra:
        if rank["passes"]:
            name = f"ml_rank_top{a.top}_h{a.h}"
            cfg = {"universe_top": a.top, "selection": "ml_rank", "sizing": "vol_target",
                   "vol_target": 0.5, "sma": 100, "top_k": 5, "tranches": 7,
                   "model": a.model, "horizon": a.h}
            rep["registered"].append([name, *C.register(name, cfg, source="ml_lab")])
        if rep["trend_meta_model"].get("passes"):
            name = f"btc_eth_trend_meta_h{a.h}"
            rep["registered"].append([name, *C.register(name, meta_cfg, source="ml_lab")])
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "ml_lab_latest.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(json.dumps({k: rep[k] for k in ("rank_model", "trend_meta_model", "registered")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
