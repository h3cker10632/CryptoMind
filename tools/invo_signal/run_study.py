"""CLI: run the Invo-signal edge study and print a verdict-style report.

Usage:
  python -m tools.invo_signal.run_study \
      --snapshots snaps.json --prices prices.json \
      [--baseline baseline.json] [--horizon-hours 4] \
      [--rank-decay 1.0] [--use-score]

All inputs are files you provide (see collector.py for schemas). Nothing is
fetched from Invo. The report tells you, per asset and pooled, whether the
signal predicts forward returns and whether it adds anything beyond the
existing funding/OI/long-short baseline.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from .collector import load_snapshots, load_prices, load_baseline
from .signal import signal_series, align_forward_returns
from . import study


def _baseline_for(asset, rows, baseline):
    """Nearest-at-or-before baseline feature row for each aligned ts."""
    if not baseline or asset not in baseline:
        return None
    b = baseline[asset]
    out = []
    for _, _, ts in rows:
        chosen = None
        for entry in b:
            if entry[0] <= ts:
                chosen = entry[1:]
            else:
                break
        out.append(chosen if chosen else [0.0] * (len(b[0]) - 1))
    return out


def run(snapshots, prices, baseline=None, horizon_hours=4.0,
        rank_decay=1.0, use_score=False):
    snaps = load_snapshots(snapshots)
    px = load_prices(prices)
    base = load_baseline(baseline) if baseline else None
    sig = signal_series(snaps, rank_decay=rank_decay, use_score=use_score)
    aligned = align_forward_returns(sig, px, horizon_hours * 3600.0)

    per_asset, pooled_sig, pooled_ret = {}, [], []
    for asset, rows in sorted(aligned.items()):
        br = _baseline_for(asset, rows, base)
        per_asset[asset] = study.evaluate_asset(rows, br)
        pooled_sig += [r[0] for r in rows]
        pooled_ret += [r[1] for r in rows]

    pooled = {}
    if pooled_sig:
        s = np.array(pooled_sig); r = np.array(pooled_ret)
        pooled = {"n": len(s), "ic": study.spearman(s, r),
                  "pearson": study.pearson(s, r),
                  "mutual_info": study.mutual_info(s, r),
                  "p_value": study.permutation_pvalue(s, r)}
    return {"per_asset": per_asset, "pooled": pooled,
            "params": {"horizon_hours": horizon_hours,
                       "rank_decay": rank_decay, "use_score": use_score}}


def _verdict(pooled):
    if not pooled:
        return "NO DATA — no (signal, forward-return) pairs could be aligned."
    ic, p, n = pooled["ic"], pooled["p_value"], pooled["n"]
    if n < 30:
        return f"INCONCLUSIVE — only {n} samples; collect more before trusting IC."
    if p > 0.05:
        return (f"NO EDGE — IC={ic:+.3f} not distinguishable from noise "
                f"(p={p:.3f}). Do NOT add the feature.")
    if abs(ic) < 0.03:
        return (f"NEGLIGIBLE — IC={ic:+.3f} is significant but tiny; unlikely "
                f"to survive costs/latency. Lean against adding it.")
    return (f"PROMISING — IC={ic:+.3f} (p={p:.3f}). Check per-asset partial_ic "
            f"& ablation d_acc: if it stays positive after the baseline, ship it "
            f"as a measured feature.")


def _print(report):
    print("=" * 70)
    print("INVO SIGNAL EDGE STUDY")
    print("params:", report["params"])
    print("=" * 70)
    for a, m in report["per_asset"].items():
        line = (f"{a:6s} n={m['n']:4d}  IC={m['ic']:+.3f}  p={m['p_value']:.3f}"
                f"  MI={m['mutual_info']:.3f}")
        if m.get("partial_ic") is not None:
            line += f"  partIC={m['partial_ic']:+.3f}"
        ab = m.get("ablation", {})
        if ab and not ab.get("insufficient") and "d_acc" in ab:
            line += f"  ablation d_acc={ab['d_acc']:+.3f} d_logloss={ab['d_logloss']:+.3f}"
        print(line)
    print("-" * 70)
    p = report["pooled"]
    if p:
        print(f"POOLED n={p['n']}  IC={p['ic']:+.3f}  pearson={p['pearson']:+.3f}"
              f"  MI={p['mutual_info']:.3f}  p={p['p_value']:.3f}")
    print("-" * 70)
    print("VERDICT:", _verdict(p))
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--horizon-hours", type=float, default=4.0)
    ap.add_argument("--rank-decay", type=float, default=1.0)
    ap.add_argument("--use-score", action="store_true")
    ap.add_argument("--json", action="store_true", help="dump raw JSON too")
    a = ap.parse_args()
    rep = run(a.snapshots, a.prices, a.baseline, a.horizon_hours,
              a.rank_decay, a.use_score)
    _print(rep)
    if a.json:
        print(json.dumps(rep, indent=2, default=float))
