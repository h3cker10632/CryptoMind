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
from .signal import signal_series, align_forward_returns, _price_at, _price_after
from . import study


def _diagnose(snaps, sig, px, aligned, horizon_sec):
    """Explain, stage by stage, where the (signal, forward-return) pipeline
    loses everything — so 'NO DATA' becomes actionable instead of opaque."""
    n_traders = sum(len(s.traders) for s in snaps)
    n_positions = sum(len(t.positions) for s in snaps for t in s.traders)
    assets_signal = set(sig.keys())
    assets_priced = set(px.keys())
    matched = assets_signal & assets_priced
    # why aligned rows were dropped, among matched assets
    no_p0 = no_forward = ok = 0
    for a in matched:
        p = px.get(a) or []
        for ts, _lean in sig.get(a, []):
            if _price_at(p, ts) is None:
                no_p0 += 1
            elif _price_after(p, ts + horizon_sec - 1e-6) is None:
                no_forward += 1        # horizon window hasn't matured yet
            else:
                ok += 1
    return {
        "snapshots": len(snaps),
        "traders_total": n_traders,
        "positions_total": n_positions,
        "assets_in_signal": sorted(assets_signal),
        "assets_priced": sorted(assets_priced),
        "assets_matched": sorted(matched),
        "signal_points": sum(len(v) for v in sig.values()),
        "dropped_no_price_history": no_p0,
        "dropped_horizon_not_matured": no_forward,
        "aligned_pairs": sum(len(v) for v in aligned.values()),
    }



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
    horizon_sec = horizon_hours * 3600.0
    sig = signal_series(snaps, rank_decay=rank_decay, use_score=use_score)
    aligned = align_forward_returns(sig, px, horizon_sec)
    diag = _diagnose(snaps, sig, px, aligned, horizon_sec)

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
    return {"per_asset": per_asset, "pooled": pooled, "diagnostics": diag,
            "params": {"horizon_hours": horizon_hours,
                       "rank_decay": rank_decay, "use_score": use_score}}


def _verdict(pooled, diag=None):
    if not pooled:
        if diag:
            # pinpoint the stage that collapsed to zero
            if diag["positions_total"] == 0:
                return ("NO DATA — your snapshots contain traders but ZERO "
                        "positions. The leaderboard endpoint (e.g. get_users) "
                        "doesn't include open positions, so there is no lean to "
                        "score. Map invo_map_positions to a real positions list, "
                        "or set a per-trader positions endpoint (invo_positions_tmpl "
                        "+ invo_map_side/size/asset).")
            if not diag["assets_in_signal"]:
                return ("NO DATA — positions exist but none produced a usable lean "
                        "(check invo_map_asset / invo_map_side / invo_map_size — "
                        "asset was empty or notional <= 0 for every row).")
            if not diag["assets_matched"]:
                return ("NO DATA — signal assets "
                        f"{diag['assets_in_signal'][:8]} don't match your priced "
                        f"universe {diag['assets_priced'][:8]}. Symbols must match "
                        "the bare base symbol of a traded product (e.g. BTC, SOL).")
            if diag["dropped_horizon_not_matured"] and not diag["aligned_pairs"]:
                return ("NO DATA — every snapshot is too recent: the "
                        f"{diag['dropped_horizon_not_matured']} signal points have "
                        "no price yet at the chosen horizon into the future. Wait "
                        "for the forward-return window to mature, or lower "
                        "invo_horizon_hours.")
            if diag["dropped_no_price_history"] and not diag["aligned_pairs"]:
                return ("NO DATA — matched assets have no candle history at the "
                        "snapshot timestamps. Let the market feed accumulate "
                        "candles that overlap the collection window.")
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
    d = report.get("diagnostics")
    if d:
        print(f"pipeline: {d['snapshots']} snaps -> {d['traders_total']} traders "
              f"-> {d['positions_total']} positions -> {d['signal_points']} signal "
              f"points -> {len(d['assets_matched'])} priced assets "
              f"-> {d['aligned_pairs']} aligned pairs")
        if not p:
            print(f"          dropped: {d['dropped_no_price_history']} no-price, "
                  f"{d['dropped_horizon_not_matured']} horizon-not-matured")
        print("-" * 70)
    print("VERDICT:", _verdict(p, d))
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
