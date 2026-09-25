"""Generate labeled synthetic datasets to VALIDATE THE HARNESS ITSELF.

This is a correctness fixture, NOT Invo data and never presented as such: it
writes snapshot/price files with a *known* embedded relationship so we can
confirm the study recovers real edge and rejects noise. Two modes:

  edge  : trader lean genuinely leads forward return with strength `rho`
  null  : trader lean is pure noise, independent of returns

If the harness reports significant IC on `edge` and no edge on `null`, the
measurement machinery works — then it can be trusted on real collected data.
"""
from __future__ import annotations
import json, argparse
import numpy as np

ASSETS = ["BTC", "ETH", "SOL"]


def generate(mode="edge", n_snaps=300, rho=0.25, seed=0, step_sec=3600.0):
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000.0
    snaps, prices = [], {a: [] for a in ASSETS}
    # per-asset running price
    px = {a: 100.0 * (1 + i) for i, a in enumerate(ASSETS)}
    for a in ASSETS:                       # seed first price point
        prices[a].append([t0 - step_sec, px[a]])

    for k in range(n_snaps):
        ts = t0 + k * step_sec
        traders = []
        for a_i, a in enumerate(ASSETS):
            lean = float(np.clip(rng.normal(0, 0.5), -1, 1))   # crowd lean
            noise = rng.normal(0, 0.02)
            if mode == "edge":
                fwd = rho * lean * 0.02 + noise    # lean genuinely predicts
            else:
                fwd = noise                        # lean is irrelevant
            # build traders whose aggregate net lean ~= `lean`
            for r in range(1, 6):
                d = 1 if (lean + rng.normal(0, 0.3)) > 0 else -1
                traders_pos = [{"asset": a, "direction": d,
                                "notional": 1000.0 * (1 if d * lean > 0 else 0.4),
                                "leverage": 3}]
                traders.append({"id": f"{a}_t{r}", "rank": r, "score": 0.6,
                                "positions": traders_pos})
            px[a] *= (1 + fwd)
            prices[a].append([ts + step_sec, px[a]])
        snaps.append({"ts": ts, "traders": traders})
    return snaps, prices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["edge", "null"], default="edge")
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="/tmp/invo_syn")
    a = ap.parse_args()
    snaps, prices = generate(a.mode, a.n, a.rho, a.seed)
    with open(f"{a.out_prefix}_snapshots.json", "w") as f:
        json.dump(snaps, f)
    with open(f"{a.out_prefix}_prices.json", "w") as f:
        json.dump(prices, f)
    print(f"wrote {a.out_prefix}_snapshots.json ({len(snaps)} snaps) "
          f"and {a.out_prefix}_prices.json  mode={a.mode} rho={a.rho}")


if __name__ == "__main__":
    main()
