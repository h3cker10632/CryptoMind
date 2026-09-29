"""Heuristic edge signals for Polymarket binary markets.

IMPORTANT — these are transparent HEURISTIC PRIORS, not claimed alpha. Each one
emits a small directional "lean" on which outcome looks mispriced, drawn from a
REAL, observable market feature. The engine combines them using bandit weights,
and the learner then MEASURES — from actual resolutions — whether each heuristic
predicts anything. A heuristic with no edge gets its weight decayed toward zero;
nothing here is trusted on faith. This mirrors exactly how the crypto side turns
strategy votes into bandit-weighted, PnL-attributed learning.

Convention: a strategy returns a `lean` in [-1, +1] where POSITIVE means
"outcome 0 (the first listed outcome) looks UNDERPRICED — lean toward buying it"
and negative means outcome 1 looks underpriced. The engine turns the net lean
into a fair-probability nudge (bounded by `pm_edge_scale`) and hands the edge to
the sizer.

The four priors:
  * momentum       — recent price drift tends to continue over short horizons.
  * mean_revert    — a very sharp 1h move often overshoots and partly reverts.
  * longshot_fade  — the documented favourite-longshot bias: extreme longshots
                     (very cheap outcomes) are, on average, overpriced; fade them
                     by leaning toward the favourite.
  * microstructure — last trade sitting away from the mid hints at order-flow
                     pressure toward that side.
"""
from __future__ import annotations

import math

STRATEGIES = ("momentum", "mean_revert", "longshot_fade", "microstructure")


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _leans(m: dict) -> dict:
    """Per-strategy leans in [-1, 1] (positive = outcome 0 underpriced)."""
    p0 = m["prices"][0]
    mom1h = m.get("mom_1h", 0.0)
    mom1d = m.get("mom_1d", 0.0)
    # `mom_*` on Gamma is the change in outcome-0's price. A rise in p0 is
    # bullish momentum FOR outcome 0.
    momentum = _clip((0.5 * mom1h + 0.5 * mom1d) * 6.0)
    # sharp single-hour spikes tend to partly retrace -> fade the 1h move
    mean_revert = _clip(-mom1h * 8.0) if abs(mom1h) > 0.05 else 0.0
    # favourite-longshot bias: if outcome 0 is a deep longshot (cheap), lean
    # AWAY from it (negative); if it's the heavy favourite, lean toward it.
    dist = p0 - 0.5
    longshot_fade = _clip(math.copysign(min(abs(dist) * 2.0, 1.0), dist)) \
        if abs(dist) > 0.15 else 0.0
    # order-flow: last trade above the implied mid -> pressure toward outcome 0
    micro = _clip((m.get("last_trade_price", p0) - p0) * 10.0)
    return {"momentum": momentum, "mean_revert": mean_revert,
            "longshot_fade": longshot_fade, "microstructure": micro}


def evaluate(m: dict, weights: dict, edge_scale: float) -> dict:
    """Turn a market + bandit weights into a trade candidate.

    Returns a dict with the chosen outcome, its price, estimated edge (fair minus
    market prob), a [0,1] confidence, and the raw per-strategy votes (for
    attribution). `outcome_index` is which of the two tokens to buy; None means
    no actionable lean.
    """
    leans = _leans(m)
    # weighted net lean toward outcome 0 (weights default to 1.0 per strategy)
    wsum = sum(abs(weights.get(s, 1.0)) for s in STRATEGIES) or 1.0
    net = sum(weights.get(s, 1.0) * leans[s] for s in STRATEGIES) / wsum
    net = _clip(net)
    # agreement: fraction of non-zero strategies that point the same way as net
    active = [leans[s] for s in STRATEGIES if abs(leans[s]) > 1e-6]
    if active and abs(net) > 1e-6:
        agree = sum(1 for l in active if (l > 0) == (net > 0)) / len(active)
    else:
        agree = 0.0
    confidence = _clip(abs(net) * agree, 0.0, 1.0)

    # fair-prob nudge for outcome 0, bounded by edge_scale
    p0 = m["prices"][0]
    fair0 = _clip(p0 + net * edge_scale, 0.001, 0.999)
    if net >= 0:
        idx, price, fair = 0, p0, fair0
    else:
        idx, price, fair = 1, m["prices"][1], _clip(1.0 - fair0, 0.001, 0.999)
    edge = fair - price          # positive when the chosen outcome is underpriced

    # votes are signed toward the CHOSEN outcome so the bandit credits correctly
    sign = 1.0 if idx == 0 else -1.0
    votes = {s: round(leans[s] * sign, 3) for s in STRATEGIES}
    return {
        "outcome_index": idx if abs(net) > 1e-6 else None,
        "outcome": m["outcomes"][idx],
        "price": price,
        "fair": round(fair, 4),
        "edge": round(edge, 4),
        "confidence": round(confidence, 3),
        "net_lean": round(net, 3),
        "agreement": round(agree, 3),
        "votes": votes,
        "leans": {k: round(v, 3) for k, v in leans.items()},
    }
