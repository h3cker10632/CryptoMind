"""Turn raw Invo snapshots into a per-asset, per-timestamp positioning signal.

The signal for asset A at snapshot time t is the *rank-weighted net directional
lean* of the tracked traders:

    lean(A,t) = sum_i  w(rank_i) * dir * notional      /   sum_i w(rank_i) * notional
                over every position i in A across all traders

  - w(rank) = 1 / rank ** rank_decay  (top traders count more; decay tunable)
  - result is in [-1, +1]: +1 = the smart crowd is unanimously long A, -1 short.

Optionally weight by trader `score` too (e.g. win-rate) via `use_score`.

This is deliberately the SAME kind of quantity CryptoMind already has from OKX
(long_short_ratio) — which is the whole point of the orthogonality test: is a
*curated* crowd's lean any better than the whole market's lean we already use?
"""
from __future__ import annotations
from typing import List, Dict, Tuple
from .collector import InvoSnapshot


def snapshot_leans(snap: InvoSnapshot, rank_decay: float = 1.0,
                   use_score: bool = False) -> Dict[str, float]:
    """Per-asset net lean in [-1,1] for a single snapshot."""
    num: Dict[str, float] = {}
    den: Dict[str, float] = {}
    for tr in snap.traders:
        w = 1.0 / (max(1, tr.rank) ** rank_decay)
        if use_score:
            w *= max(0.0, tr.score)
        for p in tr.positions:
            contrib = w * p.notional
            num[p.asset] = num.get(p.asset, 0.0) + contrib * (1 if p.direction > 0
                                                              else -1 if p.direction < 0
                                                              else 0)
            den[p.asset] = den.get(p.asset, 0.0) + abs(contrib)
    return {a: (num[a] / den[a]) for a in num if den.get(a, 0.0) > 0}


def signal_series(snaps: List[InvoSnapshot], rank_decay: float = 1.0,
                  use_score: bool = False) -> Dict[str, List[Tuple[float, float]]]:
    """{asset: [(ts, lean), ...]} across all snapshots, sorted by ts."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    for snap in snaps:
        for a, lean in snapshot_leans(snap, rank_decay, use_score).items():
            out.setdefault(a, []).append((snap.ts, lean))
    for a in out:
        out[a].sort(key=lambda x: x[0])
    return out


def _price_at(prices: List[list], ts: float) -> float | None:
    """Most recent close at-or-before ts (no lookahead)."""
    lo, hi, best = 0, len(prices) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if prices[mid][0] <= ts:
            best = prices[mid][1]; lo = mid + 1
        else:
            hi = mid - 1
    return best


def _price_after(prices: List[list], ts: float) -> Tuple[float, float] | None:
    """First close strictly after ts, with its timestamp (for the forward leg)."""
    for t, c in prices:
        if t > ts:
            return t, c
    return None


def align_forward_returns(sig: Dict[str, List[Tuple[float, float]]],
                          prices: Dict[str, list],
                          horizon_sec: float) -> Dict[str, List[Tuple[float, float, float]]]:
    """For each (ts, lean) produce (lean, forward_return, ts) where forward_return
    is close(ts+horizon)/close(ts) - 1 using no-lookahead price lookups.
    Assets absent from `prices` are dropped (you don't trade them → useless)."""
    aligned: Dict[str, List[Tuple[float, float, float]]] = {}
    for asset, series in sig.items():
        px = prices.get(asset)
        if not px:
            continue
        rows = []
        for ts, lean in series:
            p0 = _price_at(px, ts)
            fut = _price_after(px, ts + horizon_sec - 1e-6)
            if p0 is None or fut is None or p0 <= 0:
                continue
            rows.append((lean, fut[1] / p0 - 1.0, ts))
        if rows:
            aligned[asset] = rows
    return aligned
