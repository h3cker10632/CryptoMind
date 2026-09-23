"""Per-coin realized-performance tracking (freqtrade PerformanceFilter idea).

The dynamic universe ranks discovered coins purely by mention-heat + capital-flow
signals — i.e. by how much the CROWD is talking about a coin, never by how it has
actually TRADED FOR US. freqtrade's PerformanceFilter sorts pairs by past trade
performance (winners first, untested next, losers last); this brings that idea in
as our own code.

It is a pure, stateless helper over `broker.closed_trades`: given the recent
closed-trade history it returns each coin's net realized edge and a bounded
RANKING MULTIPLIER the universe applies on top of heat, so a chronic loser is
demoted out of the contested slots and a proven winner is nudged up — without
ever touching the CORE coins (the universe caller excludes those).

Kept deliberately simple and side-effect-free so it composes with the existing
heat/multi-source ranking and is trivial to test.
"""


def _sym(product):
    return product.split("-")[0] if product else product


def performance_scores(closed_trades, lookback_sec, now, min_trades=3):
    """Aggregate realized performance per SYMBOL over the rolling window.

    Returns {SYM: {"n": int, "net": float, "net_ratio": float}} where net_ratio
    is net PnL divided by the summed entry stake (a return, comparable across
    coins of different price). Only symbols with >= 1 trade in the window appear.
    """
    cutoff = now - lookback_sec
    agg = {}
    for t in closed_trades or ():
        if t.get("closed", 0) < cutoff:
            continue
        s = _sym(t.get("product"))
        if not s:
            continue
        rec = agg.setdefault(s, {"n": 0, "net": 0.0, "stake": 0.0})
        rec["n"] += 1
        rec["net"] += t.get("pnl", 0.0)
        rec["stake"] += abs(t.get("qty", 0.0)) * t.get("entry", 0.0)
    out = {}
    for s, rec in agg.items():
        ratio = (rec["net"] / rec["stake"]) if rec["stake"] > 0 else 0.0
        out[s] = {"n": rec["n"], "net": round(rec["net"], 4),
                  "net_ratio": round(ratio, 6)}
    return out


def rank_multiplier(scores, sym, min_trades=3, max_boost=1.5, min_mult=0.3):
    """Bounded multiplier applied to a discovered coin's heat when ranking for
    universe slots. 1.0 until there is enough evidence, then:
      * proven WINNER (net_ratio > 0) -> up to `max_boost`
      * chronic LOSER  (net_ratio < 0) -> down to `min_mult`
    Saturates at +-5% realized return so a single outlier can't dominate.
    """
    rec = scores.get(sym)
    if not rec or rec["n"] < min_trades:
        return 1.0
    r = rec["net_ratio"]
    # scale: +5% net -> full boost, -5% net -> full penalty, clamped
    span = max(-1.0, min(1.0, r / 0.05))
    if span >= 0:
        return 1.0 + (max_boost - 1.0) * span
    return 1.0 + (1.0 - min_mult) * span      # span<0 -> below 1.0, floored


def performance_table(closed_trades, lookback_sec, now, min_trades=3):
    """Dashboard-friendly view: symbols split into winners / untested / losers,
    mirroring freqtrade's sort order, with the multiplier each would receive."""
    scores = performance_scores(closed_trades, lookback_sec, now, min_trades)
    rows = []
    for s, rec in scores.items():
        rows.append({"sym": s, "n": rec["n"], "net": rec["net"],
                     "net_ratio": rec["net_ratio"],
                     "mult": round(rank_multiplier(scores, s, min_trades), 3)})
    # winners first (positive & enough trades), then untested, then losers
    def _key(row):
        if row["n"] < min_trades:
            return (1, 0.0)
        return (0 if row["net_ratio"] >= 0 else 2, -row["net_ratio"])
    rows.sort(key=_key)
    return rows
