"""Genetic strategy evolution — evolves trading-rule parameters against real
historical backtests and promotes the champion genome to live trading.

Genome (7 genes):
  ema_fast, ema_slow        trend filter periods
  rsi_buy                   oversold entry assist threshold
  breakout_n                breakout lookback
  stop_atr, take_atr        exit multiples
  mom_w                     momentum-confirmation weight

Selection is MULTI-OBJECTIVE (NSGA-II) over {return, -drawdown, per-trade
Sharpe}: the GA evolves the whole Pareto front of best compromises rather than
one Calmar scalar. Promotion runs the front through a PURGED WALK-FORWARD (five
sequential OOS windows with an embargo gap) and a deflated-Sharpe gate that
prices the multiple-testing bias, then promotes a top-k champion PORTFOLIO whose
votes are averaged live. The backtest sizes positions with the SAME risk-manager
policy the live book uses (train/live parity). Runs in a thread so the event
loop never blocks.
"""
import random, math, statistics, time

GENE_SPACE = {
    "ema_fast":  (5, 20),
    "ema_slow":  (21, 60),
    "rsi_buy":   (20, 45),
    "rsi_sell":  (55, 80),     # overbought threshold for SHORT entries
    "breakout_n": (10, 40),
    "stop_atr":  (1.0, 3.5),
    "take_atr":  (1.5, 6.0),
    "mom_w":     (0.0, 1.0),
    "short_w":   (0.0, 1.0),   # >0.5 = genome may short downtrends
    # --- market-structure genes (candle-derived, no lookahead) ---------------
    # All are OPT-IN via a weight gene > 0.5, so evolution can fully disable them
    # and the enriched search space stays a strict superset of the old behaviour.
    "htf_n":     (24, 120),    # higher-timeframe trend EMA span (bars, ~1-5 days)
    "htf_w":     (0.0, 1.0),   # >0.5 = require macro-trend agreement (HTF filter)
    "volz_w":    (0.0, 1.0),   # >0.5 = require a favourable volatility regime
    "volz_max":  (0.8, 2.5),   # max ATR / trailing-median-ATR allowed to enter
    # --- regime filter (ga_regime_filter) ------------------------------------
    # Kaufman efficiency-ratio gate: only enter when the trailing ER >= er_min.
    # MEASURED: the trend-following edge is concentrated in trending regimes, so
    # evolution can raise er_min to skip low-quality chop. er_min ~ 0 is inert,
    # so this is a CONTINUOUS filter (not a binary gate that starves trades).
    "er_n":      (24, 96),     # efficiency-ratio window (bars)
    "er_min":    (0.0, 0.35),  # min trailing ER to allow an entry (0 = no filter)
}
INT_GENES = {"ema_fast", "ema_slow", "rsi_buy", "rsi_sell", "breakout_n",
             "htf_n", "er_n"}

# Trailing window (bars) for the volatility-regime normaliser (~3 days hourly).
_VOL_WINDOW = 72


# Genes pinned to an inert value unless their controlling tunable is enabled, so
# the DEFAULT search space (and every promoted champion) stays bit-identical to
# the pre-enrichment behaviour. {tunable_name: {gene_name: inert_value}}.
_PINNED = {
    # HTF/vol gates — MEASURED to starve trades and collapse DSR (default off).
    "ga_market_structure": {"htf_w": 0.0, "volz_w": 0.0},
    # Efficiency-ratio regime filter — er_min=0 disables the filter entirely.
    "ga_regime_filter": {"er_min": 0.0},
}


def _tv_on(name):
    try:
        from ..tunables import tv
        return tv(name) > 0
    except Exception:
        return False


def _apply_ms_flag(g):
    """Pin experimental genes to their inert value unless their tunable is on."""
    for tunable, pins in _PINNED.items():
        if not _tv_on(tunable):
            for k, inert in pins.items():
                if k in g:
                    g[k] = inert
    return g


def random_genome(rnd):
    g = {}
    for k, (lo, hi) in GENE_SPACE.items():
        v = rnd.uniform(lo, hi)
        g[k] = int(round(v)) if k in INT_GENES else round(v, 3)
    return _apply_ms_flag(g)


def _rand_gene(k, lo, hi, rnd):
    v = rnd.uniform(lo, hi)
    return int(round(v)) if k in INT_GENES else round(v, 3)


def normalize_genome(g, rnd):
    """Return a copy of ``g`` with every GENE_SPACE key present.

    Legacy champions persisted before a gene was added (or hand-built genomes)
    may omit newer genes. Filling the gaps with a random in-range draw lets such
    a seed enter the population and start exploring the new gene, and keeps every
    GENE_SPACE-iterating code path (dedup keys, crossover) KeyError-free.
    """
    out = dict(g)
    for k, (lo, hi) in GENE_SPACE.items():
        if k not in out:
            out[k] = _rand_gene(k, lo, hi, rnd)
    return _apply_ms_flag(out)


def mutate(g, rnd, rate=0.35):
    out = dict(g)
    for k, (lo, hi) in GENE_SPACE.items():
        if k not in out:
            # a gene absent from the seed (e.g. a legacy champion predating a
            # newly-added gene) is introduced with a random draw so the lineage
            # can start exploring it — never a KeyError.
            out[k] = _rand_gene(k, lo, hi, rnd)
            continue
        if rnd.random() < rate:
            span = (hi - lo) * 0.25
            v = out[k] + rnd.gauss(0, span)
            v = max(lo, min(hi, v))
            out[k] = int(round(v)) if k in INT_GENES else round(v, 3)
    return _apply_ms_flag(out)


def crossover(a, b, rnd):
    out = {}
    for k, (lo, hi) in GENE_SPACE.items():
        first, second = (a, b) if rnd.random() < 0.5 else (b, a)
        if k in first:
            out[k] = first[k]
        elif k in second:              # only one parent carries the gene
            out[k] = second[k]
        else:                          # neither parent has it (both legacy)
            out[k] = _rand_gene(k, lo, hi, rnd)
    return _apply_ms_flag(out)


# ---------------- NSGA-II multi-objective selection ----------------
#
# The old GA collapsed everything into one Calmar-like scalar, which quietly
# encodes a fixed profit/risk trade-off and lets a high-return / high-drawdown
# genome dominate. NSGA-II instead evolves the whole PARETO FRONT of
# non-dominated genomes across several conflicting objectives, so the search
# keeps a diverse set of "best compromises" (e.g. a calm low-return genome AND
# a punchier one) rather than converging on a single risk appetite.

def objectives(res):
    """Objective vector to MAXIMISE. All three genuinely conflict:
      * total_return    — raw profit (sizing-dependent magnitude)
      * -max_drawdown   — capital preservation
      * sharpe(trades)  — per-trade consistency (scale-free)
    A genome with too few trades is pushed to the worst corner so it can't ride
    the front on luck."""
    from ..backtest.stats import sharpe
    if res["n_trades"] < 3:
        return (-1.0, -1.0, -1.0)
    return (res["total_return"], -res["max_drawdown"], sharpe(res.get("trades", [])))


def _dominates(a, b):
    """True if objective vector a Pareto-dominates b (>= on all, > on one)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _fast_non_dominated_sort(objs):
    """Return list of fronts; each front is a list of indices into objs."""
    n = len(objs)
    S = [[] for _ in range(n)]
    ndom = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                S[p].append(q)
            elif _dominates(objs[q], objs[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def _crowding_distance(front, objs):
    """Crowding distance per index in `front` (front spread preservation)."""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    m = len(objs[front[0]])
    for k in range(m):
        order = sorted(front, key=lambda i: objs[i][k])
        dist[order[0]] = dist[order[-1]] = float("inf")
        lo, hi = objs[order[0]][k], objs[order[-1]][k]
        span = (hi - lo) or 1e-9
        for j in range(1, len(order) - 1):
            dist[order[j]] += (objs[order[j + 1]][k] - objs[order[j - 1]][k]) / span
    return dist


def _nsga2_select(pop, objs, k, rnd):
    """Pick k genomes by NSGA-II ranking: fill by front, break ties within the
    boundary front by descending crowding distance (favour diversity)."""
    fronts = _fast_non_dominated_sort(objs)
    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            cd = _crowding_distance(front, objs)
            front_sorted = sorted(front, key=lambda i: -cd[i])
            chosen.extend(front_sorted[:k - len(chosen)])
            break
    return [pop[i] for i in chosen], fronts


# ---------------- walk-forward / purged validation ----------------

def walk_forward_eval(genome, candles, n_windows=5, embargo=70, sizing="risk"):
    """Evaluate a genome across several SEQUENTIAL out-of-sample windows with an
    embargo gap between them (purged walk-forward). A single train/valid split
    can be a fluke; demanding positive OOS across most windows is the standard
    guard against a curve-fit champion.

    Returns per-window results plus a summary: fraction of windows profitable,
    mean/median OOS return, worst drawdown, and the pooled per-trade Sharpe."""
    from ..backtest.stats import sharpe
    n = len(candles)
    usable = n - embargo
    if usable < n_windows * 80:               # too little data to be meaningful
        res = simulate(genome, candles, sizing=sizing)
        return {"windows": [res], "n_windows": 1,
                "frac_positive": 1.0 if res["total_return"] > 0 else 0.0,
                "mean_return": res["total_return"],
                "median_return": res["total_return"],
                "worst_drawdown": res["max_drawdown"],
                "pooled_sharpe": sharpe(res.get("trades", [])),
                "total_trades": res["n_trades"]}
    win = usable // n_windows
    results, pooled_trades = [], []
    for w in range(n_windows):
        s = embargo + w * win
        e = s + win if w < n_windows - 1 else n
        seg = candles[max(0, s - embargo):e]
        r = simulate(genome, seg, sizing=sizing)
        results.append(r)
        pooled_trades.extend(r.get("trades", []))
    rets = [r["total_return"] for r in results]
    pos = sum(1 for x in rets if x > 0)
    rets_sorted = sorted(rets)
    med = rets_sorted[len(rets_sorted) // 2]
    return {"windows": results, "n_windows": n_windows,
            "frac_positive": pos / n_windows,
            "mean_return": sum(rets) / n_windows,
            "median_return": med,
            "worst_drawdown": max(r["max_drawdown"] for r in results),
            "pooled_sharpe": sharpe(pooled_trades),
            "total_trades": sum(r["n_trades"] for r in results)}


def pooled_train_objectives(genome, train_map, sizing="risk", target=20):
    """NSGA-II objective vector for a genome evaluated ACROSS the universe.

    Simulates the genome on every product's TRAIN segment and pools the trades,
    so the search rewards an edge that GENERALISES across assets rather than one
    that curve-fits a single product's handful of trades. Objectives (maximise):
      * mean per-product total_return  — profit that holds across the basket
      * -worst per-product drawdown    — capital preservation on the weakest name
      * pooled per-trade Sharpe        — scale-free consistency over ALL trades
      * trade adequacy                 — min(n_pooled, target)/target, saturating
                                         so it rewards enough evidence but never
                                         encourages over-trading past the floor
    """
    from ..backtest.stats import sharpe
    pooled, rets, worst_dd = [], [], 0.0
    for tr in train_map.values():
        r = simulate(genome, tr, sizing=sizing)
        pooled.extend(r.get("trades", []))
        rets.append(r["total_return"])
        worst_dd = max(worst_dd, r["max_drawdown"])
    if len(pooled) < 3:
        return (-1.0, -1.0, -1.0, -1.0)
    return (sum(rets) / len(rets), -worst_dd, sharpe(pooled),
            min(len(pooled), target) / target)


def pooled_walk_forward_eval(genome, candles_map, n_windows=5, embargo=70,
                             sizing="risk"):
    """Purged walk-forward run on EVERY product, with all OOS trades POOLED.

    For a genuinely low-frequency edge, per-product evidence is too thin to clear
    a 20-trade floor without over-trading. Pooling the out-of-sample trades across
    the whole universe meets the evidence floor by BREADTH instead, and a genome
    that survives it has an edge that repeats across many independent assets — a
    far stronger, less overfit signal than one asset's luck.

    Positivity is judged per (product x window) SEGMENT so frac_positive still
    measures consistency, not a single blended curve.
    """
    from ..backtest.stats import sharpe
    per_product, pooled_trades, seg_returns = {}, [], []
    worst_dd = 0.0
    for prod, candles in candles_map.items():
        wf = walk_forward_eval(genome, candles, n_windows=n_windows,
                               embargo=embargo, sizing=sizing)
        per_product[prod] = {"total_trades": wf["total_trades"],
                             "pooled_sharpe": round(wf["pooled_sharpe"], 4),
                             "frac_positive": wf["frac_positive"]}
        for w in wf["windows"]:
            pooled_trades.extend(w.get("trades", []))
            seg_returns.append(w["total_return"])
        worst_dd = max(worst_dd, wf["worst_drawdown"])
    n_seg = len(seg_returns) or 1
    sr = sorted(seg_returns)
    n_prod = len(per_product) or 1
    # cross-sectional robustness: fraction of PRODUCTS whose OOS edge is positive.
    # For a pooled low-frequency edge this is the meaningful consistency measure
    # (does it generalise across the universe?), not per-segment frac_positive.
    frac_products_positive = sum(
        1 for pp in per_product.values() if pp["pooled_sharpe"] > 0) / n_prod
    mean_trade = sum(pooled_trades) / len(pooled_trades) if pooled_trades else 0.0
    return {"total_trades": len(pooled_trades),
            "pooled_sharpe": sharpe(pooled_trades),
            "mean_trade": mean_trade,
            "frac_positive": sum(1 for r in seg_returns if r > 0) / n_seg,
            "frac_products_positive": frac_products_positive,
            "mean_return": sum(seg_returns) / n_seg,
            "median_return": sr[len(sr) // 2] if sr else 0.0,
            "worst_drawdown": worst_dd,
            "n_products": len(candles_map),
            "n_segments": len(seg_returns),
            "per_product": per_product,
            "pooled_trades": pooled_trades}


# ---------------- genome simulation on candle history ----------------

try:
    import numpy as _np
except Exception:                       # numpy optional — pure-Python fallback
    _np = None

# Optional Numba JIT for the EMA recurrence. EMA is a sequential first-order IIR
# filter (each value depends on the last), so it can't be numpy-vectorized
# without numerical-stability tricks — but a JIT-compiled scalar loop IS bit-
# identical to the pure-Python arithmetic (verified: 0.0 max error) and ~3.5x
# faster. Measurement showed _ema_series (3 calls/genome) was the single biggest
# pure-Python cost in the GA fitness loop, so this is where the speedup lives.
_ema_kernel = None
if _np is not None:
    try:
        from numba import njit as _njit

        @_njit(cache=True, fastmath=False)
        def _ema_kernel(x, k):          # noqa: F811 (defined only when numba present)
            out = _np.empty(x.shape[0])
            out[0] = x[0]
            one_minus_k = 1.0 - k
            for i in range(1, x.shape[0]):
                out[i] = x[i] * k + out[i - 1] * one_minus_k
            return out
    except Exception:                    # numba missing/broken -> pure-Python path
        _ema_kernel = None


def _ema_series(xs, n):
    """Exponential moving average (adjust=False, seeded with the first value).

    Uses the Numba kernel when available (bit-identical, ~3.5x faster), else the
    pure-Python recurrence. Always returns a plain list so callers that index it
    like the old locals are unaffected.
    """
    if not xs:
        return []
    k = 2.0 / (n + 1)
    if _ema_kernel is not None:
        try:
            return _ema_kernel(_np.asarray(xs, dtype=float), k).tolist()
        except Exception:
            pass
    out = [xs[0]]
    for x in xs[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def _precompute_indicators(candles, genome):
    """Vectorized indicator precompute for one genome over a candle history.

    Computes EMA(fast/slow), ATR(14), RSI(14), and rolling breakout high/low
    ONCE with NumPy instead of recomputing each window inside the per-bar trade
    loop (the old O(bars x window) hot path). Results are numerically identical
    to the original per-bar arithmetic to floating-point epsilon (verified), so
    trade decisions and promoted champions are unchanged.

    Returns a dict of Python lists (the sequential loop indexes them exactly
    like the old locals). Returns None if NumPy is unavailable, so the caller
    falls back to the original per-bar computation.
    """
    if _np is None:
        return None
    n = len(candles)
    arr = _np.asarray(candles, dtype=float)
    lows, highs, closes = arr[:, 1], arr[:, 2], arr[:, 4]
    n_bo = genome["breakout_n"]

    # ATR(14): true range then trailing 14-mean (bar i uses j in [i-13, i],
    # prev-close from i-14; matches the original loop exactly for i >= 14).
    prev_c = _np.empty(n)
    prev_c[0] = closes[0]
    prev_c[1:] = closes[:-1]
    tr = _np.maximum(highs - lows,
                     _np.maximum(_np.abs(highs - prev_c), _np.abs(lows - prev_c)))
    cs_tr = _np.concatenate(([0.0], _np.cumsum(tr)))
    atr = _np.zeros(n)
    if n > 14:
        idx = _np.arange(14, n)
        atr[idx] = (cs_tr[idx + 1] - cs_tr[idx - 13]) / 14.0

    # RSI(14): trailing 14-mean of gains/losses (same window as ATR).
    diff = _np.diff(closes, prepend=closes[0])
    gain = _np.clip(diff, 0, None)
    loss = _np.clip(-diff, 0, None)
    cs_g = _np.concatenate(([0.0], _np.cumsum(gain)))
    cs_l = _np.concatenate(([0.0], _np.cumsum(loss)))
    rsi = _np.zeros(n)
    if n > 14:
        idx = _np.arange(14, n)
        ag = (cs_g[idx + 1] - cs_g[idx - 13]) / 14.0
        al = (cs_l[idx + 1] - cs_l[idx - 13]) / 14.0
        with _np.errstate(divide="ignore", invalid="ignore"):
            r = 100.0 - 100.0 / (1.0 + ag / al)
        r[al == 0] = 100.0
        rsi[idx] = r

    # Rolling breakout high/low: for bar i, max(high[i-n_bo:i]) / min(low[...]).
    brk_up = _np.zeros(n)
    brk_dn = _np.zeros(n)
    if n_bo >= 1 and n > n_bo:
        from numpy.lib.stride_tricks import sliding_window_view
        sw_hi = sliding_window_view(highs, n_bo).max(axis=1)   # sw_hi[s]=max hi[s:s+n_bo]
        sw_lo = sliding_window_view(lows, n_bo).min(axis=1)
        idx = _np.arange(n_bo, n)
        brk_up[idx] = sw_hi[idx - n_bo]                        # window [i-n_bo, i-1]
        brk_dn[idx] = sw_lo[idx - n_bo]

    # The next three market-structure indicators are OPT-IN: simulate() only
    # reads each one when the genome activates its gate (htf_w / volz_w / er_min).
    # Computing them for a genome that never uses them is pure waste — and the
    # rolling-median vol_norm is the single most expensive precompute step — so
    # each is computed lazily only when its gate is on. Skipping an unused array
    # is bit-identical (its values are never read). A genome with the gate off is
    # returned None here and simulate()'s short-circuit guards never index it.
    closes_list = closes.tolist()

    # Higher-timeframe trend proxy: a long EMA over the SAME hourly closes.
    # Comparing close vs this slow EMA is a macro-trend agreement filter that
    # needs no coarser candles and introduces no lookahead.
    htf_ema = None
    if genome.get("htf_w", 0.0) > 0.5:
        htf_ema = _ema_series(closes_list, int(genome.get("htf_n", 48)))

    # Volatility regime: ATR normalised by its own TRAILING median (window
    # [i-VOL_WINDOW, i-1], strictly past bars -> no lookahead). ~1.0 = typical
    # vol, >1 = elevated/chaotic. Genomes can gate entries to calm regimes.
    vol_norm = None
    if genome.get("volz_w", 0.0) > 0.5:
        vol_norm = _np.ones(n)
        if n > _VOL_WINDOW:
            from numpy.lib.stride_tricks import sliding_window_view
            med = _np.median(sliding_window_view(atr, _VOL_WINDOW), axis=1)  # med[s]=median atr[s:s+W]
            idx = _np.arange(_VOL_WINDOW, n)
            base = med[idx - _VOL_WINDOW]                                    # window [i-W, i-1]
            with _np.errstate(divide="ignore", invalid="ignore"):
                vn = atr[idx] / base
            vn[~_np.isfinite(vn)] = 1.0
            vol_norm[idx] = vn
        vol_norm = vol_norm.tolist()

    # Kaufman efficiency ratio (trailing, causal) for the regime filter.
    er = None
    if genome.get("er_min", 0.0) > 0.0:
        from .regime import efficiency_ratio
        er = efficiency_ratio(closes_list, int(genome.get("er_n", 48)))

    return {"ema_fast": _ema_series(closes_list, genome["ema_fast"]),
            "ema_slow": _ema_series(closes_list, genome["ema_slow"]),
            "atr": atr.tolist(), "rsi": rsi.tolist(),
            "brk_up": brk_up.tolist(), "brk_dn": brk_dn.tolist(),
            "htf_ema": htf_ema, "vol_norm": vol_norm,
            "er": er, "closes": closes_list}


def simulate(genome, candles, fee=None, slip=None, start_cash=10_000.0,
             sizing="risk", trade_log=None):
    """Event-driven sim; LONG and (if short_w > 0.5) SHORT trades.
    side: 0 flat, +1 long, -1 short (margin-style shorts).

    sizing controls how much capital each entry deploys:
      * "risk"    — MIRRORS the live risk manager: risk a fixed fraction of
                    equity to the genome's own ATR stop, then cap at
                    max_position_pct. This is the default so the backtest the GA
                    optimises is the SAME strategy that runs live (train/live
                    parity). A genome that only looks good at 95%-of-cash
                    leverage no longer gets promoted to a book that sizes at a
                    fraction of a percent of equity per trade.
      * "fullcash"— legacy 95%-of-cash sizing (kept for comparison/tests).
    """
    from ..tunables import tv
    fee = tv("fee_rate") if fee is None else fee
    slip = (tv("slippage_bps") / 1e4) if slip is None else slip
    if sizing == "risk":
        risk_frac = tv("risk_per_trade")
        max_pos = tv("max_position_pct")
    else:
        risk_frac = max_pos = None

    def _entry_notional(equity_now, px, stop_dist):
        """Notional for one entry under the active sizing policy."""
        if sizing != "risk":
            return equity_now * 0.95
        if stop_dist <= 0:
            return 0.0
        risk_dollars = equity_now * risk_frac
        notional = risk_dollars / (stop_dist / px)   # = risk$ * px / stop_dist
        # never exceed the position cap or available cash
        return min(notional, equity_now * max_pos, equity_now * 0.95)
    closes = [c[4] for c in candles]
    n_bo = genome["breakout_n"]
    can_short = genome.get("short_w", 0) > 0.5

    # Vectorized indicator precompute (NumPy) — identical arithmetic to the old
    # per-bar loop, computed once. Falls back to per-bar computation if NumPy is
    # unavailable (`ind` stays None).
    ind = _precompute_indicators(candles, genome)
    if ind is not None:
        ef, es = ind["ema_fast"], ind["ema_slow"]
        _atr, _rsi = ind["atr"], ind["rsi"]
        _brk_up_lvl, _brk_dn_lvl = ind["brk_up"], ind["brk_dn"]
        _htf, _voln = ind["htf_ema"], ind["vol_norm"]
        _er = ind["er"]
    else:
        ef = _ema_series(closes, genome["ema_fast"])
        es = _ema_series(closes, genome["ema_slow"])
        _atr = _rsi = _brk_up_lvl = _brk_dn_lvl = None
        # HTF trend + ER still work without NumPy; vol regime needs it -> off.
        _htf = _ema_series(closes, int(genome.get("htf_n", 48)))
        from .regime import efficiency_ratio
        _er = efficiency_ratio(closes, int(genome.get("er_n", 48)))
        _voln = None

    # Opt-in market-structure gates (disabled unless the weight gene > 0.5).
    htf_on = genome.get("htf_w", 0.0) > 0.5
    vol_on = genome.get("volz_w", 0.0) > 0.5
    vol_max = genome.get("volz_max", 99.0)
    # Regime filter: only enter when trailing ER >= er_min (0 = inert).
    er_min = genome.get("er_min", 0.0)

    cash, side, qty, entry, stop, take, margin = start_cash, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    entry_i = 0                       # bar index of the current open entry
    eq, trades = [], []
    # Only extend the warmup for a market-structure feature when it is actually
    # ACTIVE, so a genome with the gates disabled is bit-identical to a legacy
    # genome that never had these genes (strict superset behaviour).
    warm = max(genome["ema_slow"], n_bo, 15,
               int(genome.get("htf_n", 48)) if htf_on else 0,
               _VOL_WINDOW if vol_on else 0,
               int(genome.get("er_n", 48)) if er_min > 0 else 0) + 1

    def close_pos(px):
        nonlocal cash, side, qty, margin
        if side > 0:
            cash += qty * px * (1 - fee)
            trades.append(px / entry - 1)
        else:
            move = qty * (entry - px)
            cash += margin + move - qty * px * fee
            trades.append((entry - px) / entry)
        # optional instrumentation: record (entry_bar_index, side) per closed
        # trade so callers can bucket P&L by the regime at entry. No overhead
        # unless a list is supplied.
        if trade_log is not None:
            trade_log.append((entry_i, side))
        side, qty, margin = 0, 0.0, 0.0

    for i in range(warm, len(candles)):
        ts, lo, hi, op, cl, vol = candles[i]

        # ATR(14) — precomputed (NumPy) or per-bar fallback
        if _atr is not None:
            atr = _atr[i]
        else:
            trs = []
            for j in range(i - 13, i + 1):
                phi, plo, pc = candles[j][2], candles[j][1], candles[j - 1][4]
                trs.append(max(phi - plo, abs(phi - pc), abs(plo - pc)))
            atr = sum(trs) / 14

        # exits (intra-bar)
        if side > 0:
            if lo <= stop:
                close_pos(stop * (1 - slip))
            elif hi >= take:
                close_pos(take * (1 - slip))
        elif side < 0:
            if hi >= stop:
                close_pos(stop * (1 + slip))
            elif lo <= take:
                close_pos(take * (1 + slip))

        if side == 0 and atr > 0:
            # RSI(14) — precomputed (NumPy) or per-bar fallback
            if _rsi is not None:
                rsi = _rsi[i]
            else:
                gains = [max(closes[j] - closes[j - 1], 0) for j in range(i - 13, i + 1)]
                losses = [max(closes[j - 1] - closes[j], 0) for j in range(i - 13, i + 1)]
                ag, al = sum(gains) / 14, sum(losses) / 14
                rsi = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)

            up_trend = ef[i] > es[i]
            if _brk_up_lvl is not None:
                brk_up = cl >= _brk_up_lvl[i]
                brk_dn = cl <= _brk_dn_lvl[i]
            else:
                brk_up = cl >= max(c[2] for c in candles[i - n_bo:i])
                brk_dn = cl <= min(c[1] for c in candles[i - n_bo:i])
            mom_up = closes[i] > closes[i - 12] if genome["mom_w"] > 0.5 else True
            mom_dn = closes[i] < closes[i - 12] if genome["mom_w"] > 0.5 else True

            # market-structure gates (opt-in; pass-through when weight <= 0.5)
            htf_ok_long = (not htf_on) or (cl > _htf[i])
            htf_ok_short = (not htf_on) or (cl < _htf[i])
            vol_ok = (not vol_on) or (_voln is None) or (_voln[i] <= vol_max)
            # regime filter: strong-enough trend to trade (er_min 0 = inert)
            regime_ok = (er_min <= 0.0) or (_er[i] >= er_min)

            if up_trend and (brk_up or rsi < genome["rsi_buy"]) and mom_up \
                    and htf_ok_long and vol_ok and regime_ok:
                px = cl * (1 + slip)
                notional = _entry_notional(cash, px, genome["stop_atr"] * atr)
                if notional <= 0:
                    eq.append(cash); continue
                qty = notional / px
                cash -= notional * (1 + fee)
                side, entry, entry_i = 1, px, i
                stop = px - genome["stop_atr"] * atr
                take = px + genome["take_atr"] * atr
            elif can_short and not up_trend and \
                    (brk_dn or rsi > genome.get("rsi_sell", 70)) and mom_dn \
                    and htf_ok_short and vol_ok and regime_ok:
                px = cl * (1 - slip)
                notional = _entry_notional(cash, px, genome["stop_atr"] * atr)
                if notional <= 0:
                    eq.append(cash); continue
                qty = notional / px
                margin = notional
                cash -= notional + notional * fee
                side, entry, entry_i = -1, px, i
                stop = px + genome["stop_atr"] * atr
                take = px - genome["take_atr"] * atr

        if side > 0:
            eq.append(cash + qty * cl)
        elif side < 0:
            eq.append(cash + margin + qty * (entry - cl))
        else:
            eq.append(cash)

    if side != 0:
        close_pos(closes[-1])
        eq[-1] = cash

    total = eq[-1] / start_cash - 1 if eq else 0.0
    peak, maxdd = 0.0, 0.0
    for e in eq:
        peak = max(peak, e); maxdd = max(maxdd, 1 - e / peak)
    wins = sum(1 for t in trades if t > 0)
    return {"total_return": total, "max_drawdown": maxdd,
            "n_trades": len(trades),
            "win_rate": wins / len(trades) if trades else 0.0,
            "trades": list(trades)}


def fitness(res):
    """Calmar-like with trade-count sanity: return / drawdown, penalize
    overtrading and undertrading."""
    if res["n_trades"] < 3:
        return -1.0 + res["total_return"]          # not enough evidence
    f = res["total_return"] / (res["max_drawdown"] + 0.02)
    if res["n_trades"] > 60:
        f *= 0.7
    return f


class Evolution:
    def __init__(self, pop_size=24, generations=8, seed=None):
        self.pop_size = pop_size
        self.generations = generations
        self.rnd = random.Random(seed)
        self.champions = {}           # product -> promoted genome (legacy: best)
        self.champion_portfolios = {} # product -> top-k promoted genomes
        self.champion_reports = {}    # product -> validation report
        self.attempt_reports = {}     # product -> compact LAST-attempt diagnostics
                                      # (kept for EVERY run, promoted or not, so the
                                      # operator can compare best-DSR-seen per product)
        self.last_attempt = {}        # product -> ts of last GA run (any outcome)
        self.history = []             # per-generation best fitness (last run)
        self.status = "idle"
        self.current_product = None
        self.generation = 0
        self.last_run = None
        self.last_universe_run = None   # most recent cross-sectional report (any outcome)

    # backward-compat: BTC champion as the generic fallback
    @property
    def champion(self):
        return self.champions.get("BTC-USD")

    def champion_for(self, product):
        return self.champions.get(product) or self.champions.get("BTC-USD")

    def portfolio_for(self, product):
        """Return the promoted champion PORTFOLIO (top-k genomes) for a product,
        falling back to BTC-USD's, then to the single champion as a 1-element
        list. Callers can average these genomes' votes for a more robust,
        less overfit signal than betting on one champion."""
        p = (self.champion_portfolios.get(product)
             or self.champion_portfolios.get("BTC-USD"))
        if p:
            return p
        g = self.champion_for(product)
        return [g] if g else []

    def evolve(self, candles, product="BTC-USD"):
        """Full GA run with train/validation split. Called from a thread."""
        self.status = "running"
        self.current_product = product
        self.history = []
        # Reproducibility: reseed this run deterministically from the global
        # validation seed + product, so the champion a promotion gate lets live
        # can be re-derived and audited. Independent per product; nondeterministic
        # when VALIDATION_SEED < 0. (Falls back to the existing rnd on any error.)
        try:
            from ..backtest.seeding import rng as _seed_rng
            self.rnd = _seed_rng("ga", product)
        except Exception:
            pass
        split = int(len(candles) * 0.65)
        train, valid = candles[:split], candles[split - 70:]

        # WARM-START: seed the population with the standing champion (if any) and
        # a spread of its mutants, instead of starting from scratch every run.
        # This turns each GA run into CONTINUED search around known-good genomes
        # rather than 192 wasted evaluations rediscovering the basics — a big
        # sample-efficiency win, and it keeps good solutions from being lost.
        pop = []
        seed_g = self.champions.get(product) or self.champions.get("BTC-USD")
        if seed_g:
            seed_g = normalize_genome(seed_g, self.rnd)   # fill any newly-added genes
            pop.append(dict(seed_g))
            for _ in range(max(1, self.pop_size // 3)):
                pop.append(mutate(dict(seed_g), self.rnd, rate=0.5))
        while len(pop) < self.pop_size:
            pop.append(random_genome(self.rnd))
        elite_n = max(2, self.pop_size // 6)

        front_genomes = []          # last generation's Pareto front (elite pool)
        trial_sharpes = []          # EVERY genome's train Sharpe, all generations
                                    # (the real dispersion of the search, used to
                                    # deflate the champion's Sharpe honestly)
        for gen in range(self.generations):
            self.generation = gen + 1
            # ADAPTIVE MUTATION: anneal from exploratory (early gens) to fine
            # (late gens) so the search broadens first, then refines.
            frac = gen / max(1, self.generations - 1)
            mut_rate = 0.45 - 0.30 * frac        # 0.45 -> 0.15
            results = [simulate(g, train) for g in pop]
            objs = [objectives(r) for r in results]
            trial_sharpes.extend(o[2] for o in objs)   # objectives()[2] = Sharpe
            # NSGA-II: rank the population by Pareto front + crowding, keep the
            # best half as the breeding/elite pool.
            elites, fronts = _nsga2_select(pop, objs, max(elite_n, self.pop_size // 2),
                                           self.rnd)
            front_genomes = [pop[i] for i in fronts[0]]
            # scalar Calmar kept purely for human-readable history/logging.
            scal = sorted(range(len(pop)), key=lambda i: -fitness(results[i]))
            self.history.append({
                "gen": gen + 1,
                "best_fitness": round(fitness(results[scal[0]]), 4),
                "best_return": round(results[scal[0]]["total_return"], 4),
                "pareto_front": len(fronts[0]),
                "mean_fitness": round(sum(fitness(r) for r in results) / len(results), 4),
            })
            # next generation: elitism (front members) + crowded tournament
            nxt = list(elites[:elite_n])

            def _tournament():
                # binary tournament on NSGA rank (front index) then crowding.
                cand = self.rnd.sample(range(len(elites)), min(2, len(elites)))
                return elites[min(cand)]        # lower list index == better rank

            while len(nxt) < self.pop_size:
                a, b = _tournament(), _tournament()
                child = mutate(crossover(a, b, self.rnd), self.rnd, rate=mut_rate)
                nxt.append(child)
            pop = nxt

        # ---- promotion via PURGED WALK-FORWARD across the Pareto front ----
        # Rank the front by full-history in-sample fitness, then validate the
        # top candidates on a purged walk-forward (several sequential OOS
        # windows). A genome must earn OOS profit across MOST windows, not just
        # get lucky on one tail slice. We apply a MULTIPLE-TESTING aware gate
        # (the GA evaluated pop_size*generations genomes) via deflated Sharpe on
        # the pooled OOS trades.
        from ..backtest.stats import deflated_sharpe_ratio, sharpe
        import statistics as _stats
        # MULTIPLE-TESTING inputs for the deflated Sharpe, measured from THIS run
        # rather than assumed. n_trials = the genomes we actually evaluated;
        # trial_sr_std = the real dispersion of their Sharpes across ALL
        # generations (including the diverse early ones — NOT just the converged
        # survivors, which would understate dispersion and be over-lenient).
        # The old code hard-coded trial_sr_std=0.5, an ~2x over-estimate of the
        # dispersion for most products, which inflated the expected-max-Sharpe
        # benchmark so far that no genome could ever clear it (deflated Sharpe
        # pinned at ~0). Feeding the statistic its true input is a correctness
        # fix, not a relaxation: it RAISES the bar for products whose trials are
        # genuinely dispersed and lowers it only where they cluster tightly.
        n_trials = len(trial_sharpes) or (self.pop_size * self.generations)
        # floor guards the degenerate case (near-identical trials -> ~0 std ->
        # no deflation at all, which would be unsafe).
        TRIAL_STD_FLOOR = 0.05
        trial_sr_std = max(TRIAL_STD_FLOOR,
                           _stats.pstdev(trial_sharpes) if len(trial_sharpes) > 1
                           else TRIAL_STD_FLOOR)
        # de-dup the front and cap how many we fully walk-forward validate
        seen_keys, front = set(), []
        for g in sorted(front_genomes, key=lambda g: -fitness(simulate(g, train))):
            key = tuple(round(g[k], 3) for k in GENE_SPACE)
            if key not in seen_keys:
                seen_keys.add(key); front.append(g)
            if len(front) >= 6:
                break

        # Promotion-gate thresholds are OPERATOR-TUNABLE (dashboard → Evolution).
        # They default to strict anti-overfit values; lowering them lets the GA
        # promote on thinner / less certain evidence — a deliberate, observable
        # operator choice. The ONE bar that is NOT tunable is oos_ok: a genome
        # that is net-losing out-of-sample is never promoted, at any setting.
        from ..tunables import tv
        MIN_OOS_TRADES = int(tv("ga_min_oos_trades"))
        MIN_FRAC_POSITIVE = float(tv("ga_min_frac_positive"))   # positive OOS windows
        DSR_MIN = float(tv("ga_dsr_min"))
        FALLBACK_SHARPE_MIN = float(tv("ga_fallback_sharpe_min"))
        MAX_WF_DD = float(tv("ga_max_wf_drawdown"))
        candidates = []                   # (score, genome, wf, train_res, dsr)
        # per-gate diagnostics: tally the FIRST gate each front genome fails, so
        # the operator can SEE which bar is the blocker before lowering it.
        fails = {"too_few_trades": 0, "not_robust": 0,
                 "oos_negative": 0, "weak_sharpe": 0}
        # best metrics ACTUALLY OBSERVED across the front (even if nothing was
        # promoted), so the operator can calibrate ga_dsr_min to real data
        # instead of guessing how far below the bar the challengers landed.
        best_obs_dsr = None
        best_obs_pooled = None
        for g in front:
            train_res = simulate(g, train)
            wf = walk_forward_eval(g, candles, n_windows=5, embargo=70)
            pooled = [t for r in wf["windows"] for t in r.get("trades", [])]
            dsr = None
            try:
                if len(pooled) >= 5:
                    dsr = deflated_sharpe_ratio(pooled, n_trials=n_trials,
                                                trial_sr_std=trial_sr_std)
            except Exception:
                dsr = None
            enough = wf["total_trades"] >= MIN_OOS_TRADES
            robust = wf["frac_positive"] >= MIN_FRAC_POSITIVE
            oos_ok = wf["median_return"] > 0 and wf["mean_return"] > 0
            dd_ok = wf["worst_drawdown"] <= MAX_WF_DD
            # SCALE-FREE gate: under risk-parity sizing raw returns are tiny
            # fractions, so we judge quality by the deflated / pooled Sharpe of
            # the OOS trades, not by a return magnitude threshold.
            dsr_ok = ((dsr is not None and dsr > DSR_MIN) or
                      (dsr is None and wf["pooled_sharpe"] > FALLBACK_SHARPE_MIN and dd_ok))
            if dsr is not None and (best_obs_dsr is None or dsr > best_obs_dsr):
                best_obs_dsr = dsr
            _ps = wf.get("pooled_sharpe")
            if _ps is not None and (best_obs_pooled is None or _ps > best_obs_pooled):
                best_obs_pooled = _ps
            if not enough:
                fails["too_few_trades"] += 1
            elif not robust:
                fails["not_robust"] += 1
            elif not oos_ok:
                fails["oos_negative"] += 1
            elif not dsr_ok:
                fails["weak_sharpe"] += 1
            ok = bool(enough and robust and oos_ok and dsr_ok)
            if ok:
                # rank promotable genomes by pooled OOS Sharpe (scale-free)
                candidates.append((wf["pooled_sharpe"], g, wf, train_res, dsr))

        candidates.sort(key=lambda t: -t[0])
        TOP_K = 3                         # champion PORTFOLIO size per product

        def _wf_summary(wf):
            return {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                    for k, v in wf.items() if k != "windows"}

        promoted = bool(candidates)
        portfolio = [g for _, g, _, _, _ in candidates[:TOP_K]]
        best = candidates[0] if candidates else None
        report = {
            "product": product,
            "genome": best[1] if best else None,
            "portfolio": portfolio,
            "portfolio_size": len(portfolio),
            "train": ({k: round(v, 4) for k, v in best[3].items() if k != "trades"}
                      if best else None),
            "walk_forward": _wf_summary(best[2]) if best else None,
            "train_fitness": round(fitness(best[3]), 4) if best else None,
            "pooled_oos_sharpe": round(best[0], 4) if best else None,
            "deflated_sharpe": round(best[4], 4) if best and best[4] is not None else None,
            "n_trials": n_trials,
            "trial_sr_std": round(trial_sr_std, 4),
            "front_size": len(front),
            "n_candidates_passing": len(candidates),
            "promoted": promoted,
            "gate": {"min_oos_trades": MIN_OOS_TRADES,
                     "min_frac_positive": MIN_FRAC_POSITIVE,
                     "dsr_min": DSR_MIN,
                     "fallback_sharpe_min": FALLBACK_SHARPE_MIN,
                     "max_wf_drawdown": MAX_WF_DD},
            "gate_fail_breakdown": fails,
            "best_observed_dsr": (round(best_obs_dsr, 4)
                                  if best_obs_dsr is not None else None),
            "best_observed_pooled_sharpe": (round(best_obs_pooled, 4)
                                            if best_obs_pooled is not None else None),
            "ts": time.time(),
        }
        if promoted:
            self.champions[product] = best[1]              # legacy single champion
            self.champion_portfolios[product] = portfolio  # NEW: top-k portfolio
            self.champion_reports[product] = report
        elif product in self.champions:
            # A rerun failed to re-validate the incumbent's regime, yet the OLD
            # champion is still voting live (this is exactly the stale-SOL /
            # UNI case). Evict it: a champion that can no longer earn promotion
            # on fresh out-of-sample data has no business steering allocation.
            evicted = self.champions.pop(product, None)
            self.champion_portfolios.pop(product, None)
            self.champion_reports.pop(product, None)
            report["evicted_stale_champion"] = bool(evicted)
            from .. import db
            db.log_event("learn",
                         f"GA champion for {product} EVICTED — latest rerun "
                         f"produced no genome passing purged walk-forward "
                         f"(front {len(front)}, 0 candidates cleared the gate); "
                         f"no live vote until a genome re-validates.")
        self.last_run = report
        # compact per-product diagnostics for EVERY attempt (promoted or not),
        # so the dashboard can show best-DSR-seen across the whole universe.
        self.attempt_reports[product] = {
            "promoted": report["promoted"],
            "best_observed_dsr": report["best_observed_dsr"],
            "best_observed_pooled_sharpe": report["best_observed_pooled_sharpe"],
            "gate_fail_breakdown": report["gate_fail_breakdown"],
            "n_candidates_passing": report["n_candidates_passing"],
            "front_size": report["front_size"],
            "ts": report["ts"],
        }
        self.status = "done"
        return report

    def evolve_universe(self, candles_map, n_windows=5, embargo=70):
        """CROSS-SECTIONAL GA: search and validate a genome POOLED across the
        whole universe, so a genuinely low-frequency edge is validated by BREADTH
        (many independent assets) rather than by over-trading one product.

        MEASURED motivation: the trend edge produces only a handful of quality
        trades per product — too few to clear a 20-trade per-product evidence
        floor without destroying the edge. Pooling out-of-sample trades across a
        basket meets the floor honestly, and a genome that survives has an edge
        that REPEATS across the universe. Promotes ONE portfolio applied to every
        product in the basket (stored per-product so all downstream consumers,
        e.g. portfolio_for(), work unchanged).
        """
        self.status = "running"
        self.current_product = "(universe)"
        self.history = []
        try:
            from ..backtest.seeding import rng as _seed_rng
            self.rnd = _seed_rng("ga", "_UNIVERSE")
        except Exception:
            pass
        products = list(candles_map.keys())
        train_map = {p: c[:int(len(c) * 0.65)] for p, c in candles_map.items()}

        # warm-start from the standing universe champion, if any
        pop = []
        seed_g = self.champions.get("_UNIVERSE")
        if seed_g:
            seed_g = normalize_genome(seed_g, self.rnd)
            pop.append(dict(seed_g))
            for _ in range(max(1, self.pop_size // 3)):
                pop.append(mutate(dict(seed_g), self.rnd, rate=0.5))
        while len(pop) < self.pop_size:
            pop.append(random_genome(self.rnd))
        elite_n = max(2, self.pop_size // 6)

        from ..tunables import tv
        target = int(tv("ga_min_oos_trades"))
        front_genomes, trial_sharpes = [], []
        for gen in range(self.generations):
            self.generation = gen + 1
            frac = gen / max(1, self.generations - 1)
            mut_rate = 0.45 - 0.30 * frac
            objs = [pooled_train_objectives(g, train_map, target=target) for g in pop]
            trial_sharpes.extend(o[2] for o in objs)   # pooled train Sharpe per genome
            elites, fronts = _nsga2_select(pop, objs,
                                           max(elite_n, self.pop_size // 2), self.rnd)
            front_genomes = [pop[i] for i in fronts[0]]
            self.history.append({
                "gen": gen + 1,
                "pareto_front": len(fronts[0]),
                "best_pooled_sharpe": round(max(o[2] for o in objs), 4),
            })
            nxt = list(elites[:elite_n])

            def _tournament():
                cand = self.rnd.sample(range(len(elites)), min(2, len(elites)))
                return elites[min(cand)]

            while len(nxt) < self.pop_size:
                child = mutate(crossover(_tournament(), _tournament(), self.rnd),
                               self.rnd, rate=mut_rate)
                nxt.append(child)
            pop = nxt

        from ..backtest.stats import deflated_sharpe_ratio
        import statistics as _stats
        n_trials = len(trial_sharpes) or (self.pop_size * self.generations)
        TRIAL_STD_FLOOR = 0.05
        trial_sr_std = max(TRIAL_STD_FLOOR,
                           _stats.pstdev(trial_sharpes) if len(trial_sharpes) > 1
                           else TRIAL_STD_FLOOR)
        # de-dup the front by pooled train Sharpe
        seen_keys, front = set(), []
        for g in sorted(front_genomes,
                        key=lambda g: -pooled_train_objectives(g, train_map,
                                                               target=target)[2]):
            key = tuple(round(g[k], 3) for k in GENE_SPACE)
            if key not in seen_keys:
                seen_keys.add(key); front.append(g)
            if len(front) >= 6:
                break

        MIN_OOS_TRADES = int(tv("ga_min_oos_trades"))
        MIN_FRAC_POSITIVE = float(tv("ga_min_frac_positive"))
        DSR_MIN = float(tv("ga_dsr_min"))
        FALLBACK_SHARPE_MIN = float(tv("ga_fallback_sharpe_min"))
        MAX_WF_DD = float(tv("ga_max_wf_drawdown"))
        candidates = []
        fails = {"too_few_trades": 0, "not_robust": 0,
                 "oos_negative": 0, "weak_sharpe": 0}
        best_obs_dsr = best_obs_pooled = None
        best_obs_wf = None            # wf of the strongest front genome (even if unpromoted)
        best_obs_rank = None
        for g in front:
            wf = pooled_walk_forward_eval(g, candles_map, n_windows=n_windows,
                                          embargo=embargo)
            pooled = wf["pooled_trades"]
            dsr = None
            try:
                if len(pooled) >= 5:
                    dsr = deflated_sharpe_ratio(pooled, n_trials=n_trials,
                                                trial_sr_std=trial_sr_std)
            except Exception:
                dsr = None
            enough = wf["total_trades"] >= MIN_OOS_TRADES
            # cross-sectional robustness: edge holds on >= MIN_FRAC of PRODUCTS
            robust = wf["frac_products_positive"] >= MIN_FRAC_POSITIVE
            # hard non-losing floor: the pooled edge must be net-positive per trade
            oos_ok = wf["mean_trade"] > 0
            dd_ok = wf["worst_drawdown"] <= MAX_WF_DD
            dsr_ok = ((dsr is not None and dsr > DSR_MIN) or
                      (dsr is None and wf["pooled_sharpe"] > FALLBACK_SHARPE_MIN and dd_ok))
            if dsr is not None and (best_obs_dsr is None or dsr > best_obs_dsr):
                best_obs_dsr = dsr
            if best_obs_pooled is None or wf["pooled_sharpe"] > best_obs_pooled:
                best_obs_pooled = wf["pooled_sharpe"]
            # keep the wf of the strongest genome (by DSR, else pooled Sharpe) so
            # the dashboard can show the per-product breakdown even when nothing
            # cleared the gate — the most useful diagnostic case.
            rank = dsr if dsr is not None else wf["pooled_sharpe"]
            if best_obs_rank is None or rank > best_obs_rank:
                best_obs_rank, best_obs_wf = rank, wf
            if not enough:
                fails["too_few_trades"] += 1
            elif not robust:
                fails["not_robust"] += 1
            elif not oos_ok:
                fails["oos_negative"] += 1
            elif not dsr_ok:
                fails["weak_sharpe"] += 1
            if enough and robust and oos_ok and dsr_ok:
                candidates.append((wf["pooled_sharpe"], g, wf, dsr))

        candidates.sort(key=lambda t: -t[0])
        TOP_K = 3
        promoted = bool(candidates)
        portfolio = [g for _, g, _, _ in candidates[:TOP_K]]
        best = candidates[0] if candidates else None

        def _wf_summary(wf):
            return {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                    for k, v in wf.items() if k not in ("windows", "pooled_trades")}

        report = {
            "product": "(universe)",
            "basket": products,
            "genome": best[1] if best else None,
            "portfolio": portfolio,
            "portfolio_size": len(portfolio),
            "walk_forward": _wf_summary(best[2] if best else best_obs_wf)
                            if (best or best_obs_wf) else None,
            "pooled_oos_sharpe": round(best[0], 4) if best else None,
            "deflated_sharpe": round(best[3], 4) if best and best[3] is not None else None,
            "n_trials": n_trials,
            "trial_sr_std": round(trial_sr_std, 4),
            "front_size": len(front),
            "n_candidates_passing": len(candidates),
            "promoted": promoted,
            "gate": {"min_oos_trades": MIN_OOS_TRADES,
                     "min_frac_positive": MIN_FRAC_POSITIVE,
                     "dsr_min": DSR_MIN,
                     "fallback_sharpe_min": FALLBACK_SHARPE_MIN,
                     "max_wf_drawdown": MAX_WF_DD},
            "gate_fail_breakdown": fails,
            "best_observed_dsr": (round(best_obs_dsr, 4)
                                  if best_obs_dsr is not None else None),
            "best_observed_pooled_sharpe": (round(best_obs_pooled, 4)
                                            if best_obs_pooled is not None else None),
            "ts": time.time(),
        }
        if promoted:
            self.champions["_UNIVERSE"] = best[1]
            self.champion_portfolios["_UNIVERSE"] = portfolio
            self.champion_reports["_UNIVERSE"] = report
            # apply the universe portfolio to every product in the basket so all
            # downstream consumers (portfolio_for, voting) work unchanged
            for p in products:
                self.champions[p] = best[1]
                self.champion_portfolios[p] = portfolio
                self.champion_reports[p] = report
        self.last_run = report
        self.last_universe_run = report
        self.attempt_reports["_UNIVERSE"] = {
            "promoted": report["promoted"],
            "best_observed_dsr": report["best_observed_dsr"],
            "best_observed_pooled_sharpe": report["best_observed_pooled_sharpe"],
            "gate_fail_breakdown": report["gate_fail_breakdown"],
            "n_candidates_passing": report["n_candidates_passing"],
            "front_size": report["front_size"],
            "ts": report["ts"],
        }
        self.status = "done"
        return report

    def stats(self):
        return {"status": self.status, "generation": self.generation,
                "generations_total": self.generations,
                "population": self.pop_size,
                "current_product": self.current_product,
                "last_attempt": self.last_attempt,
                "history": self.history,
                "champions": self.champions,
                "champion_portfolios": self.champion_portfolios,
                "portfolio_sizes": {p: len(g) for p, g in
                                    self.champion_portfolios.items()},
                "champion_reports": self.champion_reports,
                "attempt_reports": self.attempt_reports,
                "champion": self.champion,                # legacy field
                "champion_report": self.champion_reports.get("BTC-USD"),
                "last_run": self.last_run,
                "last_universe_run": self.last_universe_run}


evolution = Evolution()
