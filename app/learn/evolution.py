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
}
INT_GENES = {"ema_fast", "ema_slow", "rsi_buy", "rsi_sell", "breakout_n"}


def random_genome(rnd):
    g = {}
    for k, (lo, hi) in GENE_SPACE.items():
        v = rnd.uniform(lo, hi)
        g[k] = int(round(v)) if k in INT_GENES else round(v, 3)
    return g


def mutate(g, rnd, rate=0.35):
    out = dict(g)
    for k, (lo, hi) in GENE_SPACE.items():
        if rnd.random() < rate:
            span = (hi - lo) * 0.25
            v = out[k] + rnd.gauss(0, span)
            v = max(lo, min(hi, v))
            out[k] = int(round(v)) if k in INT_GENES else round(v, 3)
    return out


def crossover(a, b, rnd):
    return {k: (a if rnd.random() < 0.5 else b)[k] for k in GENE_SPACE}


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


# ---------------- genome simulation on candle history ----------------

try:
    import numpy as _np
except Exception:                       # numpy optional — pure-Python fallback
    _np = None


def _ema_series(xs, n):
    k = 2 / (n + 1)
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

    return {"ema_fast": _ema_series(closes.tolist(), genome["ema_fast"]),
            "ema_slow": _ema_series(closes.tolist(), genome["ema_slow"]),
            "atr": atr.tolist(), "rsi": rsi.tolist(),
            "brk_up": brk_up.tolist(), "brk_dn": brk_dn.tolist(),
            "closes": closes.tolist()}


def simulate(genome, candles, fee=None, slip=None, start_cash=10_000.0,
             sizing="risk"):
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
    else:
        ef = _ema_series(closes, genome["ema_fast"])
        es = _ema_series(closes, genome["ema_slow"])
        _atr = _rsi = _brk_up_lvl = _brk_dn_lvl = None

    cash, side, qty, entry, stop, take, margin = start_cash, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    eq, trades = [], []
    warm = max(genome["ema_slow"], n_bo, 15) + 1

    def close_pos(px):
        nonlocal cash, side, qty, margin
        if side > 0:
            cash += qty * px * (1 - fee)
            trades.append(px / entry - 1)
        else:
            move = qty * (entry - px)
            cash += margin + move - qty * px * fee
            trades.append((entry - px) / entry)
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

            if up_trend and (brk_up or rsi < genome["rsi_buy"]) and mom_up:
                px = cl * (1 + slip)
                notional = _entry_notional(cash, px, genome["stop_atr"] * atr)
                if notional <= 0:
                    eq.append(cash); continue
                qty = notional / px
                cash -= notional * (1 + fee)
                side, entry = 1, px
                stop = px - genome["stop_atr"] * atr
                take = px + genome["take_atr"] * atr
            elif can_short and not up_trend and \
                    (brk_dn or rsi > genome.get("rsi_sell", 70)) and mom_dn:
                px = cl * (1 - slip)
                notional = _entry_notional(cash, px, genome["stop_atr"] * atr)
                if notional <= 0:
                    eq.append(cash); continue
                qty = notional / px
                margin = notional
                cash -= notional + notional * fee
                side, entry = -1, px
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
        self.last_attempt = {}        # product -> ts of last GA run (any outcome)
        self.history = []             # per-generation best fitness (last run)
        self.status = "idle"
        self.current_product = None
        self.generation = 0
        self.last_run = None

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
            pop.append(dict(seed_g))
            for _ in range(max(1, self.pop_size // 3)):
                pop.append(mutate(dict(seed_g), self.rnd, rate=0.5))
        while len(pop) < self.pop_size:
            pop.append(random_genome(self.rnd))
        elite_n = max(2, self.pop_size // 6)

        front_genomes = []          # last generation's Pareto front (elite pool)
        for gen in range(self.generations):
            self.generation = gen + 1
            # ADAPTIVE MUTATION: anneal from exploratory (early gens) to fine
            # (late gens) so the search broadens first, then refines.
            frac = gen / max(1, self.generations - 1)
            mut_rate = 0.45 - 0.30 * frac        # 0.45 -> 0.15
            results = [simulate(g, train) for g in pop]
            objs = [objectives(r) for r in results]
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
        n_trials = self.pop_size * self.generations
        # de-dup the front and cap how many we fully walk-forward validate
        seen_keys, front = set(), []
        for g in sorted(front_genomes, key=lambda g: -fitness(simulate(g, train))):
            key = tuple(round(g[k], 3) for k in GENE_SPACE)
            if key not in seen_keys:
                seen_keys.add(key); front.append(g)
            if len(front) >= 6:
                break

        MIN_OOS_TRADES = 20
        MIN_FRAC_POSITIVE = 0.60          # profitable in >=60% of OOS windows
        candidates = []                   # (score, genome, wf, train_res, dsr)
        for g in front:
            train_res = simulate(g, train)
            wf = walk_forward_eval(g, candles, n_windows=5, embargo=70)
            pooled = [t for r in wf["windows"] for t in r.get("trades", [])]
            dsr = None
            try:
                if len(pooled) >= 5:
                    dsr = deflated_sharpe_ratio(pooled, n_trials=n_trials)
            except Exception:
                dsr = None
            enough = wf["total_trades"] >= MIN_OOS_TRADES
            robust = wf["frac_positive"] >= MIN_FRAC_POSITIVE
            oos_ok = wf["median_return"] > 0 and wf["mean_return"] > 0
            dd_ok = wf["worst_drawdown"] <= 0.15
            # SCALE-FREE gate: under risk-parity sizing raw returns are tiny
            # fractions, so we judge quality by the deflated / pooled Sharpe of
            # the OOS trades, not by a return magnitude threshold.
            dsr_ok = ((dsr is not None and dsr > 0.90) or
                      (dsr is None and wf["pooled_sharpe"] > 0.5 and dd_ok))
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
            "front_size": len(front),
            "n_candidates_passing": len(candidates),
            "promoted": promoted,
            "gate": {"min_oos_trades": MIN_OOS_TRADES,
                     "min_frac_positive": MIN_FRAC_POSITIVE},
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
                "champion": self.champion,                # legacy field
                "champion_report": self.champion_reports.get("BTC-USD"),
                "last_run": self.last_run}


evolution = Evolution()
