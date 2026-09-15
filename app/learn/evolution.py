"""Genetic strategy evolution — evolves trading-rule parameters against real
historical backtests and promotes the champion genome to live trading.

Genome (7 genes):
  ema_fast, ema_slow        trend filter periods
  rsi_buy                   oversold entry assist threshold
  breakout_n                breakout lookback
  stop_atr, take_atr        exit multiples
  mom_w                     momentum-confirmation weight

Fitness = calmar-like score on an in-sample slice, validated out-of-sample
(champion must be OOS-positive-ish to be promoted). Runs in a thread so the
event loop never blocks.
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


# ---------------- genome simulation on candle history ----------------

def _ema_series(xs, n):
    k = 2 / (n + 1)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def simulate(genome, candles, fee=None, slip=None, start_cash=10_000.0):
    """Event-driven sim; LONG and (if short_w > 0.5) SHORT trades.
    side: 0 flat, +1 long, -1 short (margin-style shorts)."""
    from ..tunables import tv
    fee = tv("fee_rate") if fee is None else fee
    slip = (tv("slippage_bps") / 1e4) if slip is None else slip
    closes = [c[4] for c in candles]
    ef = _ema_series(closes, genome["ema_fast"])
    es = _ema_series(closes, genome["ema_slow"])
    n_bo = genome["breakout_n"]
    can_short = genome.get("short_w", 0) > 0.5

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

        # ATR(14)
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
            # RSI(14)
            gains = [max(closes[j] - closes[j - 1], 0) for j in range(i - 13, i + 1)]
            losses = [max(closes[j - 1] - closes[j], 0) for j in range(i - 13, i + 1)]
            ag, al = sum(gains) / 14, sum(losses) / 14
            rsi = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)

            up_trend = ef[i] > es[i]
            brk_up = cl >= max(c[2] for c in candles[i - n_bo:i])
            brk_dn = cl <= min(c[1] for c in candles[i - n_bo:i])
            mom_up = closes[i] > closes[i - 12] if genome["mom_w"] > 0.5 else True
            mom_dn = closes[i] < closes[i - 12] if genome["mom_w"] > 0.5 else True

            if up_trend and (brk_up or rsi < genome["rsi_buy"]) and mom_up:
                px = cl * (1 + slip)
                notional = cash * 0.95
                qty = notional / px
                cash -= notional * (1 + fee)
                side, entry = 1, px
                stop = px - genome["stop_atr"] * atr
                take = px + genome["take_atr"] * atr
            elif can_short and not up_trend and \
                    (brk_dn or rsi > genome.get("rsi_sell", 70)) and mom_dn:
                px = cl * (1 - slip)
                notional = cash * 0.95
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
        self.champions = {}           # product -> promoted genome
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

    def evolve(self, candles, product="BTC-USD"):
        """Full GA run with train/validation split. Called from a thread."""
        self.status = "running"
        self.current_product = product
        self.history = []
        split = int(len(candles) * 0.65)
        train, valid = candles[:split], candles[split - 70:]

        pop = [random_genome(self.rnd) for _ in range(self.pop_size)]
        elite_n = max(2, self.pop_size // 6)

        for gen in range(self.generations):
            self.generation = gen + 1
            scored = []
            for g in pop:
                res = simulate(g, train)
                scored.append((fitness(res), g, res))
            scored.sort(key=lambda t: -t[0])
            self.history.append({
                "gen": gen + 1,
                "best_fitness": round(scored[0][0], 4),
                "best_return": round(scored[0][2]["total_return"], 4),
                "mean_fitness": round(sum(s[0] for s in scored) / len(scored), 4),
            })
            # next generation: elitism + tournament + crossover + mutation
            elites = [g for _, g, _ in scored[:elite_n]]
            nxt = list(elites)
            while len(nxt) < self.pop_size:
                a = min(self.rnd.sample(scored, 3), key=lambda t: -t[0])[1]
                b = min(self.rnd.sample(scored, 3), key=lambda t: -t[0])[1]
                child = mutate(crossover(a, b, self.rnd), self.rnd)
                nxt.append(child)
            pop = nxt

        # validate the best genome out-of-sample, then apply a MULTIPLE-TESTING
        # aware promotion gate (the GA evaluated pop_size*generations genomes,
        # so a raw OOS-positive is not enough — price the selection bias).
        best_fit, best_g, best_train = scored[0]
        val = simulate(best_g, valid)
        n_trials = self.pop_size * self.generations
        dsr = None
        val_trades = val.get("trades", [])
        try:
            from ..backtest.stats import deflated_sharpe_ratio
            if len(val_trades) >= 5:
                dsr = deflated_sharpe_ratio(val_trades, n_trials=n_trials)
        except Exception:
            dsr = None
        # gates: OOS profitable AND enough OOS trades AND deflated-Sharpe passes
        # (or DSR unavailable but OOS is clearly positive on decent samples).
        # A larger OOS sample is required now — 3-trade "champions" (SOL) were
        # pure luck; the deflated-Sharpe path needs enough trades to be usable
        # and the no-DSR fallback demands both a healthy sample AND a bounded
        # drawdown so a 21%-DD genome (UNI) can't sneak through on raw return.
        MIN_OOS_TRADES = 20
        enough = val["n_trades"] >= MIN_OOS_TRADES
        oos_ok = fitness(val) > 0 and val["total_return"] > 0
        dd_ok = val.get("max_drawdown", 1.0) <= 0.15
        dsr_ok = ((dsr is not None and dsr > 0.90 and val["n_trades"] >= 20) or
                  (dsr is None and val["total_return"] > 0.03 and dd_ok))
        promoted = bool(enough and oos_ok and dsr_ok)
        report = {
            "product": product,
            "genome": best_g,
            "train": {k: round(v, 4) for k, v in best_train.items() if k != "trades"},
            "validation": {k: round(v, 4) for k, v in val.items() if k != "trades"},
            "train_fitness": round(best_fit, 4),
            "validation_fitness": round(fitness(val), 4),
            "deflated_sharpe": round(dsr, 4) if dsr is not None else None,
            "n_trials": n_trials,
            "promoted": promoted,
            "gate": {"enough_oos_trades": enough, "oos_profitable": oos_ok,
                     "deflated_sharpe_ok": dsr_ok},
            "ts": time.time(),
        }
        if promoted:
            self.champions[product] = best_g
            self.champion_reports[product] = report
        elif product in self.champions:
            # A rerun failed to re-validate the incumbent's regime, yet the OLD
            # champion is still voting live (this is exactly the stale-SOL /
            # UNI case). Evict it: a champion that can no longer earn promotion
            # on fresh out-of-sample data has no business steering allocation.
            evicted = self.champions.pop(product, None)
            self.champion_reports.pop(product, None)
            report["evicted_stale_champion"] = bool(evicted)
            from .. import db
            db.log_event("learn",
                         f"GA champion for {product} EVICTED — latest rerun "
                         f"failed promotion (OOS return {val['total_return']:+.1%}, "
                         f"{val['n_trades']} trades, DD "
                         f"{val.get('max_drawdown', 0):.1%}); no live vote until "
                         f"a genome re-validates.")
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
                "champion_reports": self.champion_reports,
                "champion": self.champion,                # legacy field
                "champion_report": self.champion_reports.get("BTC-USD"),
                "last_run": self.last_run}


evolution = Evolution()
