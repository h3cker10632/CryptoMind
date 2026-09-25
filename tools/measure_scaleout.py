"""Measure the profit-ladder (runner) scale-out prototype on NET-of-cost returns.

The premise under test is "bank half of a meme markup winner, let the rest ride
risk-free." Scale-out only pays when the return distribution has a fat right tail
(true runners); on choppy/mean-reverting tape it should just add cost. So we
Monte-Carlo three regimes and A/B run_composite with vs without the ladder,
reporting mean/median net return, drawdown, and how often scale-out wins.

Run:  python -m tools.measure_scaleout
"""
import random
import statistics as stats
from app.backtest.composite import run_composite

BARS = 400
PATHS = 400


def _bar(px, r):
    hi = px * (1 + abs(r.gauss(0, 0.004)))
    lo = px * (1 - abs(r.gauss(0, 0.004)))
    op = lo + (hi - lo) * r.random()
    return [0, lo, hi, op, px, 1000 + r.random() * 500]


def gen_choppy(seed):
    r = random.Random(seed); px = 100.0; cs = []
    for _ in range(BARS):
        # mean-reverting around 100, no persistent drift, no fat tail
        pull = (100.0 - px) * 0.02
        px *= (1 + pull / px + r.gauss(0, 0.012))
        cs.append(_bar(px, r))
    return cs


def gen_trend(seed):
    r = random.Random(seed); px = 100.0; cs = []
    for _ in range(BARS):
        px *= (1 + 0.0006 + r.gauss(0, 0.010))
        cs.append(_bar(px, r))
    return cs


def gen_meme(seed):
    """Fat-tailed 'meme' tape: modest base drift + occasional violent pumps and
    crashes → genuine runners (and blow-offs) for the ladder to act on."""
    r = random.Random(seed); px = 100.0; cs = []
    for _ in range(BARS):
        base = 0.0004 + r.gauss(0, 0.011)
        jump = 0.0
        u = r.random()
        if u < 0.02:                       # 2% chance of a pump leg
            jump = r.uniform(0.05, 0.22)
        elif u > 0.985:                    # 1.5% chance of a crash
            jump = -r.uniform(0.05, 0.18)
        px *= (1 + base + jump)
        px = max(px, 1.0)
        cs.append(_bar(px, r))
    return cs


def summarize(name, gen, scale_cfg):
    base_rets, so_rets, base_dd, so_dd, wins = [], [], [], [], 0
    for s in range(PATHS):
        cs = gen(s)
        b = run_composite(cs, scale_out=None)
        o = run_composite(cs, scale_out=scale_cfg)
        base_rets.append(b["total_return"]); so_rets.append(o["total_return"])
        base_dd.append(b["max_drawdown"]);   so_dd.append(o["max_drawdown"])
        if o["total_return"] > b["total_return"]:
            wins += 1
    def m(x): return round(stats.mean(x), 4)
    def md(x): return round(stats.median(x), 4)
    print(f"\n=== {name}  (n={PATHS} paths) ===")
    print(f"  net return   mean: base {m(base_rets):+.4f}  ->  scale {m(so_rets):+.4f}  "
          f"(Δ {m(so_rets)-m(base_rets):+.4f})")
    print(f"  net return median: base {md(base_rets):+.4f}  ->  scale {md(so_rets):+.4f}")
    print(f"  max drawdown mean: base {m(base_dd):+.4f}  ->  scale {m(so_dd):+.4f}  "
          f"(Δ {m(so_dd)-m(base_dd):+.4f}, less negative = smoother)")
    print(f"  scale-out beat baseline on {wins}/{PATHS} paths ({100*wins/PATHS:.0f}%)")


if __name__ == "__main__":
    cfg = {"arm_atr": 1.0, "frac": 0.5, "breakeven": True}
    print(f"scale-out config: {cfg}")
    summarize("CHOPPY / mean-reverting", gen_choppy, cfg)
    summarize("TREND", gen_trend, cfg)
    summarize("MEME (fat-tailed pumps/crashes)", gen_meme, cfg)
    # sensitivity: a later, smaller bank on the meme tape
    print("\n--- sensitivity: arm later (1.5 ATR), bank a third ---")
    summarize("MEME  arm=1.5 frac=0.33", gen_meme,
              {"arm_atr": 1.5, "frac": 0.33, "breakeven": True})
