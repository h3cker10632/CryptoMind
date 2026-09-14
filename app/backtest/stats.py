"""Statistical-validity toolkit — the anti-overfitting gate.

Because CryptoMind searches heavily (a GA of ~192 evals/coin, re-run on a
rotation, plus an 8-strategy ensemble selected by a bandit), any single
backtest Sharpe is inflated by multiple testing. These functions price that
inflation so promotion decisions use honest numbers:

  * probabilistic_sharpe_ratio (PSR)   — P(true Sharpe > benchmark)
  * deflated_sharpe_ratio (DSR)        — PSR corrected for N trials + fat tails
  * probability_of_backtest_overfitting (PBO) via CSCV
  * wilson_interval / bootstrap_ci     — honest error bars on win-rate/expectancy

References: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014) and
"The Probability of Backtest Overfitting" (2017).
"""
from __future__ import annotations
import math
import random
import statistics
from itertools import combinations


SQRT2 = math.sqrt(2.0)
EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _moments(returns):
    n = len(returns)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    mean = statistics.fmean(returns)
    sd = statistics.pstdev(returns)
    if sd == 0:
        return mean, 0.0, 0.0, 3.0
    skew = sum(((r - mean) / sd) ** 3 for r in returns) / n
    kurt = sum(((r - mean) / sd) ** 4 for r in returns) / n
    return mean, sd, skew, kurt


def sharpe(returns) -> float:
    mean, sd, _, _ = _moments(returns)
    return mean / sd if sd > 0 else 0.0


def probabilistic_sharpe_ratio(returns, benchmark_sr=0.0) -> float:
    """P(true per-observation Sharpe > benchmark) given estimation error."""
    n = len(returns)
    if n < 3:
        return 0.0
    sr = sharpe(returns)
    _, _, skew, kurt = _moments(returns)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr * sr))
    return _norm_cdf((sr - benchmark_sr) * math.sqrt(n - 1) / denom)


def expected_max_sharpe(n_trials: int, trial_sr_std: float) -> float:
    """False-Strategy-Theorem expected max Sharpe of N skill-less trials."""
    n = max(2, n_trials)
    z1 = _norm_ppf(1 - 1.0 / n)
    z2 = _norm_ppf(1 - 1.0 / (n * math.e))
    return trial_sr_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(returns, n_trials: int, trial_sr_std=None) -> float:
    """DSR: PSR against the expected-max-Sharpe-under-the-null benchmark.

    Values approach 1.0 for genuine edge; the conventional significance bar
    is 0.95. If the dispersion of the trials' Sharpes isn't supplied we fall
    back to a conservative default.
    """
    if trial_sr_std is None or trial_sr_std <= 0:
        trial_sr_std = 0.5    # conservative: trials' Sharpes vary a lot
    sr0 = expected_max_sharpe(n_trials, trial_sr_std)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr0)


def probability_of_backtest_overfitting(matrix, n_splits=10):
    """PBO via Combinatorially-Symmetric Cross-Validation (CSCV).

    `matrix` is a list of per-strategy return series (each same length),
    i.e. matrix[strategy][t]. Returns the fraction of combinatorial IS/OOS
    splits where the in-sample-best strategy ranks below the OOS median —
    the probability the chosen configuration is overfit. Lower is better;
    promote only when PBO < 0.30.
    """
    n_strats = len(matrix)
    if n_strats < 2:
        return None
    T = min(len(s) for s in matrix)
    if T < n_splits * 2:
        return None
    block = T // n_splits
    blocks = [list(range(i * block, (i + 1) * block)) for i in range(n_splits)]
    half = n_splits // 2
    if half == 0:
        return None
    logits = []
    for is_idx in combinations(range(n_splits), half):
        oos_idx = [b for b in range(n_splits) if b not in is_idx]
        is_rows = [i for b in is_idx for i in blocks[b]]
        oos_rows = [i for b in oos_idx for i in blocks[b]]

        def perf(rows):
            out = []
            for s in matrix:
                seg = [s[i] for i in rows]
                out.append(sharpe(seg))
            return out

        is_perf = perf(is_rows)
        oos_perf = perf(oos_rows)
        best = max(range(n_strats), key=lambda k: is_perf[k])
        # rank of the IS-best in OOS (0..1, higher = better)
        rank = sum(1 for v in oos_perf if v <= oos_perf[best]) / n_strats
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
    if not logits:
        return None
    return sum(1 for w in logits if w <= 0) / len(logits)


def wilson_interval(wins: int, n: int, z=1.96):
    """95% Wilson score interval for a win rate — honest small-sample bounds."""
    if n == 0:
        return (None, None, None)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(p, 4), round(max(0.0, centre - margin), 4),
            round(min(1.0, centre + margin), 4))


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=13):
    """Bootstrap CI for the mean (e.g. per-trade expectancy)."""
    n = len(values)
    if n < 2:
        return (None, None, None)
    rnd = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(rnd.choice(values) for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot) - 1]
    return (round(statistics.fmean(values), 6), round(lo, 6), round(hi, 6))
