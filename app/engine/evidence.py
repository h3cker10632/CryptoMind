"""Evidence statistics for the research loop — how much a stretch of returns
actually tells us, so decisions wait for evidence and no longer.

  newey_west_var       long-run variance of a series (overlap / autocorrelation)
  paired_sequential    challenger vs champion on the SAME days: is the
                       difference real? An always-valid (mixture SPRT) test,
                       so it can be checked every day without inflating the
                       false-positive rate, and a clear loser is dropped early.
  promotion_decision   the champion / challenger promotion rule, one pure
                       function used by the live loop (challengers.run) and by
                       the historical replay of that loop (promotion_backtest).

Why paired: over 30 daily returns an annualized Sharpe estimate has a
standard error of ~3.5, so "forward Sharpe above the champion's" was close to
a coin flip. Most candidates are variants of the same BTC/ETH trend and move
together; the day-by-day DIFFERENCE between two of them is far less noisy
than either series, so comparing them on the same days needs much less data.
The challenger is first scaled to the champion's volatility over the window,
so taking more risk can't pass for skill.
"""
from __future__ import annotations
import math

import numpy as np


def newey_west_var(x, lags=None):
    """Long-run variance of the MEAN's numerator (per observation): the
    variance plus Bartlett-weighted autocovariances. `lags` defaults to
    floor(4 (n/100)^(2/9))."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    lags = max(0, min(int(lags), n - 1))
    d = x - x.mean()
    v = float(d @ d) / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * float(d[k:] @ d[:-k]) / n
    return max(v, 0.0)


def nw_tstat(x, lags=None):
    """t-statistic of the mean of `x` with a Newey-West standard error."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    v = newey_west_var(x, lags)
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else float("nan")


def msprt_lr(z, n, c):
    """Normal-mixture SPRT likelihood ratio for a mean, in standardized units:
    z = mean * sqrt(n) / sd, mixture prior on the per-observation standardized
    mean N(0, c^2). Rejecting 'no difference' when LR >= 1/alpha keeps the
    error rate <= alpha however often it is checked (Robbins; Johari et al.)."""
    a = n * c * c
    return (1 + a) ** -0.5 * math.exp(min(700.0, a / (1 + a) * z * z / 2))


def paired_sequential(challenger, champion, alpha=0.05, tau=0.1, lags=None):
    """Compare two return series over the same, most recent days.

    The challenger is scaled to the champion's volatility, then the daily
    difference d is tested: 'better' / 'worse' once the always-valid
    likelihood ratio passes 1/alpha (one-sided each way), else 'undecided'.
    `tau` is the mixture prior's scale for the standardized daily mean
    difference (0.1 ~ an annualized Sharpe gap of ~1.9 for uncorrelated
    series; much less for correlated ones, where the difference is quiet)."""
    a = np.asarray(challenger, dtype=float)
    b = np.asarray(champion, dtype=float)
    k = min(len(a), len(b))
    out = {"n": int(k), "decision": "undecided", "mean_diff_bps": None, "z": None,
           "lr": None, "vol_scale": None}
    if k < 5:
        return out
    a, b = a[-k:], b[-k:]
    sa, sb = a.std(), b.std()
    scale = (sb / sa) if sa > 0 and sb > 0 else 1.0
    d = a * scale - b
    v = newey_west_var(d, lags)
    if not (v > 0):
        return out
    z = float(d.mean() * math.sqrt(k / v))
    lr = msprt_lr(z, k, tau)
    out.update(mean_diff_bps=round(float(d.mean()) * 1e4, 3), z=round(z, 3),
               lr=round(min(lr, 1e12), 3), vol_scale=round(float(scale), 3))
    if lr >= 1 / alpha:
        out["decision"] = "better" if z > 0 else "worse"
    return out


def promotion_decision(backtest, forward, champion, n_trials, trial_sr_std=None,
                       min_dsr=0.95, min_forward_days=30, max_forward_days=180,
                       alpha=0.05, tau=0.1, retired=()):
    """The promotion rule. `backtest` / `forward`: {name: daily returns} on the
    same day grid (both ending on the same day). A challenger is ELIGIBLE when

      * its backtest beats the champion's on Sharpe in BOTH halves,
      * its deflated Sharpe (counting `n_trials`) >= `min_dsr`,
      * it has >= `min_forward_days` of forward tracking and the paired
        sequential test vs the champion is 'better' — or, without a decision,
        it has tracked `max_forward_days` with a positive paired difference
        (weak evidence: the backtest gates carry that call),
      * it is not retired (a 'worse' verdict retires it for good).

    Returns ({name: verdict dict}, winner or None)."""
    from .backtest import report, stats
    c_bt = np.asarray(backtest[champion], dtype=float)
    h = len(c_bt) // 2
    verdicts = {}
    for name, bt in backtest.items():
        bt = np.asarray(bt, dtype=float)
        rep = report(bt, n_trials=n_trials, trial_sr_std=trial_sr_std)
        beats_bt = name != champion and all(
            stats(bt[sl])["sharpe"] > stats(c_bt[sl])["sharpe"]
            for sl in (slice(0, h), slice(h, None)))
        fwd = np.asarray(forward.get(name, []), dtype=float)
        cf = np.asarray(forward.get(champion, []), dtype=float)
        k = min(len(fwd), len(cf))
        test = (paired_sequential(fwd, cf, alpha=alpha, tau=tau) if name != champion
                else {"n": int(k), "decision": "champion"})
        is_retired = name in retired or test["decision"] == "worse"
        fwd_ok = k >= min_forward_days and (
            test["decision"] == "better"
            or (test["decision"] == "undecided" and k >= max_forward_days
                and (test.get("mean_diff_bps") or 0) > 0))
        eligible = (name != champion and not is_retired and beats_bt
                    and rep["deflated_sharpe"] >= min_dsr and fwd_ok)
        f_sh = stats(fwd[-k:])["sharpe"] if k > 1 else None
        c_sh = stats(cf[-k:])["sharpe"] if k > 1 else None
        verdicts[name] = {
            "backtest": {k2: rep[k2] for k2 in ("full", "first_half", "second_half",
                                                "cagr_pct", "deflated_sharpe")},
            "beats_champion_backtest_both_halves": bool(beats_bt),
            "forward_days": int(len(fwd)), "forward": stats(fwd),
            "forward_sharpe_vs_champion": [f_sh, c_sh] if k > 1 else None,
            "forward_test": test, "retired": bool(is_retired),
            "eligible_for_promotion": bool(eligible)}
    winners = [n for n, v in verdicts.items() if v["eligible_for_promotion"]]
    winner = max(winners, key=lambda n: (verdicts[n]["forward_test"].get("z") or 0.0)) \
        if winners else None
    return verdicts, winner
