"""Purged, embargoed walk-forward validation — the trustworthy promotion gate.

The crypto_ml backtest, run on the same rows the model trained on, produces
in-sample fantasy numbers (we observed ~393,000% return / 50% drawdown). Gating
promotion on that would ship overfit models. This module instead estimates
GENUINE out-of-sample performance the way a quant should:

  * split the prepared feature table into expanding walk-forward folds,
  * drop an EMBARGO gap between each train block and its test block so a label's
    forward-return horizon can't leak across the boundary,
  * train a throwaway model on each fold's train block and score it ONLY on the
    held-out test block it never saw,
  * pool the per-fold out-of-sample results and require CONSISTENCY across folds
    (fraction of folds positive), not just a good average.

The promoted artifact is still the model trained on ALL data; these fold models
exist only to produce the honest gate metric. This file is pure/lab-independent:
fold geometry + aggregation are plain Python so they unit-test without pandas or
crypto_ml. The orchestration that shells out to the lab lives in ml_trainer.
"""
import statistics


def walk_forward_folds(n_rows, n_folds=4, embargo=0):
    """Expanding-window, purged, embargoed walk-forward folds over row indices.

    Returns a list of (train_start, train_end, test_start, test_end) tuples.
    For fold k (1..n_folds): train = rows [0, k*step), then an `embargo`-row gap
    is skipped, then test = [k*step + embargo, (k+1)*step) (the last fold's test
    runs to the end). Folds whose test block would be empty after the embargo are
    dropped, so tiny datasets degrade gracefully to fewer folds.
    """
    n_rows = int(n_rows)
    n_folds = max(1, int(n_folds))
    embargo = max(0, int(embargo))
    folds = []
    if n_rows < 2:
        return folds
    step = n_rows // (n_folds + 1)
    if step < 1:
        return folds
    for k in range(1, n_folds + 1):
        train_end = step * k
        test_start = train_end + embargo
        test_end = step * (k + 1) if k < n_folds else n_rows
        if train_end < 1 or test_start >= test_end or test_start >= n_rows:
            continue
        folds.append((0, train_end, test_start, min(test_end, n_rows)))
    return folds


def aggregate_oos(per_fold):
    """Pool per-fold out-of-sample results into the gate metric dict.

    per_fold: list of {"total_return": float|None, "max_drawdown": float|None}.
    Returns a dict shaped so the existing gate (return + drawdown) reads it
    directly, plus robustness fields, or None if nothing usable came back.

    Headline `total_return` is the MEDIAN fold return (robust to one lucky/unlucky
    fold); `max_drawdown` is the WORST fold drawdown; `oos_frac_folds_positive`
    is how many folds were profitable out of sample.
    """
    rs = [f["total_return"] for f in per_fold
          if f.get("total_return") is not None]
    dds = [abs(f["max_drawdown"]) for f in per_fold
           if f.get("max_drawdown") is not None]
    if not rs:
        return None
    return {
        "total_return": statistics.median(rs),
        "mean_return": statistics.fmean(rs),
        "max_drawdown": (-max(dds) if dds else None),
        "oos_frac_folds_positive": sum(1 for r in rs if r > 0) / len(rs),
        "oos_n_folds": len(rs),
        "per_fold": per_fold,
        "_source": "purged_walkforward",
    }
