"""Statistics engine for the Invo signal study — pure numpy (no scipy dep).

Everything operates on aligned (signal, forward_return) pairs. The headline
numbers:
  * IC        — Spearman rank correlation of signal vs forward return (the
                standard cross-sectional "does it predict" metric).
  * pearson   — linear correlation (secondary).
  * mutual_info - captures non-linear dependence a rank corr can miss.
  * p_value   — permutation test: probability of an |IC| this large by chance.
  * partial_ic- IC AFTER regressing the forward return on the existing baseline
                features (funding/OI/L-S). If this collapses toward 0, the Invo
                signal adds nothing you don't already have.
  * ablation  — walk-forward directional accuracy + log-loss, baseline features
                vs baseline+Invo. The bottom-line "does adding it help" test.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    return pearson(_rank(x), _rank(y))


def mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 6) -> float:
    if len(x) < 8:
        return 0.0
    c, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = c / c.sum()
    px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz] + 1e-12))
    return float(max(0.0, mi))


def permutation_pvalue(x: np.ndarray, y: np.ndarray, n: int = 2000,
                       seed: int = 0) -> float:
    obs = abs(spearman(x, y))
    if obs == 0.0 or len(x) < 4:
        return 1.0
    rng = np.random.default_rng(seed)
    yc = y.copy(); hits = 0
    for _ in range(n):
        rng.shuffle(yc)
        if abs(spearman(x, yc)) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def _ols_residual(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residual of y after least-squares regression on Z (with intercept)."""
    A = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_ic(sig: np.ndarray, ret: np.ndarray, baseline: np.ndarray) -> float:
    """Spearman IC between signal and forward return, after removing whatever the
    baseline features already explain in BOTH. Near-zero => not orthogonal."""
    if baseline is None or len(sig) < 5:
        return spearman(sig, ret)
    rr = _ols_residual(ret, baseline)
    rs = _ols_residual(sig, baseline)
    return spearman(rs, rr)


# ---------------- walk-forward logistic-regression ablation ----------------

def _logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                  iters: int = 300, lr: float = 0.1) -> np.ndarray:
    Xs = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        grad = Xs.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y)
        w -= lr * grad
    return w


def _logistic_pred(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xs = np.column_stack([np.ones(len(X)), X])
    return 1.0 / (1.0 + np.exp(-Xs @ w))


def _standardize(train: np.ndarray, test: np.ndarray):
    mu = train.mean(0); sd = train.std(0); sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def walk_forward_ablation(feat: np.ndarray, invo: np.ndarray, ret: np.ndarray,
                          folds: int = 5) -> Dict[str, float]:
    """Compare P(up) models: baseline features vs baseline+Invo, via expanding
    walk-forward. Returns dir-accuracy and log-loss for each + deltas.
    `feat` may be None (then baseline is an intercept-only / invo-only compare)."""
    y = (ret > 0).astype(float)
    n = len(y)
    if n < 20:
        return {"n": n, "insufficient": True}

    def _run(X):
        accs, lls = [], []
        step = n // (folds + 1)
        for k in range(1, folds + 1):
            tr = slice(0, step * k); te = slice(step * k, step * (k + 1))
            if te.stop > n or (te.stop - te.start) < 3:
                break
            ytr, yte = y[tr], y[te]
            if len(np.unique(ytr)) < 2:
                continue
            if X is None:
                p = np.full(len(yte), ytr.mean())
            else:
                Xtr, Xte = _standardize(X[tr], X[te])
                w = _logistic_fit(Xtr, ytr)
                p = _logistic_pred(w, Xte)
            p = np.clip(p, 1e-4, 1 - 1e-4)
            accs.append(float(np.mean((p > 0.5) == (yte > 0.5))))
            lls.append(float(-np.mean(yte * np.log(p) + (1 - yte) * np.log(1 - p))))
        return (float(np.mean(accs)) if accs else float("nan"),
                float(np.mean(lls)) if lls else float("nan"))

    base_X = feat
    both_X = invo.reshape(-1, 1) if feat is None else np.column_stack([feat, invo])
    b_acc, b_ll = _run(base_X)
    w_acc, w_ll = _run(both_X)
    return {"n": n,
            "base_acc": b_acc, "base_logloss": b_ll,
            "with_invo_acc": w_acc, "with_invo_logloss": w_ll,
            "d_acc": w_acc - b_acc, "d_logloss": w_ll - b_ll}


def evaluate_asset(rows: List[Tuple[float, float, float]],
                   baseline_rows: List[list] | None = None) -> Dict:
    """rows: [(lean, fwd_ret, ts)]. baseline_rows aligned same order (feature cols)."""
    sig = np.array([r[0] for r in rows], dtype=float)
    ret = np.array([r[1] for r in rows], dtype=float)
    base = np.array(baseline_rows, dtype=float) if baseline_rows else None
    out = {
        "n": len(rows),
        "ic": spearman(sig, ret),
        "pearson": pearson(sig, ret),
        "mutual_info": mutual_info(sig, ret),
        "p_value": permutation_pvalue(sig, ret),
    }
    if base is not None and len(base) == len(rows):
        out["partial_ic"] = partial_ic(sig, ret, base)
        out["ablation"] = walk_forward_ablation(base, sig, ret)
    else:
        out["partial_ic"] = None
        out["ablation"] = walk_forward_ablation(None, sig, ret)
    return out
