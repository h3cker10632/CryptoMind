"""Models for the pooled panel, fitted WALK-FORWARD: a prediction for day t
comes from a model trained only on samples whose labels had fully happened
before the model's refit day (label window t..t+h ends <= refit day - 1).

  ridge     weighted ridge regression (numpy, closed form) — on noisy
            financial data a heavily regularized linear model is hard to beat
  gbt       gradient-boosted trees (scikit-learn HistGradientBoosting), shallow
            and strongly regularized
  ensemble  average of the two models' per-day cross-sectional RANKS
  logistic  (meta-labeling) weighted L2 logistic regression (numpy IRLS)
"""
from __future__ import annotations
import numpy as np


class Ridge:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, X, y, w):
        sw = w.sum()
        mu = (w[:, None] * X).sum(0) / sw
        sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0) / sw)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (X - mu) / sd
        ym = (w * y).sum() / sw
        A = Z.T @ (w[:, None] * Z) + self.alpha * np.eye(Z.shape[1]) * sw / len(w)
        b = Z.T @ (w * (y - ym))
        self.coef = np.linalg.solve(A, b)
        self.mu, self.sd, self.ym = mu, sd, ym
        return self

    def predict(self, X):
        return self.ym + ((X - self.mu) / self.sd) @ self.coef


class GBT:
    def __init__(self, max_iter=150, seed=0, classify=False):
        self.max_iter, self.seed, self.classify = max_iter, seed, classify

    def fit(self, X, y, w):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        kw = dict(max_iter=self.max_iter, learning_rate=0.05, max_leaf_nodes=15,
                  min_samples_leaf=max(50, len(y) // 200), l2_regularization=1.0,
                  random_state=self.seed)
        self.m = (HistGradientBoostingClassifier if self.classify
                  else HistGradientBoostingRegressor)(**kw)
        self.m.fit(X, y, sample_weight=w / w.mean())
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1] if self.classify else self.m.predict(X)


class Logistic:
    def __init__(self, alpha=1.0, iters=25):
        self.alpha, self.iters = alpha, iters

    def fit(self, X, y, w):
        sw = w.sum()
        mu = (w[:, None] * X).sum(0) / sw
        sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0) / sw)
        self.mu, self.sd = mu, np.where(sd > 0, sd, 1.0)
        Z = np.column_stack([np.ones(len(y)), (X - self.mu) / self.sd])
        beta = np.zeros(Z.shape[1])
        wn = w / w.mean()
        reg = self.alpha * np.eye(Z.shape[1])
        reg[0, 0] = 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
            g = Z.T @ (wn * (p - y)) + reg @ beta
            Hm = Z.T @ ((wn * p * (1 - p))[:, None] * Z) + reg
            step = np.linalg.solve(Hm, g)
            beta -= step
            if np.abs(step).max() < 1e-8:
                break
        self.beta = beta
        return self

    def predict(self, X):
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-np.clip(Z @ self.beta, -30, 30)))


def has_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def make(kind, classify=False, seed=0):
    if kind == "ridge":
        return Logistic() if classify else Ridge()
    if kind == "gbt":
        if not has_sklearn():
            return Logistic() if classify else Ridge()
        return GBT(seed=seed, classify=classify)
    raise ValueError(kind)


def _rank_rows(p):
    from ..engine.features import cross_rank
    return cross_rank(p)


def walk_forward(X, y, w, model="ensemble", h=7, refit_days=30, min_train_days=365,
                 classify=False, start=0, max_rows=200_000, seed=0):
    """(T, N) out-of-sample predictions. Refit every `refit_days` from the
    first day with `min_train_days` of labelled history; the model fitted at
    day r predicts days r .. r + refit_days - 1 and was trained on samples
    t <= r - h - 1 (labels ended before r). NaN where no model exists yet or
    features are incomplete."""
    T, N, K = X.shape
    pred = np.full((T, N), np.nan)
    labelled = w > 0
    days_with = np.flatnonzero(labelled.any(axis=1))
    if not len(days_with):
        return pred
    first = max(start, int(days_with[0]) + min_train_days + h + 1)
    kinds = ["ridge", "gbt"] if model == "ensemble" else [model]
    feat_ok = np.isfinite(X).all(axis=2)
    for r in range(first, T, refit_days):
        tr = labelled[: r - h]
        ti, tj = np.nonzero(tr)
        if len(ti) < 200:
            continue
        if len(ti) > max_rows:                       # keep the most recent samples
            keep = np.argsort(ti)[-max_rows:]
            ti, tj = ti[keep], tj[keep]
        Xt, yt, wt = X[ti, tj], y[ti, tj], w[ti, tj]
        hi = min(T, r + refit_days)
        pi, pj = np.nonzero(feat_ok[r:hi])
        if not len(pi):
            continue
        Xp = X[r + pi, pj]
        outs = []
        for kind in kinds:
            mdl = make(kind, classify=classify, seed=seed).fit(Xt, yt, wt)
            block = np.full((hi - r, N), np.nan)
            block[pi, pj] = mdl.predict(Xp)
            outs.append(block)
        if len(outs) == 1 or classify:
            pred[r:hi] = sum(outs) / len(outs)
        else:                                        # average of per-day ranks
            pred[r:hi] = sum(_rank_rows(o) for o in outs) / len(outs)
    return pred
