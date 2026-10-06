"""Trade filter — a learned "is THIS signal worth taking?" gate (meta-labeling).

The strategies decide direction; this model only judges whether a given
actionable signal is likely to pay for itself. It is trained on history the
bot already downloads for the replay:

  samples  every bar where the (history-computable) signal engine produced an
           actionable signal, for every coin;
  features a small, clean, DIRECTION-ALIGNED set (trend, momentum, breakout
           position, cross-coin strength, timeframe agreement, volatility,
           BTC's trend as a market factor, market trendiness, signal
           confidence) — see `features()`, used identically live and in
           history;
  label    what the trade would actually have done: simulate the bot's own
           exits (ATR stop, take-profit, trailing stop) bar by bar up to
           `MAX_HOLD_BARS`, subtract round-trip costs, and mark it a win if the
           NET return is > 0 ("target-or-stop first", a.k.a. triple barrier).

Model: scikit-learn HistGradientBoostingClassifier (shallow, regularised) when
available, otherwise a numpy logistic regression on quantile-binned features.

Honest evaluation: `walk_forward_probs()` produces OUT-OF-SAMPLE probabilities
for the whole history by expanding-window refits in which a sample can only be
trained on once its trade had fully RESOLVED before the test window starts
(no leakage). The replay compares trading with vs without the filter on those
probabilities; in `auto` mode the live gate is on only while that comparison
favours the filter in BOTH halves.

Decision rule: take a signal only if P(win) >= the training set's base win
rate x `filter_strictness` — i.e. skip trades the model rates below average.
"""
from __future__ import annotations
import math
import threading
import time

FEATURE_NAMES = (
    "direction", "confidence", "trend_24_96", "mom_72", "mom_48", "mom_12",
    "breakout_pos", "xs_mom", "mtf_align", "atr_pct", "volatility",
    "rsi", "vol_ratio", "btc_trend", "market_trendiness",
)
MAX_HOLD_BARS = 7 * 24            # a filter label resolves within a week
MIN_TRAIN = 300                   # samples needed before any model is fit
FOLD_DAYS = 30                    # walk-forward refit cadence


def _c(x, lo=-5.0, hi=5.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return max(lo, min(hi, x))


def features(f, direction, confidence, btc_f=None, market_trend=None):
    """Feature vector for one signal. `f` is the coin's feature dict at the
    bar (market.features() live, _HistMarket.features() in history)."""
    d = 1.0 if direction > 0 else -1.0
    px = f.get("price") or 1.0
    e24, e96 = f.get("ema24"), f.get("ema96")
    trend = (e24 / e96 - 1) * 20 if e24 and e96 else 0.0
    hi, lo = f.get("hi48"), f.get("lo48")
    brk = ((px - lo) / (hi - lo) * 2 - 1) if hi and lo and hi > lo else 0.0
    bt = 0.0
    if btc_f and btc_f.get("ema24") and btc_f.get("ema96"):
        bt = (btc_f["ema24"] / btc_f["ema96"] - 1) * 20
    return [
        d,
        _c(confidence, 0, 1),
        _c(trend * d),
        _c((f.get("mom_72") or 0.0) * 10 * d),
        _c((f.get("mom_4h") or 0.0) * 15 * d),
        _c((f.get("mom_1h") or 0.0) * 30 * d),
        _c(brk * d, -3, 3),
        _c((f.get("xs_mom") or 0.0) * d),
        _c((f.get("mtf_align") or 0.0) * d, -1, 1),
        _c((f.get("atr_swing") or f.get("atr") or 0.0) / px * 100, 0, 20),
        _c((f.get("volatility") or 0.0) * 100, 0, 20),
        _c(((f.get("rsi") or 50.0) - 50) / 50 * d, -1, 1),
        _c(f.get("vol_ratio") or 1.0, 0, 10),
        _c(bt * d),
        _c(market_trend if market_trend is not None else 0.0, 0, 1),
    ]


# ---------------------------------------------------------------- labels
def triple_barrier_net(closes, i, direction, atr, stop_k, tp_k, trail_k,
                       entry_cost, taker_cost, tp_cost, max_hold=MAX_HOLD_BARS):
    """Net return of a trade entered at closes[i] with the bot's own exits,
    evaluated on bar closes (same convention as the replay). Returns
    (net_return, bars_held) or (None, None) if it hasn't resolved yet."""
    x0 = closes[i]
    if not x0 or not atr or atr <= 0:
        return None, None
    s = 1 if direction > 0 else -1
    stop = x0 - s * stop_k * atr
    tp = x0 + s * tp_k * atr
    water = x0
    n = len(closes)
    for j in range(i + 1, min(n, i + 1 + max_hold)):
        x = closes[j]
        water = max(water, x) if s > 0 else min(water, x)
        trail = water - s * trail_k * atr
        stop = max(stop, trail) if s > 0 else min(stop, trail)
        if (x - stop) * s <= 0:
            return (x / x0 - 1) * s - entry_cost - taker_cost, j - i
        if (x - tp) * s >= 0:
            return (tp / x0 - 1) * s - entry_cost - tp_cost, j - i
    if i + max_hold < n:                      # time exit at the horizon
        x = closes[i + max_hold]
        return (x / x0 - 1) * s - entry_cost - taker_cost, max_hold
    return None, None                         # still open: not a label yet


# ---------------------------------------------------------------- models
class _BinnedLogit:
    """Numpy fallback: logistic regression on quantile-binned (one-hot)
    features — captures simple non-linear effects without scikit-learn."""

    def __init__(self, bins=6, l2=1.0, iters=300, lr=0.5):
        self.bins, self.l2, self.iters, self.lr = bins, l2, iters, lr

    def _encode(self, X):
        import numpy as np
        cols = []
        for k, edges in enumerate(self.edges):
            idx = np.searchsorted(edges, X[:, k], side="right")
            oh = np.zeros((X.shape[0], len(edges) + 1))
            oh[np.arange(X.shape[0]), idx] = 1.0
            cols.append(oh)
        return np.hstack(cols + [np.ones((X.shape[0], 1))])

    def fit(self, X, y):
        import numpy as np
        X = np.asarray(X, float); y = np.asarray(y, float)
        qs = np.linspace(0, 1, self.bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, k], qs)) for k in range(X.shape[1])]
        Z = self._encode(X)
        w = np.zeros(Z.shape[1])
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (p - y) / len(y) + self.l2 * w / len(y)
            w -= self.lr * g
        self.w = w
        return self

    def predict_proba(self, X):
        import numpy as np
        p = 1 / (1 + np.exp(-self._encode(np.asarray(X, float)) @ self.w))
        return np.column_stack([1 - p, p])


def make_model():
    """(model, backend_name)."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return (HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=False, random_state=7),
            "sklearn-hgb")
    except Exception:
        return _BinnedLogit(), "numpy-binned-logit"


def _fit(X, y):
    import numpy as np
    y = np.asarray(y, int)
    if len(y) < MIN_TRAIN or y.min() == y.max():
        return None, None, None
    m, backend = make_model()
    m.fit(np.asarray(X, float), y)
    return m, backend, float(y.mean())


# ---------------------------------------------------------------- dataset
def build_dataset(samples):
    """samples: list of dicts with keys t, product, x (features), net,
    exit_t (resolution time). Returns them sorted by time."""
    return sorted(samples, key=lambda s: s["t"])


def walk_forward_probs(samples, fold_sec=FOLD_DAYS * 86400):
    """{(t, product): out-of-sample P(win)} for every sample whose fold had a
    trainable model. Each fold's model only sees samples that RESOLVED before
    the fold started. Also returns {(t, product): base_rate} so the decision
    threshold is out-of-sample too."""
    import numpy as np
    if not samples:
        return {}, {}
    ss = build_dataset(samples)
    t_first, t_last = ss[0]["t"], ss[-1]["t"]
    probs, bases = {}, {}
    start = t_first + fold_sec                     # first fold needs history
    while start <= t_last:
        end = start + fold_sec
        train = [s for s in ss if s["exit_t"] < start]
        test = [s for s in ss if start <= s["t"] < end]
        if test:
            m, _, base = _fit([s["x"] for s in train], [s["net"] > 0 for s in train])
            if m is not None:
                p = m.predict_proba(np.asarray([s["x"] for s in test], float))[:, 1]
                for s, pi in zip(test, p):
                    probs[(s["t"], s["product"])] = float(pi)
                    bases[(s["t"], s["product"])] = base
        start = end
    return probs, bases


# ---------------------------------------------------------------- live model
class TradeFilter:
    def __init__(self):
        self.model = None
        self.backend = None
        self.base_rate = None
        self.n_train = 0
        self.trained_at = None
        self.auto_on = False          # set from the replay A/B
        self.last_eval = None         # A/B summary from the last replay
        self._lock = threading.Lock()

    def fit(self, samples, now=None):
        """Train the live model on every sample that has RESOLVED by now."""
        now = now or time.time()
        done = [s for s in samples if s["exit_t"] <= now]
        m, backend, base = _fit([s["x"] for s in done], [s["net"] > 0 for s in done])
        with self._lock:
            if m is not None:
                self.model, self.backend, self.base_rate = m, backend, base
                self.n_train, self.trained_at = len(done), now
        return m is not None

    def active(self):
        from .. import settings
        mode = settings.get("trade_filter_mode")
        return self.model is not None and (mode == "on" or (mode == "auto" and self.auto_on))

    def threshold(self):
        from ..tunables import tv
        return (self.base_rate or 0.0) * tv("filter_strictness")

    def prob(self, x):
        import numpy as np
        with self._lock:
            m = self.model
        if m is None:
            return None
        return float(m.predict_proba(np.asarray([x], float))[0, 1])

    def allows(self, x):
        """(allowed?, prob). Always allowed when inactive/untrained."""
        if not self.active():
            return True, None
        p = self.prob(x)
        if p is None:
            return True, None
        return p >= self.threshold(), p

    def snapshot(self):
        return {"trained": self.model is not None, "backend": self.backend,
                "train_samples": self.n_train, "trained_at": self.trained_at,
                "base_win_rate": None if self.base_rate is None else round(self.base_rate, 3),
                "threshold": None if self.base_rate is None else round(self.threshold(), 3),
                "auto_on": self.auto_on, "active": self.active(),
                "last_eval": self.last_eval}


trade_filter = TradeFilter()
