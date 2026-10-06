"""Online neural network — a tiny MLP trained continuously (SGD + AdaGrad)
on live market features vs realized forward returns.

Continual-learning features:
  * experience replay buffer, PRIORITIZED by prediction error
  * AdaGrad per-weight adaptive learning rates
  * directional-accuracy tracking on held-out (pre-prediction) labels
  * drift-triggered learning-rate boost
  * online per-feature input standardization (Welford)

Uncertainty features (Phase 3):
  * QUANTILE heads (P10/P90) trained with pinball loss on top of the shared
    hidden layer → an ALEATORIC uncertainty band around each prediction.
  * A COMMITTEE of differently-seeded members whose disagreement gives an
    EPISTEMIC uncertainty. Both feed the live sizer so the book bets small when
    the model is unsure and only presses size when the heads agree.
"""
import math, random
from ..config import CANDLE_GRANULARITY as _GRAN
from collections import deque

N_IN = 30          # 18 core + 6 chart-pattern + 2 advisor-lean + 4 slow-horizon (see build_x / FEAT_NAMES)
N_HID = 16

# Minimum |prediction| (on the scaled ±1 return axis) for a recorded prediction
# to count toward DIRECTIONAL accuracy. A warmup / just-reset head emits 0.0
# ("no opinion"); scoring 0.0 as a directional miss (0*target is never > 0)
# pinned accuracy near zero and triggered perpetual resets, which kept the ml
# sleeve at zero weight forever. A near-zero stance is abstention, not a miss.
ACC_MIN_CONVICTION = 0.05

# Reset rule for a head that has fitted the wrong mapping. It used to fire at
# acc <= 50% after only 20 scored predictions — pure noise for a coin-flip
# head (SE ~11%), so it reset ~1,000 times in a one-year replay and never kept
# anything it learned. Now: a real sample, and accuracy CONFIDENTLY below a
# coin flip (2 standard errors).
RESET_MIN_UPDATES = 500
RESET_MIN_SCORED = 200


def confidently_broken(stats):
    """True when a head's measured directional accuracy is reliably worse than
    a coin flip (see RESET_MIN_* above)."""
    acc, n = stats.get("directional_accuracy"), stats.get("acc_samples", 0)
    if acc is None or n < RESET_MIN_SCORED or stats.get("n_updates", 0) < RESET_MIN_UPDATES:
        return False
    return acc < 0.5 - 2 * math.sqrt(0.25 / n)


# quantile levels for the aleatoric band (P10 / P90)
QUANTILES = (0.10, 0.90)

# Auxiliary MULTI-HORIZON heads (Phase 2). The primary median head predicts the
# 30-min forward return; these extra heads predict the SAME snapshot's return at
# other horizons (a fast one and a slow one — see loop.AUX_HORIZONS) as
# auxiliary tasks on top of the shared hidden layer. They are never read by
# predict()/sizing; their only job is to force the shared representation to
# encode structure that is predictive across timescales, which both speeds up
# feedback (the fast head labels in minutes) and adds longer-context signal
# (the slow head) without changing the primary output or its persistence.
AUX_HEADS = 2

FEAT_NAMES = ["rsi", "macd", "macd_delta", "mom_1h", "mom_4h", "vol_ratio",
              "imbalance", "spread", "asset_sent", "market_sent",
              "price_vs_sma20", "sma20_vs_sma50", "volatility",
              "funding", "oi_change", "ls_crowding", "taker_aggression",
              "mtf_align",
              # chart-pattern features (appended so N_IN migration is clean)
              "pat_structure", "pat_sr", "pat_reversal",
              "pat_continuation", "pat_candle", "pat_divergence",
              # advisor leans appended LAST so the model can learn whether the
              # LLM / trained-model opinions add predictive value. Zero (and thus
              # inert) unless the respective advisor is enabled + llm_ml_feature.
              "llm_lean", "model_lean",
              # slow-horizon features (see build_x)
              "ema24_vs_96", "mom_72", "brk48_pos", "xs_mom"]


# Label scale: a forward return of +/- TARGET_SCALE maps to a full +/-1 target.
# Set to roughly the ~1.2% round-trip trading cost, so "full signal" means
# "a move big enough to pay for the trade". (Was 0.4%, i.e. sized for a 30-min
# 5m-bar horizon where no prediction could ever clear costs.)
TARGET_SCALE = 0.012
# accuracy only counts samples whose move was >= this fraction of TARGET_SCALE
ACC_MIN_MOVE = 0.5


def _clip(x, lo=-3.0, hi=3.0):
    return max(lo, min(hi, x))


# Inputs held at a constant 0 (they stay in the vector so N_IN and saved
# models keep their shape). Measured ~no edge (order-book imbalance/spread are
# seconds-scale; sentiment was negative; chart patterns ~0 over 162k scored
# signals) and they don't exist in history, so masking them also keeps the
# history warm-start and live inputs identical. A constant input standardises
# to 0 and simply never moves the net.
MASKED_FEATURES = frozenset({
    "imbalance", "spread", "asset_sent", "market_sent",
    "pat_structure", "pat_sr", "pat_reversal", "pat_continuation",
    "pat_candle", "pat_divergence",
})


def build_x(f, asset_sent, market_sent, deriv=None):
    """Build a pre-scaled feature vector from market features + sentiment
    + derivatives metrics (funding / OI / positioning from OKX), with the
    MASKED_FEATURES held at 0."""
    x = _build_x_raw(f, asset_sent, market_sent, deriv)
    return [0.0 if name in MASKED_FEATURES else v for name, v in zip(FEAT_NAMES, x)]


def _build_x_raw(f, asset_sent, market_sent, deriv=None):
    atr = f["atr"] or 1e-9
    d = deriv or {}
    return [
        f["rsi"] / 100 - 0.5,
        _clip(f["macd"] / atr),
        _clip(f["macd_delta"] / (0.25 * atr + 1e-9)),
        # mom_1h / mom_4h are 12-bar / 48-bar returns (12h / 48h on 1h bars)
        _clip(f["mom_1h"] * 30),
        _clip(f["mom_4h"] * 15),
        _clip(f["vol_ratio"] - 1, -2, 2),
        _clip(f["imbalance"], -1, 1),
        min(2.0, f["spread_bps"] / 10),
        _clip(asset_sent, -1, 1),
        _clip(market_sent, -1, 1),
        _clip((f["price"] / f["sma20"] - 1) * 100),
        _clip((f["sma20"] / f["sma50"] - 1) * 100),
        # per-bar volatility normalised to the native bar size (1h ≈ 1%)
        min(3.0, f["volatility"] * 300 * (300 / _GRAN) ** 0.5),
        d.get("funding_norm", 0.0),
        d.get("oi_change_norm", 0.0),
        d.get("ls_crowding", 0.0),
        d.get("taker_aggression", 0.0),
        # multi-timeframe trend alignment in [-1,1] (mean of 15m/1h/4h trend
        # signs) — a cross-timeframe confirmation feature for the model.
        _clip(f.get("mtf_align", 0.0), -1, 1),
    ] + _pattern_feats(f) + [
        # advisor leans (0 unless the caller stamped them AND llm_ml_feature is
        # on): this is how "the LLM teaches the ML" — the online model gets the
        # LLM's / trained-model's directional opinion as an input feature and
        # learns from realized outcomes whether it's worth anything.
        _clip(f.get("llm_lean", 0.0), -1, 1),
        _clip(f.get("model_lean", 0.0), -1, 1),
        # slow-horizon features (appended LAST; adding them changes N_IN, which
        # makes persistence start a fresh model instead of loading weights
        # learned on the old 5m/30-min mapping)
        _clip(((f.get("ema24") or 0) / (f.get("ema96") or 1) - 1) * 20)
            if f.get("ema24") and f.get("ema96") else 0.0,
        _clip((f.get("mom_72") or 0.0) * 10),
        _breakout_pos(f),
        _clip(f.get("xs_mom", 0.0), -3, 3),
    ]


def _breakout_pos(f):
    """Position of price inside the prior 48-bar channel, mapped to [-1, 1]
    (beyond the channel saturates at +/-1). 0 when unavailable."""
    hi, lo = f.get("hi48"), f.get("lo48")
    if not hi or not lo or hi <= lo:
        return 0.0
    return _clip((f["price"] - lo) / (hi - lo) * 2 - 1, -1.5, 1.5)


def stamp_advisor_leans(f, product):
    """Write the LLM + model-advisor leans onto a features dict so build_x picks
    them up as input features. No-op (leaves them 0 / inert) when the
    `llm_ml_feature` toggle is off, so the ML ignores the advisors by default.

    Called on BOTH the training path (loop.collect_features) and the inference
    path (signals.compute) so the model is trained and scored on the same inputs.
    Any advisor error is swallowed — a missing/disabled advisor just leaves 0.
    """
    if not isinstance(f, dict):
        return
    from ..tunables import tv
    if not tv("llm_ml_feature"):
        f.setdefault("llm_lean", 0.0)
        f.setdefault("model_lean", 0.0)
        return
    try:
        from .llm_advisor import advisor as _llm
        f["llm_lean"] = float(_llm.lean(product))
    except Exception:
        f.setdefault("llm_lean", 0.0)
    try:
        from .model_advisor import advisor as _model_adv
        f["model_lean"] = float(_model_adv.lean(product))
    except Exception:
        f.setdefault("model_lean", 0.0)


def _pattern_feats(f):
    """The 6 chart-pattern scores (already in [-1,1]) appended to the ML input
    vector. Defaults to zeros when no pattern report is on the features dict
    (short history / backtest) so the vector width is always N_IN."""
    from ..signals.patterns import feature_vector
    rep = f.get("patterns") if isinstance(f, dict) else None
    return [_clip(v, -1, 1) for v in feature_vector(rep)]


class TinyMLP:
    """17 → 16(tanh) → 1(tanh) regressor predicting scaled forward return."""

    def __init__(self, n_in=N_IN, n_hid=N_HID, lr=0.03, l2=1e-5, seed=7):
        rnd = random.Random(seed)
        # dedicated RNG for replay sampling so training is deterministic and
        # independent of global random state / test execution order (the shared
        # `random` module made committee uncertainty flaky across suite runs).
        self._rng = random.Random(seed * 2 + 1)
        s1 = (2.0 / n_in) ** 0.5
        s2 = (2.0 / n_hid) ** 0.5
        self.W1 = [[rnd.gauss(0, s1) for _ in range(n_in)] for _ in range(n_hid)]
        self.b1 = [0.0] * n_hid
        self.W2 = [rnd.gauss(0, s2) for _ in range(n_hid)]
        self.b2 = 0.0
        # QUANTILE heads (P10/P90): one linear head per quantile on top of the
        # SHARED hidden layer, trained with pinball loss. Initialised near the
        # median head so the band starts tight and widens as data disagrees.
        self.quantiles = list(QUANTILES)
        self.qW = [[rnd.gauss(0, s2) for _ in range(n_hid)]
                   for _ in self.quantiles]
        self.qb = [0.0 for _ in self.quantiles]
        # AUXILIARY multi-horizon heads (Phase 2): one linear+tanh head per aux
        # horizon on top of the SHARED hidden layer. Trained with plain MSE
        # against that horizon's scaled forward return. They shape the hidden
        # layer but are never surfaced to sizing / predict().
        self.aW = [[rnd.gauss(0, s2) for _ in range(n_hid)]
                   for _ in range(AUX_HEADS)]
        self.ab = [0.0 for _ in range(AUX_HEADS)]
        # AdaGrad accumulators
        self.gW1 = [[1e-8] * n_in for _ in range(n_hid)]
        self.gb1 = [1e-8] * n_hid
        self.gW2 = [1e-8] * n_hid
        self.gb2 = 1e-8
        self.gqW = [[1e-8] * n_hid for _ in self.quantiles]
        self.gqb = [1e-8 for _ in self.quantiles]
        self.gaW = [[1e-8] * n_hid for _ in range(AUX_HEADS)]
        self.gab = [1e-8 for _ in range(AUX_HEADS)]
        self.lr = lr
        self.lr_boost = 1.0          # raised temporarily on drift
        self.l2 = l2
        self.n_updates = 0
        # prioritized experience replay: parallel deques of samples and their
        # last-seen squared error (sampling weight). High-error, informative
        # samples get replayed more often than easy ones -> better sample
        # efficiency than uniform replay.
        self.replay = deque(maxlen=4000)          # (x, target)
        self.replay_pr = deque(maxlen=4000)       # priority (last sq-error + eps)
        self.acc_window = deque(maxlen=300)   # directional accuracy
        self.loss_window = deque(maxlen=300)
        # live sample stream for the ML-learning dashboard: each matured update
        # records (predicted, realized-target, correct?) so the UI can show the
        # model actually learning. Cosmetic only — nothing reads it for control.
        self.recent = deque(maxlen=150)
        # online input standardization (Welford running mean/var per feature),
        # so feature scaling self-calibrates as the universe/regime changes
        # instead of relying on hand-tuned constants in build_x.
        self.feat_n = 0
        self.feat_mean = [0.0] * n_in
        self.feat_M2 = [0.0] * n_in

    # ---------- online input standardization ----------
    def _observe_features(self, x):
        """Update running per-feature mean/var (Welford) from a raw input."""
        self.feat_n += 1
        for i, xi in enumerate(x):
            d = xi - self.feat_mean[i]
            self.feat_mean[i] += d / self.feat_n
            self.feat_M2[i] += d * (xi - self.feat_mean[i])

    def _standardize(self, x):
        """Center/scale a raw input by the running stats. Falls back to the raw
        (already roughly-scaled) value until we have enough samples to trust the
        estimate, and clips to keep tanh units sane."""
        if self.feat_n < 30:
            return list(x)
        out = []
        for i, xi in enumerate(x):
            var = self.feat_M2[i] / (self.feat_n - 1) if self.feat_n > 1 else 1.0
            sd = math.sqrt(var) if var > 1e-12 else 1.0
            out.append(_clip((xi - self.feat_mean[i]) / sd))
        return out

    # ---------- forward ----------
    def _fwd(self, x):
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b)
             for row, b in zip(self.W1, self.b1)]
        y = math.tanh(sum(w * hi for w, hi in zip(self.W2, h)) + self.b2)
        return h, y

    def _quantiles_from_h(self, h):
        """Quantile-head outputs (same tanh-bounded scale as the median)."""
        return [math.tanh(sum(w * hi for w, hi in zip(qw, h)) + qb)
                for qw, qb in zip(self.qW, self.qb)]

    def predict(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != len(self.feat_mean):
            return 0.0
        return self._fwd(self._standardize(x))[1]

    def predict_quantiles(self, x):
        """Return (p_lo, median, p_hi) on the model's scaled-return scale.
        The band is sorted so p_lo <= median <= p_hi even before the heads have
        fully separated. Width = ALEATORIC (irreducible) uncertainty."""
        h, y = self._fwd(self._standardize(x))
        q = self._quantiles_from_h(h)
        lo, hi = min(q), max(q)
        return (min(lo, y), y, max(hi, y))

    # ---------- backward (single sample SGD) ----------
    def _sgd(self, x_raw, target):
        x = self._standardize(x_raw)
        h, y = self._fwd(x)
        err = y - target
        self.loss_window.append(err * err)
        dy = err * (1 - y * y)                       # dL/dz2
        lr = self.lr * self.lr_boost
        # output layer
        for j in range(len(self.W2)):
            g = dy * h[j] + self.l2 * self.W2[j]
            self.gW2[j] += g * g
            self.W2[j] -= lr / math.sqrt(self.gW2[j]) * g
        gb = dy
        self.gb2 += gb * gb
        self.b2 -= lr / math.sqrt(self.gb2) * gb
        # ---- QUANTILE heads: pinball (quantile) loss on the shared hidden ----
        # For quantile tau, gradient of pinball loss wrt the head output is
        # -(tau)      when target > q   (under-estimate -> push q up)
        # +(1 - tau)  when target <= q  (over-estimate  -> push q down)
        # Each head backprops into its own weights AND into the hidden layer so
        # the band shape can specialise. dq accumulates the hidden-layer signal.
        dq_hidden = [0.0] * len(h)
        for qi, tau in enumerate(self.quantiles):
            qz = sum(w * hi for w, hi in zip(self.qW[qi], h)) + self.qb[qi]
            qy = math.tanh(qz)
            grad = -tau if target > qy else (1 - tau)     # d pinball / d qy
            dqz = grad * (1 - qy * qy)                     # through tanh
            for j in range(len(self.qW[qi])):
                g = dqz * h[j] + self.l2 * self.qW[qi][j]
                self.gqW[qi][j] += g * g
                self.qW[qi][j] -= lr / math.sqrt(self.gqW[qi][j]) * g
                dq_hidden[j] += dqz * self.qW[qi][j]
            self.gqb[qi] += dqz * dqz
            self.qb[qi] -= lr / math.sqrt(self.gqb[qi]) * dqz
        # hidden layer (median head + quantile heads share these weights)
        for j in range(len(self.W1)):
            dh = (dy * self.W2[j] + dq_hidden[j]) * (1 - h[j] * h[j])
            row, grow = self.W1[j], self.gW1[j]
            for i in range(len(row)):
                g = dh * x[i] + self.l2 * row[i]
                grow[i] += g * g
                row[i] -= lr / math.sqrt(grow[i]) * g
            self.gb1[j] += dh * dh
            self.b1[j] -= lr / math.sqrt(self.gb1[j]) * dh
        return err * err

    def _sample_replay_idx(self):
        """Priority-proportional sample from the replay buffer (roulette wheel).
        Falls back to uniform if priorities aren't populated."""
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = self._rng = random.Random(12345)
        total = sum(self.replay_pr)
        if total <= 0:
            return rng.randrange(len(self.replay))
        r = rng.random() * total
        acc = 0.0
        for i, p in enumerate(self.replay_pr):
            acc += p
            if acc >= r:
                return i
        return len(self.replay) - 1

    def update(self, x, fwd_return, pred_at_record=None):
        """Learn from a labeled sample; also replays PRIORITIZED past samples."""
        if not isinstance(x, (list, tuple)) or len(x) != len(self.feat_mean):
            return
        target = _clip(fwd_return / TARGET_SCALE, -1, 1)     # ±1.2% move = full signal
        # Score directional accuracy ONLY on samples where the realized move was
        # meaningful AND the head actually took a directional stance. Abstentions
        # (|pred| ~ 0, i.e. warmup / just-reset) are excluded — counting them as
        # misses is what pinned accuracy at 0 and drove the reset doom loop.
        # "meaningful" = at least half the round-trip cost (|target| >= 0.5),
        # so accuracy reflects moves a trade could actually profit from — not
        # a coin-flip on noise-sized wiggles.
        scored = (pred_at_record is not None and abs(target) >= ACC_MIN_MOVE
                  and abs(pred_at_record) > ACC_MIN_CONVICTION)
        if scored:
            self.acc_window.append(1 if pred_at_record * target > 0 else 0)
            self.recent.append((round(float(pred_at_record), 4),
                                round(float(target), 4),
                                bool(pred_at_record * target > 0)))
        # update running feature stats from the raw input, then train
        self._observe_features(x)
        se = self._sgd(x, target)
        self.replay.append((x, target))
        self.replay_pr.append(se + 1e-4)              # priority = recent error
        # prioritized experience replay: 6 samples per new one, biased toward
        # high-error (informative) memories; refresh their priority as we go.
        for _ in range(min(6, len(self.replay) - 1)):
            idx = self._sample_replay_idx()
            rx, rt = self.replay[idx]
            new_se = self._sgd(rx, rt)
            self.replay_pr[idx] = new_se + 1e-4
        self.n_updates += 1
        if self.lr_boost > 1.0:                       # decay drift boost
            self.lr_boost = max(1.0, self.lr_boost * 0.995)

    def update_aux(self, x, fwd_return, head):
        """Train ONE auxiliary horizon head on a matured sample.

        Unlike update(), this does NOT touch the primary median head, quantile
        heads, replay buffer, n_updates or the directional-accuracy window — it
        only fits the given aux head AND backprops that head's error into the
        SHARED hidden layer, so multi-horizon structure improves the common
        representation the primary head reads from. `head` selects which aux
        horizon (0..AUX_HEADS-1).
        """
        if not isinstance(x, (list, tuple)) or len(x) != len(self.feat_mean):
            return
        if not (0 <= head < len(self.aW)):
            return
        target = _clip(fwd_return / TARGET_SCALE, -1, 1)
        self._observe_features(x)
        xs = self._standardize(x)
        h, _ = self._fwd(xs)
        lr = self.lr * self.lr_boost
        az = sum(w * hi for w, hi in zip(self.aW[head], h)) + self.ab[head]
        ay = math.tanh(az)
        daz = (ay - target) * (1 - ay * ay)
        dh = [0.0] * len(h)
        for j in range(len(self.aW[head])):
            g = daz * h[j] + self.l2 * self.aW[head][j]
            self.gaW[head][j] += g * g
            dh[j] = daz * self.aW[head][j]
            self.aW[head][j] -= lr / math.sqrt(self.gaW[head][j]) * g
        self.gab[head] += daz * daz
        self.ab[head] -= lr / math.sqrt(self.gab[head]) * daz
        # backprop the aux error into the SHARED hidden layer (this is the whole
        # point — the primary head benefits from the auxiliary supervision).
        for j in range(len(self.W1)):
            dhj = dh[j] * (1 - h[j] * h[j])
            row, grow = self.W1[j], self.gW1[j]
            for i in range(len(row)):
                g = dhj * xs[i] + self.l2 * row[i]
                grow[i] += g * g
                row[i] -= lr / math.sqrt(grow[i]) * g
            self.gb1[j] += dhj * dhj
            self.b1[j] -= lr / math.sqrt(self.gb1[j]) * dhj

    # ---------- introspection ----------
    def reset(self, seed=7):
        """Re-init weights and wipe online stats. Used when directional
        accuracy is stuck at/below a coin flip so we stop fitting the
        wrong mapping (47k updates at 26% acc does not 'warm up' — it
        digs in)."""
        fresh = TinyMLP(n_in=len(self.feat_mean), n_hid=len(self.W1),
                        lr=self.lr, l2=self.l2, seed=seed)
        self.W1, self.b1 = fresh.W1, fresh.b1
        self.W2, self.b2 = fresh.W2, fresh.b2
        self.gW1, self.gb1 = fresh.gW1, fresh.gb1
        self.gW2, self.gb2 = fresh.gW2, fresh.gb2
        self.qW, self.qb = fresh.qW, fresh.qb
        self.gqW, self.gqb = fresh.gqW, fresh.gqb
        self.aW, self.ab = fresh.aW, fresh.ab
        self.gaW, self.gab = fresh.gaW, fresh.gab
        self.lr_boost = 1.0
        self.n_updates = 0
        self.replay = deque(maxlen=self.replay.maxlen)
        self.replay_pr = deque(maxlen=self.replay_pr.maxlen)
        self.acc_window = deque(maxlen=self.acc_window.maxlen)
        self.loss_window = deque(maxlen=self.loss_window.maxlen)
        self.recent = deque(maxlen=self.recent.maxlen)
        self.feat_n = 0
        self.feat_mean = list(fresh.feat_mean)
        self.feat_M2 = list(fresh.feat_M2)

    def stats(self):
        acc = (sum(self.acc_window) / len(self.acc_window)
               if self.acc_window else None)
        loss = (sum(self.loss_window) / len(self.loss_window)
                if self.loss_window else None)
        return {
            "n_updates": self.n_updates,
            "replay_buffer": len(self.replay),
            "directional_accuracy": round(acc, 3) if acc is not None else None,
            "acc_samples": len(self.acc_window),   # scored (non-abstention) preds
            "avg_mse": round(loss, 4) if loss is not None else None,
            "lr_boost": round(self.lr_boost, 3),
            "warmed_up": self.n_updates >= 40,
        }


# ---------------------------------------------------------------- conformal
# Target MISCOVERAGE for the calibrated predictive interval. alpha=0.20 asks for
# an 80% interval: over the long run the realized (scaled) return should fall
# inside [lo, hi] ~80% of the time. Coverage is what the sizer's confidence is
# ultimately staking on, so we want it to MEAN something.
CONFORMAL_ALPHA = 0.20
# rolling calibration-set size and ACI step. The window keeps calibration LOCAL
# (recent regime) rather than averaging over ancient history; gamma is how
# aggressively Adaptive Conformal Inference chases the coverage target when the
# regime shifts and the raw bands stop covering.
CONFORMAL_WINDOW = 500
CONFORMAL_GAMMA = 0.02


def _quantile_sorted(sorted_vals, q):
    """Linear-interpolation quantile of an ALREADY-SORTED list, q in [0,1].
    Pure-Python (this module is numpy-free on the hot path)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (n - 1)
    lo_i = int(math.floor(pos))
    hi_i = int(math.ceil(pos))
    if lo_i == hi_i:
        return sorted_vals[lo_i]
    frac = pos - lo_i
    return sorted_vals[lo_i] * (1 - frac) + sorted_vals[hi_i] * frac


class ConformalCalibrator:
    """Split/Adaptive CONFORMAL PREDICTION over the committee's raw band.

    The committee's quantile heads are TRAINED but never CALIBRATED — nothing
    guarantees the realized return actually lands inside [lo, hi] at the claimed
    rate, and on fat-tailed crypto trained quantiles are usually OVER-confident
    (bands too narrow → the sizer presses size when it shouldn't).

    This wraps the model (no retraining) with an inductive-conformal correction:
      * keep a rolling calibration set of CQR nonconformity scores
            s = max(lo - y, y - hi)
        (how far outside the raw band the truth fell; NEGATIVE when it fell
        comfortably inside).
      * qhat = the (1-alpha)-quantile of those scores, with the standard
        finite-sample (n+1)/n inflation. The calibrated interval is then
            [lo - qhat, hi + qhat]
        which carries a distribution-free ~(1-alpha) marginal coverage
        guarantee. qhat < 0 means the raw bands were too WIDE and we TIGHTEN.
      * ADAPTIVE CONFORMAL INFERENCE (ACI): the effective miscoverage alpha_t is
        nudged every observation — widen after a miss, tighten after a hit — so
        coverage self-corrects through regime change instead of drifting.

    All state is cheap and JSON-serialisable so it rides the normal snapshot.
    """

    def __init__(self, alpha=CONFORMAL_ALPHA, window=CONFORMAL_WINDOW,
                 gamma=CONFORMAL_GAMMA):
        self.alpha_target = alpha        # desired long-run miscoverage
        self.window = window
        self.gamma = gamma
        self.alpha_t = alpha             # ACI-adapted miscoverage (live)
        self.scores = deque(maxlen=window)      # CQR nonconformity scores
        self._cov = deque(maxlen=window)        # 1/0 coverage over recent obs
        self.n_seen = 0

    # need a minimum calibration set before the quantile is trustworthy;
    # below it we behave exactly like the old (uncalibrated) path (qhat=0).
    @property
    def ready(self):
        return len(self.scores) >= 20

    def qhat(self):
        """Conformal correction on the same scaled-return axis as the band."""
        if not self.ready:
            return 0.0
        n = len(self.scores)
        level = 1.0 - self.alpha_t                    # ACI-adapted coverage
        # finite-sample conformal rank (1-alpha)(n+1)/n, capped at 1.0
        rank = min(1.0, max(0.0, level * (n + 1) / n))
        return _quantile_sorted(sorted(self.scores), rank)

    def calibrate(self, lo, hi):
        """Widen (or tighten) a raw band into a coverage-calibrated one."""
        q = self.qhat()
        cal_lo, cal_hi = lo - q, hi + q
        if cal_hi < cal_lo:                            # extreme tighten: collapse
            mid = 0.5 * (cal_lo + cal_hi)
            cal_lo = cal_hi = mid
        return cal_lo, cal_hi, q

    def observe(self, lo, hi, y):
        """Fold one matured (raw_band, realized_y) pair into the calibration set
        and run the ACI coverage update. Call with the RAW band that was (re)
        produced for this sample and the realized scaled return y."""
        # coverage of the interval we WOULD have emitted (uses current qhat),
        # tallied BEFORE this score joins the set.
        q = self.qhat()
        covered = (lo - q) <= y <= (hi + q)
        self._cov.append(1 if covered else 0)
        # ACI: alpha_{t+1} = alpha_t + gamma*(alpha - err). err=1 on a miss lowers
        # alpha_t → higher coverage target → wider next time; a hit nudges back.
        err = 0.0 if covered else 1.0
        self.alpha_t = min(0.60, max(0.01,
                                     self.alpha_t + self.gamma * (self.alpha_target - err)))
        # CQR nonconformity score (negative when y sat comfortably inside).
        self.scores.append(max(lo - y, y - hi))
        self.n_seen += 1

    def coverage(self):
        """Empirical rolling coverage of the calibrated interval, or None."""
        if not self._cov:
            return None
        return sum(self._cov) / len(self._cov)

    def stats(self):
        return {
            "n_scores": len(self.scores),
            "n_seen": self.n_seen,
            "alpha_target": round(self.alpha_target, 4),
            "alpha_t": round(self.alpha_t, 4),
            "qhat": round(self.qhat(), 5),
            "coverage": (round(self.coverage(), 4)
                         if self.coverage() is not None else None),
            "ready": self.ready,
        }

    def to_dict(self):
        return {
            "alpha_target": self.alpha_target,
            "alpha_t": self.alpha_t,
            "window": self.window,
            "gamma": self.gamma,
            "scores": list(self.scores),
            "cov": list(self._cov),
            "n_seen": self.n_seen,
        }

    def load_dict(self, d):
        if not d:
            return False
        try:
            self.alpha_target = float(d.get("alpha_target", self.alpha_target))
            self.alpha_t = float(d.get("alpha_t", self.alpha_target))
            self.gamma = float(d.get("gamma", self.gamma))
            w = int(d.get("window", self.window))
            self.window = w
            self.scores = deque((float(s) for s in d.get("scores", [])), maxlen=w)
            self._cov = deque((int(c) for c in d.get("cov", [])), maxlen=w)
            self.n_seen = int(d.get("n_seen", len(self.scores)))
            return True
        except (TypeError, ValueError):
            return False


class Committee:
    """A small ENSEMBLE of independently-seeded TinyMLPs.

    Two complementary uncertainties come out of this:
      * EPISTEMIC — how much the members DISAGREE about the median. High when
        the model is in unfamiliar territory / hasn't learned the pattern yet.
        Shrinks as they all converge on the same answer with more data.
      * ALEATORIC — the average P10..P90 quantile-band width across members.
        The irreducible noise of the target itself.
    The live sizer uses the combined uncertainty to bet small when unsure and
    only press size when the members agree AND their bands are tight.

    members[0] is the "primary" and keeps the exact persistence schema of the
    old single model, so existing snapshots load unchanged.
    """

    def __init__(self, n_members=3, base_seed=7, **kw):
        self.members = [TinyMLP(seed=base_seed + 101 * i, **kw)
                        for i in range(max(1, n_members))]
        # CONFORMAL layer: calibrates the raw ensemble band to a guaranteed
        # coverage level and derives the sizer's confidence from the calibrated
        # (not the trained-but-uncalibrated) width. Fed matured labels via
        # observe_outcome() from the training loop.
        self.calibrator = ConformalCalibrator()

    @property
    def primary(self):
        return self.members[0]

    # proxy the scalar bits loop.py / engine.py read off the old `model`
    @property
    def n_updates(self):
        return self.primary.n_updates

    @property
    def lr_boost(self):
        return self.primary.lr_boost

    @lr_boost.setter
    def lr_boost(self, v):
        for m in self.members:
            m.lr_boost = v

    def predict(self, x):
        """Committee mean of the median heads."""
        return sum(m.predict(x) for m in self.members) / len(self.members)

    def update(self, x, fwd_return, pred_at_record=None):
        """Train every member. Members differ only by init seed + replay
        sampling, which is enough to keep their errors partially decorrelated."""
        for m in self.members:
            m.update(x, fwd_return, pred_at_record=pred_at_record)

    def update_aux(self, x, fwd_return, head):
        """Train the auxiliary horizon head on every member."""
        for m in self.members:
            m.update_aux(x, fwd_return, head)

    def reset(self, base_seed=7):
        for i, m in enumerate(self.members):
            m.reset(seed=base_seed + 101 * i)

    def _raw_band(self, x):
        """Committee mean + the RAW (uncalibrated) predictive band and its two
        uncertainty components. Shared by prediction and calibration so the
        calibration score is measured against the exact band the sizer sees."""
        meds, halfwidths = [], []
        for m in self.members:
            lo, med, hi = m.predict_quantiles(x)
            meds.append(med)
            halfwidths.append((hi - lo) / 2)
        mean = sum(meds) / len(meds)
        if len(meds) > 1:
            var = sum((v - mean) ** 2 for v in meds) / (len(meds) - 1)
            epistemic = math.sqrt(var)
        else:
            epistemic = 0.0
        aleatoric = sum(halfwidths) / len(halfwidths)
        total = math.sqrt(epistemic ** 2 + aleatoric ** 2)
        return mean, epistemic, aleatoric, total

    def predict_with_uncertainty(self, x):
        """Return a dict:
          mean          committee mean of member medians
          epistemic     stdev of member medians (disagreement)
          aleatoric     mean member P10..P90 band half-width
          lo, hi        CONFORMAL-CALIBRATED predictive band
          raw_lo, raw_hi the pre-calibration (mean ± total) band
          qhat          conformal correction applied (+widen / -tighten)
          calibrated    True once the calibration set is warm
          confidence in [0,1]: derived from the CALIBRATED half-width, so it
                        reflects an empirically-verified ~(1-alpha) interval
                        rather than the model's trained-but-unchecked band.
        """
        mean, epistemic, aleatoric, total = self._raw_band(x)
        raw_lo, raw_hi = mean - total, mean + total
        cal_lo, cal_hi, qhat = self.calibrator.calibrate(raw_lo, raw_hi)
        # confidence from the CALIBRATED half-width on the ±1 scaled-return axis.
        cal_half = max(0.0, (cal_hi - cal_lo) / 2)
        confidence = 1.0 / (1.0 + 4.0 * cal_half)
        return {"mean": mean, "epistemic": epistemic, "aleatoric": aleatoric,
                "lo": cal_lo, "hi": cal_hi,
                "raw_lo": raw_lo, "raw_hi": raw_hi,
                "qhat": qhat, "calibrated": self.calibrator.ready,
                "confidence": max(0.0, min(1.0, confidence))}

    def observe_outcome(self, x, fwd_return):
        """Fold a matured (features, realized forward-return) pair into the
        conformal calibration set. `fwd_return` is the RAW return; it is scaled
        onto the model's ±1 target axis exactly like update() does so the score
        lives on the same axis as the band. Safe to call before the band heads
        have warmed — the calibrator just accumulates until it is `ready`."""
        if not isinstance(x, (list, tuple)) or len(x) != len(self.primary.feat_mean):
            return
        y = _clip(fwd_return / TARGET_SCALE, -1, 1)
        mean, _epi, _ale, total = self._raw_band(x)
        raw_lo, raw_hi = mean - total, mean + total
        self.calibrator.observe(raw_lo, raw_hi, y)

    def stats(self):
        st = dict(self.primary.stats())
        # disagreement across members on the last replayed samples is expensive;
        # expose a cheap structural summary instead.
        st["committee_members"] = len(self.members)
        st["conformal"] = self.calibrator.stats()
        return st

    def live_state(self, x=None):
        """Introspection payload for the ML-learning dashboard tab. Shows the
        data flowing through the net: feature names, the online standardization
        stats, the live learning curves, the recent (pred vs realized) stream,
        and — when a feature row `x` is supplied — that row raw + standardized
        with the model's current prediction and uncertainty band."""
        m = self.primary
        state = {
            "feature_names": list(FEAT_NAMES),
            "n_updates": m.n_updates,
            "directional_accuracy": (round(sum(m.acc_window) / len(m.acc_window), 3)
                                     if m.acc_window else None),
            "avg_mse": (round(sum(m.loss_window) / len(m.loss_window), 5)
                        if m.loss_window else None),
            "lr_boost": round(m.lr_boost, 3),
            "warmed_up": m.n_updates >= 40,
            "replay_buffer": len(m.replay),
            "committee_members": len(self.members),
            # learning curves (oldest→newest); rolling directional-accuracy and MSE
            "acc_curve": [round(sum(list(m.acc_window)[max(0, i - 20):i + 1]) /
                                len(list(m.acc_window)[max(0, i - 20):i + 1]), 3)
                          for i in range(len(m.acc_window))][-120:],
            "loss_curve": [round(v, 5) for v in list(m.loss_window)][-120:],
            "recent": [{"pred": p, "target": t, "correct": ok}
                       for (p, t, ok) in list(m.recent)[-40:]],
            "feat_mean": [round(v, 4) for v in m.feat_mean],
            "feat_std": [round((m.feat_M2[i] / m.feat_n) ** 0.5, 4)
                         if m.feat_n > 1 else 0.0 for i in range(len(m.feat_mean))],
            "conformal": self.calibrator.stats(),
        }
        if isinstance(x, (list, tuple)) and len(x) == len(m.feat_mean):
            u = self.predict_with_uncertainty(x)
            state["sample"] = {
                "raw": [round(float(v), 4) for v in x],
                "standardized": [round(float(v), 3) for v in m._standardize(x)],
                "prediction": round(u["mean"], 4),
                "lo": round(u["lo"], 4), "hi": round(u["hi"], 4),
                "epistemic": round(u["epistemic"], 4),
                "aleatoric": round(u["aleatoric"], 4),
                "confidence": round(u["confidence"], 3),
                "calibrated": u["calibrated"],
            }
        return state


# `committee` is the live ensemble; `model` aliases its primary member so the
# existing persistence schema and any direct references keep working unchanged.
committee = Committee(n_members=3)
model = committee.primary
