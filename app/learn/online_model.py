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
from collections import deque

N_IN = 18
N_HID = 16

# quantile levels for the aleatoric band (P10 / P90)
QUANTILES = (0.10, 0.90)

FEAT_NAMES = ["rsi", "macd", "macd_delta", "mom_1h", "mom_4h", "vol_ratio",
              "imbalance", "spread", "asset_sent", "market_sent",
              "price_vs_sma20", "sma20_vs_sma50", "volatility",
              "funding", "oi_change", "ls_crowding", "taker_aggression",
              "mtf_align"]


def _clip(x, lo=-3.0, hi=3.0):
    return max(lo, min(hi, x))


def build_x(f, asset_sent, market_sent, deriv=None):
    """Build a pre-scaled feature vector from market features + sentiment
    + derivatives metrics (funding / OI / positioning from OKX)."""
    atr = f["atr"] or 1e-9
    d = deriv or {}
    return [
        f["rsi"] / 100 - 0.5,
        _clip(f["macd"] / atr),
        _clip(f["macd_delta"] / (0.25 * atr + 1e-9)),
        _clip(f["mom_1h"] * 100),
        _clip(f["mom_4h"] * 50),
        _clip(f["vol_ratio"] - 1, -2, 2),
        _clip(f["imbalance"], -1, 1),
        min(2.0, f["spread_bps"] / 10),
        _clip(asset_sent, -1, 1),
        _clip(market_sent, -1, 1),
        _clip((f["price"] / f["sma20"] - 1) * 100),
        _clip((f["sma20"] / f["sma50"] - 1) * 100),
        min(3.0, f["volatility"] * 300),
        d.get("funding_norm", 0.0),
        d.get("oi_change_norm", 0.0),
        d.get("ls_crowding", 0.0),
        d.get("taker_aggression", 0.0),
        # multi-timeframe trend alignment in [-1,1] (mean of 15m/1h/4h trend
        # signs) — a cross-timeframe confirmation feature for the model.
        _clip(f.get("mtf_align", 0.0), -1, 1),
    ]


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
        # AdaGrad accumulators
        self.gW1 = [[1e-8] * n_in for _ in range(n_hid)]
        self.gb1 = [1e-8] * n_hid
        self.gW2 = [1e-8] * n_hid
        self.gb2 = 1e-8
        self.gqW = [[1e-8] * n_hid for _ in self.quantiles]
        self.gqb = [1e-8 for _ in self.quantiles]
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
        target = _clip(fwd_return / 0.004, -1, 1)     # ±0.4% move = full signal
        if pred_at_record is not None and abs(target) > 0.15:
            self.acc_window.append(1 if pred_at_record * target > 0 else 0)
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
        self.lr_boost = 1.0
        self.n_updates = 0
        self.replay = deque(maxlen=self.replay.maxlen)
        self.replay_pr = deque(maxlen=self.replay_pr.maxlen)
        self.acc_window = deque(maxlen=self.acc_window.maxlen)
        self.loss_window = deque(maxlen=self.loss_window.maxlen)
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
            "avg_mse": round(loss, 4) if loss is not None else None,
            "lr_boost": round(self.lr_boost, 3),
            "warmed_up": self.n_updates >= 40,
        }


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

    def reset(self, base_seed=7):
        for i, m in enumerate(self.members):
            m.reset(seed=base_seed + 101 * i)

    def predict_with_uncertainty(self, x):
        """Return a dict:
          mean       committee mean of member medians
          epistemic  stdev of member medians (disagreement)
          aleatoric  mean member P10..P90 band half-width
          lo, hi     combined predictive band (mean ± total uncertainty)
          confidence in [0,1]: high when BOTH uncertainties are small.
        """
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
        # map total predictive uncertainty (on the ±1 scaled-return axis) to a
        # confidence multiplier: ~0 unc -> 1.0, growing unc -> toward 0.
        confidence = 1.0 / (1.0 + 4.0 * total)
        return {"mean": mean, "epistemic": epistemic, "aleatoric": aleatoric,
                "lo": mean - total, "hi": mean + total,
                "confidence": max(0.0, min(1.0, confidence))}

    def stats(self):
        st = dict(self.primary.stats())
        # disagreement across members on the last replayed samples is expensive;
        # expose a cheap structural summary instead.
        st["committee_members"] = len(self.members)
        return st


# `committee` is the live ensemble; `model` aliases its primary member so the
# existing persistence schema and any direct references keep working unchanged.
committee = Committee(n_members=3)
model = committee.primary
