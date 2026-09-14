"""Online neural network — a tiny MLP trained continuously (SGD + AdaGrad)
on live market features vs realized forward returns.

Continual-learning features:
  * experience replay buffer (mitigates catastrophic forgetting)
  * AdaGrad per-weight adaptive learning rates
  * directional-accuracy tracking on held-out (pre-prediction) labels
  * drift-triggered learning-rate boost
"""
import math, random
from collections import deque

N_IN = 17
N_HID = 16

FEAT_NAMES = ["rsi", "macd", "macd_delta", "mom_1h", "mom_4h", "vol_ratio",
              "imbalance", "spread", "asset_sent", "market_sent",
              "price_vs_sma20", "sma20_vs_sma50", "volatility",
              "funding", "oi_change", "ls_crowding", "taker_aggression"]


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
    ]


class TinyMLP:
    """13 → 16(tanh) → 1(tanh) regressor predicting scaled forward return."""

    def __init__(self, n_in=N_IN, n_hid=N_HID, lr=0.03, l2=1e-5, seed=7):
        rnd = random.Random(seed)
        s1 = (2.0 / n_in) ** 0.5
        s2 = (2.0 / n_hid) ** 0.5
        self.W1 = [[rnd.gauss(0, s1) for _ in range(n_in)] for _ in range(n_hid)]
        self.b1 = [0.0] * n_hid
        self.W2 = [rnd.gauss(0, s2) for _ in range(n_hid)]
        self.b2 = 0.0
        # AdaGrad accumulators
        self.gW1 = [[1e-8] * n_in for _ in range(n_hid)]
        self.gb1 = [1e-8] * n_hid
        self.gW2 = [1e-8] * n_hid
        self.gb2 = 1e-8
        self.lr = lr
        self.lr_boost = 1.0          # raised temporarily on drift
        self.l2 = l2
        self.n_updates = 0
        self.replay = deque(maxlen=4000)
        self.acc_window = deque(maxlen=300)   # directional accuracy
        self.loss_window = deque(maxlen=300)

    # ---------- forward ----------
    def _fwd(self, x):
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b)
             for row, b in zip(self.W1, self.b1)]
        y = math.tanh(sum(w * hi for w, hi in zip(self.W2, h)) + self.b2)
        return h, y

    def predict(self, x):
        return self._fwd(x)[1]

    # ---------- backward (single sample SGD) ----------
    def _sgd(self, x, target):
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
        # hidden layer
        for j in range(len(self.W1)):
            dh = dy * self.W2[j] * (1 - h[j] * h[j])
            row, grow = self.W1[j], self.gW1[j]
            for i in range(len(row)):
                g = dh * x[i] + self.l2 * row[i]
                grow[i] += g * g
                row[i] -= lr / math.sqrt(grow[i]) * g
            self.gb1[j] += dh * dh
            self.b1[j] -= lr / math.sqrt(self.gb1[j]) * dh

    def update(self, x, fwd_return, pred_at_record=None):
        """Learn from a labeled sample; also replays random past samples."""
        target = _clip(fwd_return / 0.004, -1, 1)     # ±0.4% move = full signal
        if pred_at_record is not None and abs(target) > 0.15:
            self.acc_window.append(1 if pred_at_record * target > 0 else 0)
        self._sgd(x, target)
        self.replay.append((x, target))
        # experience replay: 6 random past samples per new sample
        for _ in range(min(6, len(self.replay) - 1)):
            rx, rt = random.choice(self.replay)
            self._sgd(rx, rt)
        self.n_updates += 1
        if self.lr_boost > 1.0:                       # decay drift boost
            self.lr_boost = max(1.0, self.lr_boost * 0.995)

    # ---------- introspection ----------
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


model = TinyMLP()
