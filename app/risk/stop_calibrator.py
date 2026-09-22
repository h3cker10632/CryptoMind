"""Conformal calibration of the ATR stop multiple from realized excursions.

The risk manager sizes every stop as `stop_atr_mult * ATR` — a GUESSED constant
(default 2.0). Too tight and normal volatility shakes good trades out at a loss;
too wide and each loser bleeds more than it needs to. Nothing in the guess is
tied to how far price *actually* wanders against a position in this market.

This module applies CONFORMAL PREDICTION (the same distribution-free calibration
technique adopted for the online model) to turn that guess into a coverage
guarantee: choose the stop multiple as the (1-alpha) conformal quantile of the
realized MAXIMUM ADVERSE EXCURSION (MAE), measured in ATR units, so only ~alpha
of otherwise-surviving trades get stopped by within-trade noise.

CENSORING — the one subtlety that makes this correct:
    A trade closed BY ITS STOP has an MAE bounded by the stop distance itself;
    it cannot tell us how far price *would* have travelled. Feeding those back
    would be circular (the stop calibrating on itself). So we only fold in the
    UNCENSORED excursions — trades that exited for a NON-STOP reason
    (take-profit, signal flip, peak-giveback, exit-advisor cut). Those are
    exactly the trades that dipped against us and SURVIVED, which is the
    distribution we want the stop to sit just outside of.

All state is small and JSON-serialisable so it rides the normal snapshot.
"""
from collections import deque

# target premature-stop rate: fraction of otherwise-surviving trades we accept
# being shaken out by noise. 0.10 -> a stop wide enough to keep ~90% of them.
STOP_ALPHA = 0.10
STOP_WINDOW = 300          # rolling calibration set (recent regime)
MIN_CALIB = 30             # need this many uncensored excursions before trusting


def _quantile_sorted(sorted_vals, q):
    """Linear-interpolation quantile of an ALREADY-SORTED list, q in [0,1]."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (n - 1)
    import math
    lo_i = int(math.floor(pos))
    hi_i = int(math.ceil(pos))
    if lo_i == hi_i:
        return sorted_vals[lo_i]
    frac = pos - lo_i
    return sorted_vals[lo_i] * (1 - frac) + sorted_vals[hi_i] * frac


class StopCalibrator:
    """Rolling conformal calibrator for the ATR stop (and, by ratio, take)."""

    def __init__(self, alpha=STOP_ALPHA, window=STOP_WINDOW):
        self.alpha = alpha
        self.window = window
        self.scores = deque(maxlen=window)     # UNCENSORED MAE in ATR units
        self._stopped = deque(maxlen=window)   # 1/0 realized stop-out flags
        self.n_seen = 0                        # uncensored samples folded in
        self.n_total = 0                       # all closed trades observed

    @property
    def ready(self):
        return len(self.scores) >= MIN_CALIB

    def qhat(self):
        """Conformal (1-alpha) quantile of the uncensored MAE distribution, or
        None until the calibration set is warm."""
        if not self.ready:
            return None
        n = len(self.scores)
        rank = min(1.0, max(0.0, (1.0 - self.alpha) * (n + 1) / n))
        return _quantile_sorted(sorted(self.scores), rank)

    def observe(self, mae_atr, stopped):
        """Fold one closed trade in. `mae_atr` is its maximum adverse excursion
        divided by the ATR at entry; `stopped` marks a stop/liquidation exit
        whose excursion is CENSORED and therefore excluded from the score set."""
        self.n_total += 1
        self._stopped.append(1 if stopped else 0)
        if not stopped and mae_atr is not None:
            self.scores.append(max(0.0, float(mae_atr)))
            self.n_seen += 1

    def stop_mult(self, default_mult, lo, hi):
        """Calibrated stop multiple, clamped to the tunable's own bounds. Falls
        back to the operator's default until the calibrator is ready."""
        q = self.qhat()
        if q is None:
            return default_mult
        return max(lo, min(hi, q))

    def stop_rate(self):
        """Realized fraction of trades that exited via stop/liquidation."""
        if not self._stopped:
            return None
        return sum(self._stopped) / len(self._stopped)

    def stats(self):
        q = self.qhat()
        return {
            "ready": self.ready,
            "alpha": round(self.alpha, 4),
            "n_scores": len(self.scores),
            "n_seen": self.n_seen,
            "n_total": self.n_total,
            "qhat_stop_mult": (round(q, 3) if q is not None else None),
            "stop_rate": (round(self.stop_rate(), 4)
                          if self.stop_rate() is not None else None),
        }

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "window": self.window,
            "scores": list(self.scores),
            "stopped": list(self._stopped),
            "n_seen": self.n_seen,
            "n_total": self.n_total,
        }

    def load_dict(self, d):
        if not d:
            return False
        try:
            self.alpha = float(d.get("alpha", self.alpha))
            w = int(d.get("window", self.window))
            self.window = w
            self.scores = deque((float(s) for s in d.get("scores", [])), maxlen=w)
            self._stopped = deque((int(c) for c in d.get("stopped", [])), maxlen=w)
            self.n_seen = int(d.get("n_seen", len(self.scores)))
            self.n_total = int(d.get("n_total", len(self._stopped)))
            return True
        except (TypeError, ValueError):
            return False
