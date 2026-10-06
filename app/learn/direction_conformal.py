"""Conformal prediction SET for the long-vs-short direction gate.

The multi-timeframe veto currently fires on a HARD-CODED constant: veto a short
when mtf_align >= 0.75 (or a long when <= -0.75). That number is a guess with no
coverage guarantee — it can veto trades that would have worked and wave through
ones that won't.

This applies CONFORMAL CLASSIFICATION (prediction sets) — the same calibration
family used for the online model — to the direction decision:

  * label space = {+1, -1}: which SIDE would have won given the higher-timeframe
    context. From a closed trade, favored = side * sign(pnl) (a long that won ->
    long favored; a long that lost -> short was favored, and vice-versa).
  * feature = mtf_align at entry.
  * p(y | align) is a smoothed (kernel) estimate over the calibration set.
  * nonconformity of a calibration point = 1 - p(favored | align).
  * qhat = the (1-alpha) quantile of those scores (finite-sample corrected).
  * PREDICTION SET for a new align = { y : p(y | align) >= 1 - qhat }.

Veto rule: veto the proposed direction d ONLY when it is NOT in the calibrated
prediction set while the OPPOSITE side IS — i.e. the data, at the target
coverage, confidently says the other side is favored. An ambiguous set (both
sides plausible) never vetoes. Below a minimum calibration size the gate abstains
so the caller keeps its legacy hard-threshold behaviour unchanged.

Pure-Python, small, JSON-serialisable state -> rides the normal snapshot.
"""
import math
from collections import deque

DIR_ALPHA = 0.10          # target miscoverage of the prediction set
DIR_WINDOW = 400          # rolling calibration set
DIR_MIN_CALIB = 40        # abstain until this many labelled points exist
DIR_BANDWIDTH = 0.30      # kernel width on the mtf_align axis (range -1..1)


def _quantile_sorted(sorted_vals, q):
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


class DirectionConformalGate:
    def __init__(self, alpha=DIR_ALPHA, window=DIR_WINDOW, bandwidth=DIR_BANDWIDTH):
        self.alpha = alpha
        self.window = window
        self.h = bandwidth
        # calibration points: (align, favored) with favored in {+1, -1}
        self.points = deque(maxlen=window)
        self.n_seen = 0

    @property
    def ready(self):
        return len(self.points) >= DIR_MIN_CALIB

    def observe(self, align, favored):
        """Fold one labelled outcome in. `favored` must be +1 or -1."""
        if favored not in (1, -1):
            return
        try:
            a = max(-1.0, min(1.0, float(align)))
        except (TypeError, ValueError):
            return
        self.points.append((a, favored))
        self.n_seen += 1

    def observe_trade(self, trade):
        """Derive (align, favored) from a closed trade and fold it in."""
        align = trade.get("mtf_at_entry")
        if align is None:
            return
        side = 1 if trade.get("side", 1) > 0 else -1
        pnl = trade.get("pnl", 0.0)
        if pnl == 0:
            return
        favored = side if pnl > 0 else -side
        self.observe(align, favored)

    def _p_long(self, align):
        """Kernel (Nadaraya-Watson) estimate of P(long favored | align), with
        Laplace smoothing so a sparse neighbourhood can't saturate to 0/1."""
        num = 0.5     # pseudo-count for +1
        den = 1.0     # pseudo-counts total (0.5 each class)
        for a, fav in self.points:
            w = math.exp(-((align - a) / self.h) ** 2)
            den += w
            if fav == 1:
                num += w
        return num / den

    def _p(self, align, y):
        pl = self._p_long(align)
        return pl if y == 1 else (1.0 - pl)

    def _qhat(self):
        # O(n^2) over the calibration set, and it only changes when a point is
        # added, so cache it: veto() runs per coin per tick (and per bar in the
        # learner ablation), recomputing it every call cost ~1s/tick live.
        key = (self.n_seen, len(self.points), self.points[-1] if self.points else None,
               self.alpha, self.h)
        cached = getattr(self, "_qhat_cache", None)
        if cached and cached[0] == key:
            return cached[1]
        # nonconformity of each calibration point under the all-data estimate
        scores = [1.0 - self._p(a, fav) for a, fav in self.points]
        scores.sort()
        n = len(scores)
        rank = min(1.0, max(0.0, (1.0 - self.alpha) * (n + 1) / n))
        q = _quantile_sorted(scores, rank)
        self._qhat_cache = (key, q)
        return q

    def prediction_set(self, align):
        """Return the calibrated set of plausible favored sides at this align."""
        if not self.ready:
            return {1, -1}          # abstain: everything plausible
        try:
            a = max(-1.0, min(1.0, float(align)))
        except (TypeError, ValueError):
            return {1, -1}
        thresh = 1.0 - self._qhat()
        s = set()
        for y in (1, -1):
            if self._p(a, y) >= thresh:
                s.add(y)
        return s or {1, -1}         # never emit an empty (uninformative) set

    def veto(self, direction, align):
        """Return (veto, reason). Veto ONLY when the proposed direction is
        confidently excluded (not in the set) and the opposite side is in it.
        Abstains (no veto) until calibrated -> caller falls back to its legacy
        hard-threshold check."""
        if not self.ready:
            return False, ""
        d = 1 if direction > 0 else -1
        pset = self.prediction_set(align)
        if d not in pset and (-d) in pset:
            side = "short" if d < 0 else "long"
            return True, (f"{side} excluded by conformal direction set "
                          f"(align={align:+.2f}, set={sorted(pset)})")
        return False, ""

    def stats(self):
        return {
            "ready": self.ready,
            "alpha": round(self.alpha, 4),
            "n_points": len(self.points),
            "n_seen": self.n_seen,
            "qhat": round(self._qhat(), 4) if self.ready else None,
        }

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "window": self.window,
            "bandwidth": self.h,
            "points": [[a, f] for a, f in self.points],
            "n_seen": self.n_seen,
        }

    def load_dict(self, d):
        if not d:
            return False
        try:
            self.alpha = float(d.get("alpha", self.alpha))
            self.window = int(d.get("window", self.window))
            self.h = float(d.get("bandwidth", self.h))
            self.points = deque(
                ((float(a), int(f)) for a, f in d.get("points", [])),
                maxlen=self.window)
            self.n_seen = int(d.get("n_seen", len(self.points)))
            return True
        except (TypeError, ValueError):
            return False
