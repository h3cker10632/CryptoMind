"""Distribution-shift detection via Population Stability Index (PSI) on the
model's feature stream. On drift: boost online-model learning rate and log it."""
from collections import deque


class DriftDetector:
    def __init__(self, ref_size=240, cur_size=80, bins=8, threshold=0.25):
        self.ref = deque(maxlen=ref_size)
        self.cur = deque(maxlen=cur_size)
        self.bins = bins
        self.threshold = threshold
        self.last_psi = {}
        self.drifting = False
        self.n_drift_events = 0

    def add(self, x):
        self.ref.append(x)
        self.cur.append(x)

    def _psi_feature(self, ref_vals, cur_vals):
        lo, hi = min(ref_vals), max(ref_vals)
        if hi - lo < 1e-12:
            return 0.0
        width = (hi - lo) / self.bins
        psi = 0.0
        n_r, n_c = len(ref_vals), len(cur_vals)
        for b in range(self.bins):
            a, bnd = lo + b * width, lo + (b + 1) * width
            top = (b == self.bins - 1)
            pr = max(1e-4, sum(1 for v in ref_vals if a <= v and (v < bnd or top)) / n_r)
            pc = max(1e-4, sum(1 for v in cur_vals if a <= v and (v < bnd or top)) / n_c)
            psi += (pc - pr) * __import__("math").log(pc / pr)
        return psi

    def check(self, feat_names):
        """Compute PSI per feature; return (drifting, worst_feature, psi)."""
        if len(self.ref) < self.ref.maxlen // 2 or len(self.cur) < self.cur.maxlen:
            return False, None, 0.0
        worst_f, worst = None, 0.0
        psis = {}
        for i, name in enumerate(feat_names):
            rv = [x[i] for x in self.ref]
            cv = [x[i] for x in self.cur]
            p = self._psi_feature(rv, cv)
            psis[name] = round(p, 4)
            if p > worst:
                worst, worst_f = p, name
        self.last_psi = psis
        was = self.drifting
        self.drifting = worst > self.threshold
        if self.drifting and not was:
            self.n_drift_events += 1
        return self.drifting, worst_f, worst

    def stats(self):
        return {"drifting": self.drifting,
                "n_drift_events": self.n_drift_events,
                "psi": self.last_psi,
                "ref_samples": len(self.ref), "cur_samples": len(self.cur)}


class PageHinkley:
    """Page-Hinkley test for CONCEPT drift on the model's prediction error.

    PSI (above) catches when the INPUT distribution shifts; Page-Hinkley catches
    when the INPUT→RETURN RELATIONSHIP breaks — i.e. the model's error creeps up
    even though inputs look normal. That is the more dangerous, harder-to-see
    kind of drift. It tracks the cumulative deviation of the error stream from
    its running mean minus a tolerance `delta`; when that cumulative excess
    exceeds `lambda_`, concept drift is flagged (and the running stats reset).
    """
    def __init__(self, delta=0.005, lambda_=0.5, alpha=0.999):
        self.delta = delta          # allowed magnitude of change (slack)
        self.lambda_ = lambda_      # detection threshold
        self.alpha = alpha          # forgetting factor on the running mean
        self.mean = 0.0
        self.n = 0
        self.cum = 0.0              # cumulative (x - mean - delta)
        self.min_cum = 0.0
        self.drifting = False
        self.n_events = 0

    def add(self, error):
        """Feed one prediction error (e.g. |pred - realized|). Returns True on
        a fresh concept-drift detection."""
        self.n += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * error \
            if self.n > 1 else error
        self.cum = self.alpha * self.cum + (error - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        detected = (self.cum - self.min_cum) > self.lambda_
        self.drifting = detected
        if detected:
            self.n_events += 1
            self.cum = 0.0          # reset after a detection
            self.min_cum = 0.0
        return detected

    def stats(self):
        return {"concept_drifting": self.drifting,
                "concept_drift_events": self.n_events,
                "ph_stat": round(self.cum - self.min_cum, 4),
                "n": self.n}


page_hinkley = PageHinkley()


detector = DriftDetector()
