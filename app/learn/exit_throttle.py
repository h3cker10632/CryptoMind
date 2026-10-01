"""Exit Mechanism Throttle — learns whether a DISCRETIONARY early-exit
mechanism (pattern_exit, signal-flip) is actually earning its keep, and dials
its trigger bar up or down accordingly. It NEVER touches the hard stop-loss /
take-profit / trailing-stop / kill-switch exits — those stay fixed safety
rules, not an optimization target.

For every discretionary exit, we score the realized counterfactual ("would
holding have done better?") after a horizon — exactly the same self-learning
loop app.learn.exit_advisor already uses for its hold-vs-cut decision, just
applied to a different question (was THIS mechanism's trigger correct?).

Learning is hierarchical — (mechanism, regime, trend_bucket), pooling up to
(mechanism, regime) and then to mechanism-only — so a data-starved fine
bucket borrows strength from coarser evidence instead of sitting at an
uninformed prior for a long time. Same conjugate-Normal + Welford-pooling
technique as app.learn.bandit.RegimeBandit.

With ZERO evidence at every pooling level, throttle_factor() is EXACTLY 1.0
(no behavior change) — a cold-start safety guarantee, not a tuned heuristic.
"""
import math, time
from collections import deque


class ExitThrottle:
    PRIOR_STD = 0.01          # prior uncertainty on the counterfactual edge
    POOL_K = 6                # between-bucket extra variance (prior-obs units)
    MIN_OBS_STD = 0.01        # floor on estimated observation noise
    POOL_MAX_OBS = 30         # cap a pooled prior's effective confidence
    MAX_FACTOR = 1.5          # throttle_factor bounded to [1/MAX_FACTOR, MAX_FACTOR]

    def __init__(self):
        self.arms = {}        # (mechanism, regime, trend_bucket) -> (n, mean, M2)
        self.pending = deque(maxlen=4000)

    # ---------------- learning ----------------
    def record(self, mechanism, regime, trend_bucket, edge):
        """edge > 0 means the exit was VALIDATED (price kept moving against
        the original position after the cut); edge < 0 means it was a false
        alarm (price would have recovered had we held)."""
        key = (mechanism, regime, trend_bucket)
        n, mean, m2 = self.arms.get(key, (0, 0.0, 0.0))
        n += 1
        d = edge - mean
        mean += d / n
        m2 += d * (edge - mean)
        self.arms[key] = (n, mean, m2)

    def decay(self, gamma=0.995, prune_below=0.5):
        """Forget stale evidence so throttling tracks a non-stationary market
        (same scheme as RegimeBandit.decay)."""
        dead = []
        for key, (n, mean, m2) in list(self.arms.items()):
            n2 = n * gamma
            if n2 < prune_below:
                dead.append(key)
                continue
            self.arms[key] = (n2, mean * gamma, m2 * gamma)
        for key in dead:
            del self.arms[key]
        return len(dead)

    # ---------------- posterior / pooling ----------------
    def _obs_std(self, n, m2):
        if n >= 2:
            return max(math.sqrt(m2 / (n - 1)), self.MIN_OBS_STD)
        return self.PRIOR_STD

    @staticmethod
    def _combine_welford(stats):
        n_tot, mean, m2 = 0, 0.0, 0.0
        for n, mu, m2_i in stats:
            if n <= 0:
                continue
            if n_tot == 0:
                n_tot, mean, m2 = n, mu, m2_i
                continue
            n_new = n_tot + n
            delta = mu - mean
            mean = (n_tot * mean + n * mu) / n_new
            m2 = m2 + m2_i + delta * delta * n_tot * n / n_new
            n_tot = n_new
        return n_tot, mean, m2

    def _conjugate(self, n, mean, m2, prior_mean=0.0, prior_std=None):
        tau0 = self.PRIOR_STD if prior_std is None else prior_std
        prior_prec = 1.0 / (tau0 ** 2)
        if n <= 0:
            return prior_mean, tau0, 0
        like_prec = n / (self._obs_std(n, m2) ** 2)
        post_prec = prior_prec + like_prec
        post_mean = (prior_prec * prior_mean + like_prec * mean) / post_prec
        return post_mean, math.sqrt(1.0 / post_prec), n

    def _pool(self, match_fn, exclude_key=None):
        stats = [v for k, v in self.arms.items()
                if match_fn(k) and k != exclude_key]
        return self._combine_welford(stats)

    def _posterior(self, mechanism, regime, trend_bucket):
        key = (mechanism, regime, trend_bucket)
        n, mean, m2 = self.arms.get(key, (0, 0.0, 0.0))
        # pool 1: same (mechanism, regime), other trend buckets
        n_r, mean_r, m2_r = self._pool(
            lambda k: k[0] == mechanism and k[1] == regime, exclude_key=key)
        if n_r > 0:
            n_r_eff = min(n_r, self.POOL_MAX_OBS)
            mu_r, sd_r, _ = self._conjugate(n_r_eff, mean_r, m2_r)
            between = (self.PRIOR_STD ** 2) * self.POOL_K / n_r_eff
            prior_mean, prior_std = mu_r, math.sqrt(sd_r ** 2 + between)
        else:
            # pool 2: same mechanism, any regime (global cold-start prior)
            n_g, mean_g, m2_g = self._pool(lambda k: k[0] == mechanism)
            if n_g > 0:
                n_g_eff = min(n_g, self.POOL_MAX_OBS)
                mu_g, sd_g, _ = self._conjugate(n_g_eff, mean_g, m2_g)
                between = (self.PRIOR_STD ** 2) * self.POOL_K / n_g_eff
                prior_mean, prior_std = mu_g, math.sqrt(sd_g ** 2 + between)
            else:
                prior_mean, prior_std = 0.0, self.PRIOR_STD
        return self._conjugate(n, mean, m2, prior_mean=prior_mean, prior_std=prior_std)

    def throttle_factor(self, mechanism, regime, trend_bucket):
        """>1 = loosen (confirmed net-GOOD here); <1 = tighten (confirmed
        net-HARMFUL); exactly 1.0 with no evidence at any pooling level."""
        mu, sd, _n = self._posterior(mechanism, regime, trend_bucket)
        if sd <= 0:
            return 1.0
        z = max(-3.0, min(3.0, mu / sd))
        return math.exp(math.log(self.MAX_FACTOR) * (z / 3.0))

    # ---------------- counterfactual recording ----------------
    def record_exit(self, mechanism, regime, trend_bucket, product, side,
                    exit_price, ts=None):
        self.pending.append((ts if ts is not None else time.time(), mechanism,
                             regime, trend_bucket, product, side, exit_price))

    def score_pending(self, price_lookup, horizon_sec, now=None):
        """`price_lookup(product, ts) -> price|None`. Scores matured exits'
        realized counterfactual and folds it into the learner. Returns the
        number scored."""
        now = now if now is not None else time.time()
        updated = 0
        remaining = deque(maxlen=self.pending.maxlen)
        while self.pending:
            (ts, mechanism, regime, trend_bucket, product, side,
             exit_price) = self.pending.popleft()
            if now - ts < horizon_sec:
                remaining.append((ts, mechanism, regime, trend_bucket,
                                  product, side, exit_price))
                continue
            future = price_lookup(product, ts + horizon_sec)
            if not future or not exit_price or exit_price <= 0:
                continue
            edge = -side * (future / exit_price - 1)
            self.record(mechanism, regime, trend_bucket, edge)
            updated += 1
        self.pending = remaining
        return updated

    # ---------------- reporting / persistence ----------------
    def stats(self):
        return {
            "arms": {"|".join(map(str, k)): {"n": round(v[0], 1),
                     "mean": round(v[1], 5)} for k, v in self.arms.items()},
            "pending": len(self.pending),
        }

    def capture(self):
        return {
            "arms": {"||".join(map(str, k)): [n, mean, m2]
                    for k, (n, mean, m2) in self.arms.items()},
            "pending": [[ts, mech, reg, tb, p, side, px]
                       for (ts, mech, reg, tb, p, side, px) in self.pending],
        }

    def restore(self, d):
        if not d:
            return
        try:
            self.arms = {tuple(k.split("||")): (float(v[0]), float(v[1]), float(v[2]))
                        for k, v in (d.get("arms") or {}).items()}
            self.pending = deque(
                [(ts, mech, reg, tb, p, side, px)
                 for ts, mech, reg, tb, p, side, px in (d.get("pending") or [])],
                maxlen=self.pending.maxlen)
        except Exception:
            pass


exit_throttle = ExitThrottle()
