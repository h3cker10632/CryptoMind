"""Regime-conditioned Thompson-sampling bandit for strategy allocation.

Each (market regime, strategy) arm keeps a running Gaussian posterior over
signal-aligned forward returns. Weights are drawn by Thompson sampling —
naturally balancing exploration vs exploitation, per regime.

Posteriors are conjugate Normal-Normal (usable from n=1) with partial pooling
across regimes so a brand-new regime inherits the strategy's global estimate
instead of a cold prior.
"""
import math, random


class RegimeBandit:
    PRIOR_STD = 0.0015        # prior uncertainty on aligned return (~15 bps)
    POOL_K = 6                # between-regime extra variance (in prior-obs units)
    MIN_OBS_STD = 0.0015      # floor on estimated observation noise

    def __init__(self, strategies):
        self.strategies = list(strategies)
        self.arms = {}        # (regime, strategy) -> (n, mean, M2)

    def update(self, regime, strategy, aligned_return):
        key = (regime, strategy)
        n, mean, m2 = self.arms.get(key, (0, 0.0, 0.0))
        n += 1
        d = aligned_return - mean
        mean += d / n
        m2 += d * (aligned_return - mean)
        self.arms[key] = (n, mean, m2)

    def _obs_std(self, n, m2):
        if n >= 2:
            return max(math.sqrt(m2 / (n - 1)), self.MIN_OBS_STD)
        return self.PRIOR_STD

    def _combine_welford(self, stats):
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

    def _strategy_pool(self, strategy, exclude_regime=None):
        stats = [v for (reg, s), v in self.arms.items()
                 if s == strategy and reg != exclude_regime]
        return self._combine_welford(stats)

    def _conjugate(self, n, mean, m2, prior_mean=0.0, prior_std=None):
        """Normal-Normal posterior. Defined at n=0 and n=1."""
        tau0 = self.PRIOR_STD if prior_std is None else prior_std
        prior_prec = 1.0 / (tau0 ** 2)
        if n <= 0:
            return prior_mean, tau0, 0
        like_prec = n / (self._obs_std(n, m2) ** 2)
        post_prec = prior_prec + like_prec
        post_mean = (prior_prec * prior_mean + like_prec * mean) / post_prec
        return post_mean, math.sqrt(1.0 / post_prec), n

    def _posterior(self, regime, strategy):
        n, mean, m2 = self.arms.get((regime, strategy), (0, 0.0, 0.0))
        n_g, mean_g, m2_g = self._strategy_pool(strategy, exclude_regime=regime)
        if n_g <= 0:
            return self._conjugate(n, mean, m2)
        mu_g, sd_g, _ = self._conjugate(n_g, mean_g, m2_g)
        # Inflate so other regimes are a prior, not extra local data.
        between = (self.PRIOR_STD ** 2) * self.POOL_K / n_g
        prior_sd = math.sqrt(sd_g ** 2 + between)
        return self._conjugate(n, mean, m2, prior_mean=mu_g, prior_std=prior_sd)

    def sample_weights(self, regime, temperature=4.0):
        """Thompson-sample each arm, softmax the draws into weights."""
        draws = {}
        for s in self.strategies:
            mu, sd, n = self._posterior(regime, s)
            draws[s] = random.gauss(mu, sd)
        mx = max(draws.values())
        # *100 → percent-return units so temperature≈4 is actually exploratory
        exps = {s: math.exp(temperature * (v - mx) * 100) for s, v in draws.items()}
        z = sum(exps.values()) or 1.0
        return {s: v / z for s, v in exps.items()}

    def table(self, regime):
        out = {}
        for s in self.strategies:
            n, mean, m2 = self.arms.get((regime, s), (0, 0.0, 0.0))
            n_all, _, _ = self._strategy_pool(s)
            mu, sd, _ = self._posterior(regime, s)
            out[s] = {
                "n": n,
                "n_pool": n_all,
                "mean_bps": round(mu * 1e4, 2) if n_all else None,
                "std_bps": round(sd * 1e4, 2) if n_all else None,
                "raw_mean_bps": round(mean * 1e4, 2) if n else None,
            }
        return out
