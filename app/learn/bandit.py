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
    POOL_MAX_OBS = 30         # cap the cross-regime pool's confidence: it is a
                              # COLD-START PRIOR, never stronger than ~30 obs, so
                              # it can't overrule well-sampled local evidence.
    NET_EDGE_N = 25           # in-regime obs needed to trust the sign of the
                              # local mean for the net-edge floor gate.

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

    def decay(self, gamma=0.995, prune_below=0.05):
        """Exponentially forget old evidence so the bandit tracks a
        NON-STATIONARY market. Each cycle every arm's effective sample size,
        its mean (pulled toward 0), and its variance accumulator are multiplied
        by `gamma`. Idle arms therefore forget a stale edge instead of keeping
        a frozen mean at decaying n (the ghost-evolved bug: +8 bps forever
        with no live champion). Live arms are topped back up by update() so
        their mean tracks recent outcomes. Arms whose effective n falls below
        `prune_below` are dropped. The half-life at gamma=0.995 is ~140 cycles
        (~7h at 3-min cycles).
        """
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

    def drop_strategy(self, strategy):
        """Remove every regime arm for a strategy (silent / evicted sleeve)."""
        dead = [k for k in self.arms if k[1] == strategy]
        for k in dead:
            del self.arms[k]
        return len(dead)

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
        # The cross-regime pool is a COLD-START PRIOR, not extra local data.
        # Cap its effective sample size so a strategy that is great elsewhere
        # cannot flip the SIGN of a well-sampled negative in-regime posterior
        # (the 'evolved' bug: −10bps in bear/normal pooled up to +weight).
        n_g_eff = min(n_g, self.POOL_MAX_OBS)
        mu_g, sd_g, _ = self._conjugate(n_g_eff, mean_g, m2_g)
        # Inflate so other regimes are a prior, not extra local data.
        between = (self.PRIOR_STD ** 2) * self.POOL_K / n_g_eff
        prior_sd = math.sqrt(sd_g ** 2 + between)
        return self._conjugate(n, mean, m2, prior_mean=mu_g, prior_std=prior_sd)

    def _net_edge_ok(self, regime, strategy):
        """True unless the strategy has a WELL-SAMPLED, confidently-negative
        in-regime edge. This is the net-edge guard: a strategy that actually
        loses money in THIS regime (raw local mean < 0 over enough trades)
        must not be handed allocation just because a cross-regime pool pulled
        its posterior positive."""
        n, mean, m2 = self.arms.get((regime, strategy), (0, 0.0, 0.0))
        if n < self.NET_EDGE_N:
            return True                      # not enough local evidence to judge
        se = self._obs_std(n, m2) / math.sqrt(n)
        # negative AND statistically distinguishable from zero (~1 SE)
        return not (mean < 0 and mean + se < 0)

    def sample_weights(self, regime, temperature=4.0):
        """Thompson-sample each arm, softmax the draws into weights."""
        draws = {}
        for s in self.strategies:
            mu, sd, n = self._posterior(regime, s)
            g = random.gauss(mu, sd)
            # Pin confirmed in-regime losers to the low end of the draw range so
            # the softmax gives them near-floor weight regardless of pooling.
            if not self._net_edge_ok(regime, s):
                g = min(g, mu * 0.0 - abs(sd))     # push well below zero
            draws[s] = g
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
                # `n`/`n_pool` are EFFECTIVE sample sizes: decay() multiplies
                # them by gamma each cycle to forget stale evidence, so they are
                # floats, not integer counts. Round for display/serialization —
                # 14-digit fractions are meaningless to a human and bloat state.
                "n": round(n, 1),
                "n_pool": round(n_all, 1),
                "mean_bps": round(mu * 1e4, 2) if n_all else None,
                "std_bps": round(sd * 1e4, 2) if n_all else None,
                "raw_mean_bps": round(mean * 1e4, 2) if n else None,
            }
        return out
