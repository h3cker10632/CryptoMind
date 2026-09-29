"""The Polymarket learning core — a SEPARATE instance of the same machinery the
crypto book learns with, fed prediction-market-shaped features.

Per the operator's decision ("standalone execution, but reuse the bandit/online-
model learner"), this deliberately does NOT touch the crypto `learner`. It keeps
its own books so a Polymarket outcome can never mis-credit a crypto strategy, and
vice-versa. What it REUSES is the actual learning components:

  * RegimeBandit  — the exact Thompson-sampling bandit class from app.learn.bandit,
    instantiated over the Polymarket STRATEGIES and conditioned on a
    time-to-resolution regime bucket. `weights()` samples the current per-strategy
    trust that `signals.evaluate` uses; resolutions feed `update()`.
  * a tiny online logistic model — mirrors the online_model.TinyMLP design (SGD +
    running feature standardization) but with PM inputs (price, spread, momentum,
    time-to-resolution, liquidity) predicting P(outcome 0 wins). It learns only
    from REAL resolutions, so it is honestly starved until markets settle.

Two teachers, exactly like crypto:
  * SIGNAL teacher  — every resolved market scores each strategy's lean against
    the realized outcome (aligned return), feeding the bandit even for markets we
    didn't trade (counterfactual credit).
  * TRADE teacher (premium) — a closed paper trade attributes its net return to
    the strategies that voted for it; losers are weighted more heavily.
"""
from __future__ import annotations

import math

from ...learn.bandit import RegimeBandit
from .signals import STRATEGIES


def regime_of(ttl_hours) -> str:
    """Bucket a market by time-to-resolution — the closest PM analogue to the
    crypto volatility regime (near-dated markets behave very differently from
    months-out ones)."""
    if ttl_hours is None:
        return "unknown"
    if ttl_hours <= 24:
        return "<1d"
    if ttl_hours <= 24 * 7:
        return "1-7d"
    if ttl_hours <= 24 * 30:
        return "1-4w"
    return ">1mo"


class _PMOnline:
    """Minimal online logistic regressor: P(outcome 0 wins) from PM features.

    Same spirit as online_model.TinyMLP (running mean/var standardization + SGD),
    trimmed to a linear model because the feature set is small and we want it to
    stay well-calibrated on few samples.
    """
    N_IN = 6

    def __init__(self, lr=0.05, l2=1e-4):
        self.w = [0.0] * self.N_IN
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.n = 0
        self._mean = [0.0] * self.N_IN
        self._var = [1.0] * self.N_IN
        self._m2 = [0.0] * self.N_IN
        self._seen = 0
        self.log_loss_sum = 0.0

    @staticmethod
    def features(m: dict) -> list[float]:
        ttl = m.get("ttl_hours")
        ttl_log = math.log1p(max(0.0, ttl)) if ttl is not None else 0.0
        liq = m.get("liquidity", 0.0)
        return [
            m["prices"][0],
            m.get("spread", 0.0),
            m.get("mom_1d", 0.0),
            ttl_log,
            math.log1p(max(0.0, liq)),
            # LLM advisor lean (0 unless the LLM sleeve is on) — this is how the
            # LLM "teaches" the Polymarket ML: it learns from resolutions whether
            # the LLM's read on a market is worth anything.
            max(-1.0, min(1.0, m.get("llm_lean", 0.0))),
        ]

    def _standardize(self, x):
        return [(x[i] - self._mean[i]) / math.sqrt(self._var[i] + 1e-9)
                for i in range(self.N_IN)]

    def _observe(self, x):
        self._seen += 1
        for i in range(self.N_IN):
            d = x[i] - self._mean[i]
            self._mean[i] += d / self._seen
            self._m2[i] += d * (x[i] - self._mean[i])
            if self._seen >= 2:
                self._var[i] = self._m2[i] / (self._seen - 1)

    def predict(self, m: dict) -> float:
        z = self._standardize(self.features(m))
        s = self.b + sum(self.w[i] * z[i] for i in range(self.N_IN))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))

    def update(self, m: dict, outcome0_won: int):
        x = self.features(m)
        self._observe(x)
        z = self._standardize(x)
        s = self.b + sum(self.w[i] * z[i] for i in range(self.N_IN))
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))
        y = 1.0 if outcome0_won else 0.0
        self.log_loss_sum += -(y * math.log(p + 1e-9)
                               + (1 - y) * math.log(1 - p + 1e-9))
        g = p - y
        for i in range(self.N_IN):
            self.w[i] -= self.lr * (g * z[i] + self.l2 * self.w[i])
        self.b -= self.lr * g
        self.n += 1

    def stats(self):
        return {
            "n_updates": self.n,
            "avg_log_loss": round(self.log_loss_sum / self.n, 4) if self.n else None,
            "warmed_up": self.n >= 20,
        }


class PMLearner:
    def __init__(self):
        self.bandit = RegimeBandit(STRATEGIES)
        self.online = _PMOnline()
        self.n_signals_scored = 0
        self.n_trades_learned = 0
        # per-strategy net attribution from closed trades (premium teacher)
        self.attributions = {s: {"n": 0, "net": 0.0} for s in STRATEGIES}

    def weights(self, ttl_hours) -> dict:
        return self.bandit.sample_weights(regime_of(ttl_hours))

    # ------------------------- SIGNAL teacher -------------------------
    def score_resolution(self, market: dict, votes: dict, outcome0_won: int):
        """Counterfactual credit: reward each strategy's lean for pointing at the
        outcome that actually won, whether or not we traded this market.

        `votes` are signed toward outcome 0 (the raw leans, not the chosen-side
        votes). Aligned return = lean * realized_direction, where realized is +1
        if outcome 0 won else -1, scaled to the bandit's small-return units.
        """
        realized = 1.0 if outcome0_won else -1.0
        reg = regime_of(market.get("ttl_hours"))
        for s in STRATEGIES:
            lean = votes.get(s, 0.0)
            if abs(lean) < 1e-6:
                continue
            aligned = lean * realized * 0.002    # scale into ~return units
            self.bandit.update(reg, s, aligned)
        self.online.update(market, outcome0_won)
        self.n_signals_scored += 1

    # ------------------------- TRADE teacher --------------------------
    def on_trade_closed(self, trade: dict):
        """Premium teacher: attribute a closed paper trade's net return to the
        strategies that voted for it. Losers weighted 3x (a bad bet should teach
        harder than a good one confirms), mirroring the crypto attribution."""
        ret = trade.get("return_pct", 0.0)
        reg = trade.get("regime_at_entry") or "unknown"
        votes = trade.get("votes") or {}
        weight = 3.0 if ret < 0 else 1.0
        for s, v in votes.items():
            if abs(v) < 1e-6:
                continue
            # v is signed toward the traded side; a winning trade with v>0 credits
            # the strategy positively.
            aligned = math.copysign(min(abs(ret), 1.0), v * (1 if ret >= 0 else -1))
            aligned *= 0.01 * weight
            self.bandit.update(reg, s, aligned)
            if s in self.attributions:
                self.attributions[s]["n"] += 1
                self.attributions[s]["net"] += ret * (1 if v > 0 else -1)
        self.n_trades_learned += 1

    def decay(self):
        self.bandit.decay()

    # ------------------------------ stats -----------------------------
    def stats(self):
        regimes = ("<1d", "1-7d", "1-4w", ">1mo", "unknown")
        table = {}
        for reg in regimes:
            row = self.bandit.table(reg)
            # only surface a regime that has actually accumulated evidence
            if row and any((v.get("n_pool") or 0) > 0 for v in row.values()):
                table[reg] = row
        return {
            "n_signals_scored": self.n_signals_scored,
            "n_trades_learned": self.n_trades_learned,
            "starved": self.n_trades_learned == 0,
            "bandit_weights": table,
            "online_model": self.online.stats(),
            "attributions": {
                s: {"n": a["n"], "net": round(a["net"], 4)}
                for s, a in self.attributions.items() if a["n"] > 0
            },
        }


learner = PMLearner()
