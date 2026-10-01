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
import threading

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
    N_IN = 7

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
            # The market-only and evidence-backed advisors remain separate inputs.
            max(-1.0, min(1.0, m.get("llm_lean", 0.0))),
            max(-1.0, min(1.0, m.get("research_lean", 0.0))),
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
        self.update_features(self.features(m), outcome0_won)

    def update_features(self, features: list[float], outcome0_won: int):
        if len(features) != self.N_IN:
            raise ValueError(f"expected {self.N_IN} PM features")
        x = [float(value) for value in features]
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
        self._lock = threading.RLock()
        self.bandit = RegimeBandit(STRATEGIES)
        self.online = _PMOnline()
        self.n_signals_scored = 0
        self.n_trades_learned = 0
        # per-strategy net attribution from closed trades (premium teacher)
        self.attributions = {s: {"n": 0, "net": 0.0} for s in STRATEGIES}
        self.scored_forecast_ids: set[str] = set()

    def weights(self, ttl_hours) -> dict:
        with self._lock:
            return self.bandit.sample_weights(regime_of(ttl_hours))

    # ------------------------- SIGNAL teacher -------------------------
    def _score_signal(self, ttl_hours, votes, outcome0_won, features):
        with self._lock:
            realized = 1.0 if outcome0_won else -1.0
            reg = regime_of(ttl_hours)
            for strategy in STRATEGIES:
                lean = votes.get(strategy, 0.0)
                if abs(lean) < 1e-6:
                    continue
                aligned = lean * realized * 0.002
                self.bandit.update(reg, strategy, aligned)
            self.online.update_features(features, outcome0_won)
            self.n_signals_scored += 1

    def score_resolution(self, market: dict, votes: dict, outcome0_won: int,
                         feature_snapshot: list[float] | None = None):
        """Counterfactual credit: reward each strategy's lean for pointing at the
        outcome that actually won, whether or not we traded this market.

        `votes` are signed toward outcome 0 (the raw leans, not the chosen-side
        votes). Aligned return = lean * realized_direction, where realized is +1
        if outcome 0 won else -1, scaled to the bandit's small-return units.
        """
        features = feature_snapshot or self.online.features(market)
        self._score_signal(market.get("ttl_hours"), votes, outcome0_won, features)

    def score_forecast(self, forecast_id, ttl_hours, votes, outcome0_won,
                       feature_snapshot):
        """Apply one resolved ledger row once, using only its saved inputs."""
        key = str(forecast_id)
        with self._lock:
            if key in self.scored_forecast_ids:
                return False
            self._score_signal(ttl_hours, votes, outcome0_won, feature_snapshot)
            self.scored_forecast_ids.add(key)
            return True

    # ------------------------- TRADE teacher --------------------------
    def on_trade_closed(self, trade: dict):
        """Premium teacher: attribute a closed paper trade's net return to the
        strategies that voted for it. Losers weighted 3x (a bad bet should teach
        harder than a good one confirms), mirroring the crypto attribution."""
        with self._lock:
            ret = trade.get("return_pct", 0.0)
            reg = trade.get("regime_at_entry") or "unknown"
            votes = trade.get("votes") or {}
            weight = 3.0 if ret < 0 else 1.0
            for s, v in votes.items():
                if abs(v) < 1e-6:
                    continue
                aligned = math.copysign(min(abs(ret), 1.0),
                                        v * (1 if ret >= 0 else -1))
                aligned *= 0.01 * weight
                self.bandit.update(reg, s, aligned)
                if s in self.attributions:
                    self.attributions[s]["n"] += 1
                    self.attributions[s]["net"] += ret * (1 if v > 0 else -1)
            self.n_trades_learned += 1

    def decay(self):
        with self._lock:
            self.bandit.decay()

    def capture(self) -> dict:
        with self._lock:
            return self._capture()

    def _capture(self) -> dict:
        return {
            "version": 1,
            "feature_count": self.online.N_IN,
            "strategies": list(self.bandit.strategies),
            "bandit_arms": [[regime, strategy, *values]
                            for (regime, strategy), values
                            in sorted(self.bandit.arms.items())],
            "online": {
                "w": list(self.online.w), "b": self.online.b,
                "lr": self.online.lr, "l2": self.online.l2,
                "n": self.online.n, "mean": list(self.online._mean),
                "var": list(self.online._var), "m2": list(self.online._m2),
                "seen": self.online._seen,
                "log_loss_sum": self.online.log_loss_sum,
            },
            "n_signals_scored": self.n_signals_scored,
            "n_trades_learned": self.n_trades_learned,
            "attributions": {key: dict(value)
                             for key, value in self.attributions.items()},
            "scored_forecast_ids": sorted(self.scored_forecast_ids),
        }

    def restore(self, state: dict | None) -> bool:
        with self._lock:
            return self._restore(state)

    def _restore(self, state: dict | None) -> bool:
        """Restore only a complete state with a matching version and feature shape."""
        if (not isinstance(state, dict) or state.get("version") != 1
                or state.get("feature_count") != self.online.N_IN
                or state.get("strategies") != self.bandit.strategies):
            return False
        try:
            online = state["online"]
            arrays = [list(map(float, online[key]))
                      for key in ("w", "mean", "var", "m2")]
            if any(len(values) != self.online.N_IN for values in arrays):
                return False
            arms = {}
            for regime, strategy, n, mean, m2 in state["bandit_arms"]:
                if strategy not in self.bandit.strategies:
                    return False
                arms[(str(regime), strategy)] = (float(n), float(mean), float(m2))
            attributions = {strategy: {"n": 0, "net": 0.0}
                            for strategy in self.bandit.strategies}
            for strategy, values in (state.get("attributions") or {}).items():
                if strategy in attributions:
                    attributions[strategy] = {
                        "n": int(values.get("n", 0)),
                        "net": float(values.get("net", 0.0)),
                    }
            counters = (int(state.get("n_signals_scored", 0)),
                        int(state.get("n_trades_learned", 0)))
            scored_ids = {str(value) for value in state.get("scored_forecast_ids", [])}
            online_values = {
                "b": float(online["b"]), "lr": float(online["lr"]),
                "l2": float(online["l2"]), "n": int(online["n"]),
                "seen": int(online["seen"]),
                "log_loss_sum": float(online["log_loss_sum"]),
            }
        except (KeyError, TypeError, ValueError):
            return False
        self.bandit.arms = arms
        self.online.w, self.online._mean, self.online._var, self.online._m2 = arrays
        self.online.b = online_values["b"]
        self.online.lr = online_values["lr"]
        self.online.l2 = online_values["l2"]
        self.online.n = online_values["n"]
        self.online._seen = online_values["seen"]
        self.online.log_loss_sum = online_values["log_loss_sum"]
        self.n_signals_scored, self.n_trades_learned = counters
        self.attributions = attributions
        self.scored_forecast_ids = scored_ids
        return True

    # ------------------------------ stats -----------------------------
    def stats(self):
        with self._lock:
            return self._stats()

    def _stats(self):
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
            "starved": self.n_trades_learned == 0 and self.n_signals_scored == 0,
            "bandit_weights": table,
            "online_model": self.online.stats(),
            "attributions": {
                s: {"n": a["n"], "net": round(a["net"], 4)}
                for s, a in self.attributions.items() if a["n"] > 0
            },
        }


learner = PMLearner()
