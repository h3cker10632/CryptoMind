"""Self-Improvement Loop v2 — multi-algorithm learning stack.

Layers
  1. Signal scoring      : every strategy signal labeled with realized 1h
                           forward return (aligned to direction).
  2. Thompson bandit     : regime-conditioned Bayesian allocation over the
                           strategy ensemble (exploration/exploitation).
  3. Online neural model : TinyMLP trained continually on live features vs
                           forward returns (experience replay, AdaGrad).
  4. Drift detection     : PSI over the feature stream; drift boosts the
                           model learning rate for fast re-adaptation.
  5. RL risk controller  : Q-learning agent adapting global risk scale
                           (queried by the risk manager each tick).
  6. Genetic evolution   : background GA evolving rule parameters on real
                           history; validated champions join the ensemble
                           as the 'evolved' strategy.
"""
import time, threading, asyncio
from collections import deque
from ..config import ALLOC_LOOKBACK, SIGNAL_EVAL_HORIZON_SEC, ALLOC_TEMPERATURE
from .. import db
from ..signals.engine import STRATEGIES
from .online_model import model, committee, build_x, FEAT_NAMES
from .bandit import RegimeBandit
from .drift import detector, page_hinkley
from .rl_risk import agent as rl_agent
from .evolution import evolution

ML_HORIZON_SEC = 1800          # online model label horizon (30 min)
EVOLVE_EVERY_SEC = 1200        # a GA run every 20 min, rotating the universe
WEIGHT_SMOOTH = 0.35           # EMA smoothing of bandit draws (stability)
LOOKUP_GIVE_UP_SEC = 3 * 3600  # abandon unscored signals after this (no train)


class Learner:
    def __init__(self):
        self.strategy_stats = {}
        self.weights = {k: 1.0 / len(STRATEGIES) for k in STRATEGIES}
        self.bandit = RegimeBandit(list(STRATEGIES))
        self.last_run = 0.0
        self.price_history = {}            # product -> [(ts, price)]
        self.pending_ml = deque(maxlen=2000)  # (ts, product, x, pred)
        self.last_evolution_start = 0.0
        self._evo_thread = None
        self.drift_state = {"drifting": False, "worst_feature": None, "psi": 0.0}
        self.current_regime_label = "unknown"
        self.trade_attributions = 0

    # ------------------------------------------------ price memory
    def observe_prices(self, market):
        now = time.time()
        for p in market.tickers:
            px = market.price(p)
            if px:
                h = self.price_history.setdefault(p, [])
                h.append((now, px))
                cutoff = now - 3 * 3600
                while h and h[0][0] < cutoff:
                    h.pop(0)

    def _price_at(self, product, ts, market=None):
        h = self.price_history.get(product, [])
        best = None
        for t, px in h:
            if best is None or abs(t - ts) < abs(best[0] - ts):
                best = (t, px)
        if best and abs(best[0] - ts) < 600:
            return best[1]
        if market is not None:
            closest = None
            for c in market.candles.get(product, []):
                if closest is None or abs(c[0] - ts) < abs(closest[0] - ts):
                    closest = c
            if closest and abs(closest[0] - ts) < 600:
                return closest[4]
        return None

    # ------------------------------------------------ per-tick ML sampling
    def collect_features(self, market, nlp):
        """Record a feature snapshot + model prediction for every product;
        these get labeled with forward returns for online training."""
        from ..data.derivatives import derivatives
        now = time.time()
        for p in market.tickers:
            f = market.features(p)
            if not f:
                continue
            asset_sent, _ = nlp.asset_score(p)
            x = build_x(f, asset_sent, nlp.market_sentiment,
                        derivatives.features(p))
            pred = model.predict(x) if model.n_updates >= 10 else 0.0
            self.pending_ml.append((now, p, x, pred))
            detector.add(x)

    def _train_online_model(self, market):
        """Label matured feature snapshots and run online SGD updates."""
        now = time.time()
        trained = 0
        while self.pending_ml and now - self.pending_ml[0][0] >= ML_HORIZON_SEC:
            ts, p, x, pred = self.pending_ml.popleft()
            p0 = self._price_at(p, ts, market)
            p1 = self._price_at(p, ts + ML_HORIZON_SEC, market) or market.price(p)
            if p0 and p1 and trained < 40:            # cap per cycle
                fwd = p1 / p0 - 1
                # concept-drift signal: how wrong was the recorded prediction vs
                # the realized (scaled) return? Page-Hinkley watches this error
                # stream for a creeping breakdown of the input→return relation.
                if pred is not None:
                    target = max(-1.0, min(1.0, fwd / 0.004))
                    if page_hinkley.add(abs(pred - target)) and model.lr_boost <= 1.05:
                        committee.lr_boost = 2.0
                        db.log_event("learn", "CONCEPT DRIFT (Page-Hinkley): model "
                                     "error broke trend — LR boosted x2 to re-adapt")
                committee.update(x, fwd, pred_at_record=pred)
                trained += 1
        return trained

    # ------------------------------------------------ drift
    def _check_drift(self):
        drifting, worst_f, psi = detector.check(FEAT_NAMES)
        self.drift_state = {"drifting": drifting, "worst_feature": worst_f,
                            "psi": round(psi, 4)}
        if drifting and model.lr_boost <= 1.05:
            # Only boost the LR to re-adapt a model that was ACTUALLY WORKING.
            # Boosting the learning rate of a head that's already below a coin
            # flip just makes it fit the drift faster in the wrong direction
            # (this is exactly what happened: 20.7% dir-acc + PSI≈6 → LR×2.5).
            st = model.stats()
            acc = st["directional_accuracy"]
            if acc is not None and acc > 0.50 and model.n_updates >= 40:
                committee.lr_boost = 3.0               # fast re-adaptation (all members)
                db.log_event("learn", f"DRIFT on '{worst_f}' (PSI={psi:.3f}) "
                                      f"— LR boosted x3 (model acc {acc:.1%})")
            else:
                db.log_event("learn", f"DRIFT on '{worst_f}' (PSI={psi:.3f}) "
                                      f"— LR boost SUPPRESSED (model acc "
                                      f"{'n/a' if acc is None else format(acc,'.1%')} "
                                      f"≤ coin flip; not chasing a broken head)")

    # ------------------------------------------------ background evolution
    def _next_evolution_product(self):
        """Always evolve the least-recently-ATTEMPTED coin (never-attempted
        first, since their timestamp is 0). Attempts count regardless of
        whether the genome was promoted, so a coin that keeps failing OOS
        validation cannot hog the rotation."""
        from ..config import PRODUCTS
        return min(PRODUCTS, key=lambda p: evolution.last_attempt.get(p, 0))

    def maybe_evolve(self, product=None):
        """Kick off a GA run in a worker thread (never blocks the loop).
        With no explicit product, rotates through the whole universe."""
        now = time.time()
        if self._evo_thread and self._evo_thread.is_alive():
            return
        from ..tunables import tv
        if now - self.last_evolution_start < tv("evolve_every_sec"):
            return
        self.last_evolution_start = now
        if product is None:
            product = self._next_evolution_product()
        # record the attempt NOW (any outcome), so rotation always advances
        evolution.last_attempt[product] = now

        def _worker():
            try:
                from ..backtest.engine import fetch_history
                candles = asyncio.run(fetch_history(product))
                if len(candles) < 300:
                    db.log_event("warn", f"Evolution skipped for {product}: "
                                         f"insufficient history ({len(candles)} bars)")
                    return
                rep = evolution.evolve(candles, product=product)
                msg = (f"Evolution finished on {product}: "
                       f"train_fit={rep['train_fitness']} "
                       f"val_fit={rep['validation_fitness']} "
                       f"promoted={rep['promoted']} "
                       f"(champions: {len(evolution.champions)})")
                db.log_event("learn", msg, rep["genome"])
            except Exception as e:
                evolution.status = "error"
                db.log_event("error", f"Evolution failed on {product}: {e}")

        self._evo_thread = threading.Thread(target=_worker, daemon=True)
        self._evo_thread.start()
        db.log_event("learn", f"Genetic evolution started on {product} "
                              f"(pop={evolution.pop_size}, gens={evolution.generations})")

    # ------------------------------------------------ trade-PnL attribution
    def on_trade_closed(self, trade):
        """Learn from REAL closed-trade PnL (fees & slippage included).

        This is the highest-quality learning signal in the system: the
        return is attributed back to every strategy that voted for the
        entry, proportional to its vote strength, and fed to the bandit
        under the regime at entry. Losses directly demote the strategies
        that caused them — including trades that were directionally right
        but net-negative after costs."""
        votes = trade.get("votes") or {}
        if not votes:
            return
        regime = trade.get("regime_at_entry", "unknown")
        entry_notional = trade["qty"] * trade["entry"]
        if entry_notional <= 0:
            return
        net_return = trade["pnl"] / entry_notional     # after fees+slippage
        total_w = sum(abs(v) for v in votes.values()) or 1e-9
        # `hedge` is an attributable sleeve (the market-neutral pair book) even
        # though it is not a directional strategy that votes in the composite —
        # so its realized PnL is scored by the bandit rather than dropped.
        attributable = STRATEGIES if "hedge" in STRATEGIES else (*STRATEGIES, "hedge")
        for strat, v in votes.items():
            if strat not in attributable:
                continue
            share = abs(v) / total_w
            # a strategy that voted long gets the trade's return as-is;
            # one that voted AGAINST the entry gets the inverse credit
            aligned = net_return if v > 0 else -net_return
            # weight trade outcomes over signal outcomes (real money, real
            # costs), and losing trades N-times harder than winners —
            # "every losing trade is a lesson worth 5x more than a winner"
            from ..tunables import tv
            w = tv("trade_weight")
            if aligned < 0:
                w *= tv("loss_lesson_mult")
            self.bandit.update(regime, strat, aligned * share * w)
        self.trade_attributions += 1
        db.log_event("learn",
                     f"Trade attribution: {trade['product']} "
                     f"net={net_return:+.3%} -> "
                     f"{', '.join(f'{k}({v:+.2f})' for k, v in votes.items())} "
                     f"[regime={regime}]")

    # ------------------------------------------------ main learning cycle
    def run(self, market, regime=None):
        now = time.time()
        regime_label = (regime or {}).get("label", "unknown")
        self.current_regime_label = regime_label

        # 1. score matured strategy signals → bandit + stats
        matured = db.unscored_signals(now - SIGNAL_EVAL_HORIZON_SEC)
        n_scored, n_retry, n_abandon = 0, 0, 0
        for s in matured:
            p0 = self._price_at(s["product"], s["ts"], market)
            p1 = self._price_at(s["product"], s["ts"] + SIGNAL_EVAL_HORIZON_SEC, market) \
                 or market.price(s["product"])
            if p0 and p1:
                fwd = (p1 / p0 - 1) * s["direction"]
                db.score_signal(s["rowid"], fwd)   # store GROSS fwd for the UI
                arm_regime = s.get("regime") or regime_label
                # Feed the bandit the NET edge — the same exam the cost gate and
                # real fills face. A signal that's directionally right but can't
                # clear round-trip fees+slippage is NOT a winning arm; scoring it
                # gross is how a structurally-unprofitable sleeve keeps weight.
                from ..tunables import tv
                round_trip = 2 * tv("fee_rate") + 2 * tv("slippage_bps") / 1e4
                net = fwd - round_trip
                self.bandit.update(arm_regime, s["strategy"],
                                   net * s["confidence"])
                n_scored += 1
            elif now - s["ts"] > LOOKUP_GIVE_UP_SEC:
                db.abandon_signal(s["rowid"])
                n_abandon += 1
            else:
                n_retry += 1

        # 2. classic per-strategy stats (for dashboard)
        rows = db.strategy_scores(ALLOC_LOOKBACK)
        per = {}
        for r in rows:
            d = per.setdefault(r["strategy"], {"rets": [], "hits": 0})
            d["rets"].append(r["fwd_return"] * r["confidence"])
            if r["fwd_return"] > 0:
                d["hits"] += 1
        stats = {}
        for name in STRATEGIES:
            d = per.get(name)
            if d and d["rets"]:
                avg = sum(d["rets"]) / len(d["rets"])
                stats[name] = {"n": len(d["rets"]),
                               "avg_aligned_return_bps": round(avg * 1e4, 2),
                               "hit_rate": round(d["hits"] / len(d["rets"]), 3)}
            else:
                stats[name] = {"n": 0, "avg_aligned_return_bps": None,
                               "hit_rate": None}
        self.strategy_stats = stats

        # 2b. forget old evidence — decay bandit posteriors once per cycle so a
        # strategy that stopped working sheds its stale reputation (markets are
        # non-stationary; a 3-week-old win should not weigh like an hour-old one).
        from ..tunables import tv as _tv
        n_pruned = self.bandit.decay(gamma=_tv("bandit_decay_gamma"))

        # 3. Thompson-sampled weights (EMA-smoothed), exploration floor.
        # The 0.04 floor keeps healthy-but-unlucky sleeves alive for exploration
        # — but it must NOT prop up a sleeve we have positive evidence is broken.
        # A confirmed in-regime loser (bandit net-edge gate) or an ML head still
        # at/below a coin flip gets NO floor, so its weight can decay to ~0
        # instead of being pinned in the book at 4%.
        draw = self.bandit.sample_weights(regime_label, temperature=ALLOC_TEMPERATURE)
        base_floor = 0.04
        model_acc = model.stats()["directional_accuracy"]
        ml_broken = (model.n_updates >= 40 and
                     (model_acc is None or model_acc <= 0.50))
        mixed = {}
        for k in STRATEGIES:
            w = WEIGHT_SMOOTH * draw.get(k, 0) + (1 - WEIGHT_SMOOTH) * self.weights.get(k, 0)
            floored = not self.bandit._net_edge_ok(regime_label, k)
            if k == "ml" and ml_broken:
                floored = True
            mixed[k] = max(0.0, w) if floored else max(base_floor, w)
        z = sum(mixed.values()) or 1.0
        self.weights = {k: round(v / z, 4) for k, v in mixed.items()}

        # 4. online model training + drift check
        n_trained = self._train_online_model(market)
        self._check_drift()

        # 5. background genetic evolution
        self.maybe_evolve()

        self.last_run = now
        db.log_event("learn",
                     f"Learning cycle: {n_scored} scored, {n_retry} retry, "
                     f"{n_abandon} abandon, {n_trained} ML updates, "
                     f"regime={regime_label}",
                     self.weights)
        return self.weights

    # ------------------------------------------------ introspection
    def full_stats(self):
        return {
            "weights": self.weights,
            "regime": self.current_regime_label,
            "strategy_stats": self.strategy_stats,
            "bandit_posteriors": self.bandit.table(self.current_regime_label),
            "online_model": model.stats(),
            "drift": {**self.drift_state, **detector.stats(), **page_hinkley.stats()},
            "rl_risk": rl_agent.stats(),
            "trade_attributions": self.trade_attributions,
            "evolution": evolution.stats(),
            "last_run": self.last_run,
        }


learner = Learner()
