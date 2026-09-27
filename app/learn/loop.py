"""Self-Improvement Loop v2 — multi-algorithm learning stack.

Layers
  1. Signal scoring      : every strategy signal labeled with realized 1h
                           GROSS forward return (dashboard only). The bandit
                           is NOT fed these labels — 1h net-of-taker is an
                           exam no 5m sleeve can pass, and it drowned fills.
  2. Thompson bandit     : regime-conditioned Bayesian allocation. Updates
                           come from closed-trade net PnL (on_trade_closed).
                           Idle means decay toward 0; silent sleeves (no
                           evolved champion, ML ≤ coin-flip) get weight 0.
  3. Online neural model : TinyMLP committee; a head stuck ≤ coin-flip is
                           RESET rather than kept fitting the wrong mapping.
  4. Drift detection     : PSI over the feature stream; drift boosts the
                           model learning rate only if the head is working.
  5. RL risk controller  : Q-learning agent adapting global risk scale
                           (queried by the risk manager each tick).
  6. Genetic evolution   : background GA; only a walk-forward champion
                           votes live. An empty champion book is not an arm.
"""
import time, threading, asyncio
from collections import deque
from ..config import ALLOC_LOOKBACK, SIGNAL_EVAL_HORIZON_SEC, ALLOC_TEMPERATURE
from .. import db
from ..signals.engine import STRATEGIES
from .online_model import model, committee, build_x, FEAT_NAMES, N_IN
from .bandit import RegimeBandit
from .drift import detector, page_hinkley
from .rl_risk import agent as rl_agent
from .evolution import evolution

ML_HORIZON_SEC = 1800          # online model label horizon (30 min), primary head
# Auxiliary multi-horizon labels (Phase 2). Each entry is (aux_head_index,
# horizon_seconds): a FAST head that labels in minutes for quick feedback, and a
# SLOW head that carries longer-context structure. These train the aux heads in
# online_model (shared hidden layer) — the primary 30-min head is unchanged.
# Kept within the 3h price-history / candle window so labels can be resolved.
AUX_HORIZONS = [(0, 300), (1, 7200)]   # head 0 = 5 min, head 1 = 2 h
SKIP_EVAL_HORIZON_SEC = 1800   # Phase 3: counterfactual horizon for skipped
                               # conviction entries (30 min — one swing leg)
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
        # Phase 2: matured-at-other-horizons queue for the auxiliary heads.
        # (ts, product, x, head_index, horizon_sec). Kept separate from
        # pending_ml so the primary-head training path + its persistence schema
        # are completely untouched.
        self.pending_aux = deque(maxlen=6000)
        # Phase 3: COUNTERFACTUAL credit for actionable conviction signals the
        # book was gated out of (risk/guardian/funding/notional). Each entry is
        # (ts, product, direction, entry_price, regime, votes) — after the
        # horizon we compute the NET-of-cost return the trade would have made
        # and teach the bandit's voting strategies, exactly like a real fill
        # but at a discounted weight. This directly attacks data starvation:
        # every throttled-but-wanted trade becomes a learning sample.
        self.pending_skips = deque(maxlen=4000)
        self.skip_attributions = 0
        self.last_evolution_start = 0.0
        self._evo_thread = None
        self.drift_state = {"drifting": False, "worst_feature": None, "psi": 0.0}
        self.current_regime_label = "unknown"
        self.trade_attributions = 0
        self.last_cycle = {}
        self._ml_resets = 0

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
            # Phase 2: same snapshot also queued for each auxiliary horizon.
            for head, hz in AUX_HORIZONS:
                self.pending_aux.append((now, p, x, head, hz))
            detector.add(x)

    def _train_online_model(self, market):
        """Label matured feature snapshots and run online SGD updates."""
        now = time.time()
        trained = 0
        dropped_dim = 0
        while self.pending_ml and now - self.pending_ml[0][0] >= ML_HORIZON_SEC:
            ts, p, x, pred = self.pending_ml.popleft()
            # Snapshots taken before a feature was added (e.g. 17-d before
            # mtf_align) must not train an N_IN net — TinyMLP indexes x[i]
            # against W1 and IndexErrors the whole orchestrator tick.
            if not isinstance(x, (list, tuple)) or len(x) != N_IN:
                dropped_dim += 1
                continue
            p0 = self._price_at(p, ts, market)
            p1 = self._price_at(p, ts + ML_HORIZON_SEC, market) or market.price(p)
            if p0 and p1 and trained < 40:            # cap per cycle
                fwd = p1 / p0 - 1
                # concept-drift signal: how wrong was the recorded prediction vs
                # the realized (scaled) return? Page-Hinkley watches this error
                # stream for a creeping breakdown of the input→return relation.
                # Never boost a head that's already at/below a coin flip.
                if pred is not None:
                    target = max(-1.0, min(1.0, fwd / 0.004))
                    ph_hit = page_hinkley.add(abs(pred - target))
                    acc = model.stats()["directional_accuracy"]
                    working = (acc is not None and acc > 0.50 and model.n_updates >= 40)
                    if ph_hit and model.lr_boost <= 1.05 and working:
                        committee.lr_boost = 2.0
                        db.log_event("learn", "CONCEPT DRIFT (Page-Hinkley): model "
                                     "error broke trend — LR boosted x2 to re-adapt")
                # CONFORMAL CALIBRATION: fold this matured outcome into the
                # calibration set BEFORE the update, so the nonconformity score
                # is measured against the band the model actually emitted when it
                # made this prediction (inductive/split-conformal ordering).
                committee.observe_outcome(x, fwd)
                # run() resets a broken head BEFORE this, so samples land on a
                # fresh net. Direct callers (tests) still train — freezing is
                # the reset, not a silent skip that would drain the queue.
                committee.update(x, fwd, pred_at_record=pred)
                trained += 1
        if dropped_dim:
            db.log_event("learn",
                         f"Dropped {dropped_dim} pending ML sample(s) with "
                         f"stale feature dim (need {N_IN})")
        self._train_aux_heads(market)
        return trained

    def _train_aux_heads(self, market):
        """Label matured auxiliary-horizon snapshots and train the aux heads.

        Each entry matures at its own horizon (the deque is roughly time-ordered
        since horizons are pushed together, but a shorter horizon queued later
        can mature first, so we scan rather than only peek the head). The aux
        heads only shape the shared hidden layer — they never touch the primary
        head, replay buffer or n_updates.
        """
        now = time.time()
        trained = 0
        remaining = deque(maxlen=self.pending_aux.maxlen)
        while self.pending_aux:
            ts, p, x, head, hz = self.pending_aux.popleft()
            if now - ts < hz:
                remaining.append((ts, p, x, head, hz))
                continue
            # give up on samples too old to price (past the history window)
            if now - ts > hz + LOOKUP_GIVE_UP_SEC:
                continue
            if not isinstance(x, (list, tuple)) or len(x) != N_IN:
                continue
            if trained >= 80:                    # cap per cycle (2 heads)
                remaining.append((ts, p, x, head, hz))
                continue
            p0 = self._price_at(p, ts, market)
            p1 = self._price_at(p, ts + hz, market)
            if p0 and p1:
                committee.update_aux(x, p1 / p0 - 1, head)
                trained += 1
            # if price can't be resolved yet, keep it and retry next cycle
            elif now - ts < hz + LOOKUP_GIVE_UP_SEC:
                remaining.append((ts, p, x, head, hz))
        self.pending_aux = remaining
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
                gran = int(tv("ga_granularity"))
                chunks = int(tv("ga_history_chunks"))
                # CROSS-SECTIONAL mode: pool one genome across the whole universe
                # so a low-frequency edge is validated by breadth, not by
                # over-trading a single product.
                if tv("ga_cross_sectional") > 0:
                    from ..config import PRODUCTS
                    basket = PRODUCTS[:int(tv("ga_universe_basket"))]
                    cmap = {}
                    for p in basket:
                        c = asyncio.run(fetch_history(p, granularity=gran, chunks=chunks))
                        if len(c) >= 300:
                            cmap[p] = c
                    if len(cmap) < 2:
                        db.log_event("warn", "Cross-sectional evolution skipped: "
                                             "fewer than 2 products have enough history")
                        return
                    rep = evolution.evolve_universe(cmap)
                    db.log_event("learn",
                        f"Cross-sectional evolution finished on {len(cmap)} products: "
                        f"best_dsr={rep.get('best_observed_dsr')} "
                        f"pooled_sharpe={rep.get('best_observed_pooled_sharpe')} "
                        f"promoted={rep.get('promoted')} "
                        f"portfolio={rep.get('portfolio_size', 0)} "
                        f"gate_fail={rep.get('gate_fail_breakdown')}", rep.get("genome"))
                    return
                candles = asyncio.run(fetch_history(
                    product, granularity=gran, chunks=chunks))
                if len(candles) < 300:
                    db.log_event("warn", f"Evolution skipped for {product}: "
                                         f"insufficient history ({len(candles)} bars)")
                    return
                rep = evolution.evolve(candles, product=product)
                wf = rep.get("walk_forward") or {}
                msg = (f"Evolution finished on {product}: "
                       f"train_fit={rep.get('train_fitness')} "
                       f"pooled_oos_sharpe={rep.get('pooled_oos_sharpe')} "
                       f"oos_windows_positive={wf.get('frac_positive')} "
                       f"promoted={rep.get('promoted')} "
                       f"portfolio={rep.get('portfolio_size', 0)} "
                       f"(champions: {len(evolution.champions)})")
                db.log_event("learn", msg, rep.get("genome"))
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
        # DIRECTION LEARNER: always learn the realized (regime, direction) edge,
        # even for trades with no vote record (e.g. hedge legs) — this is what
        # teaches the system which SIDE has actually paid per regime.
        try:
            from .direction import direction_learner
            direction_learner.on_trade_closed(trade)
        except Exception:
            pass
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

    # ------------------------------------------------ counterfactual (Phase 3)
    def on_entry_skipped(self, product, direction, price, regime, votes):
        """Record a COUNTERFACTUAL: an actionable conviction signal the book was
        gated out of. `_score_skips` later credits the bandit with the net-of-
        cost return the trade WOULD have made. No-op when disabled or when the
        candidate has no attributable votes / a bad price (nothing to learn).
        """
        from ..tunables import tv
        if tv("skip_learn_weight") <= 0:
            return
        if not votes or not price or price <= 0:
            return
        # only keep votes for arms the bandit actually allocates to
        kept = {k: v for k, v in votes.items() if k in STRATEGIES}
        if not kept:
            return
        self.pending_skips.append(
            (time.time(), product, int(direction), float(price),
             regime or "unknown", kept))

    def _round_trip_cost(self):
        """Fractional round-trip cost (both fills): 2x(fee + slippage). Matches
        the cost the paper broker actually charges, so the counterfactual return
        is comparable to a real closed trade's net PnL."""
        from ..tunables import tv
        return 2.0 * (tv("fee_rate") + tv("slippage_bps") / 1e4)

    def _score_skips(self, market):
        """Mature skipped-entry counterfactuals and teach the bandit the net-of-
        cost return each would have produced. Discounted by skip_learn_weight
        and, like real fills, losers teach loss_lesson_mult times harder."""
        from ..tunables import tv
        w0 = tv("skip_learn_weight")
        if w0 <= 0:
            self.pending_skips.clear()
            return 0
        now = time.time()
        cost = self._round_trip_cost()
        loss_mult = tv("loss_lesson_mult")
        scored = 0
        remaining = deque(maxlen=self.pending_skips.maxlen)
        while self.pending_skips:
            ts, p, direction, p0, regime, votes = self.pending_skips.popleft()
            if now - ts < SKIP_EVAL_HORIZON_SEC:
                remaining.append((ts, p, direction, p0, regime, votes))
                continue
            if now - ts > SKIP_EVAL_HORIZON_SEC + LOOKUP_GIVE_UP_SEC:
                continue                       # too old to price — give up
            p1 = self._price_at(p, ts + SKIP_EVAL_HORIZON_SEC, market) \
                or market.price(p)
            if not (p0 and p1):
                remaining.append((ts, p, direction, p0, regime, votes))
                continue
            # net-of-cost directional return the skipped trade would have made
            net = (p1 / p0 - 1) * direction - cost
            total_w = sum(abs(v) for v in votes.values()) or 1e-9
            for strat, v in votes.items():
                share = abs(v) / total_w
                aligned = net if v > 0 else -net
                w = w0 * (loss_mult if aligned < 0 else 1.0)
                self.bandit.update(regime, strat, aligned * share * w)
            self.skip_attributions += 1
            scored += 1
        self.pending_skips = remaining
        if scored:
            db.log_event("learn",
                         f"Counterfactual credit: {scored} skipped entr"
                         f"{'y' if scored == 1 else 'ies'} scored net-of-cost")
        return scored

    def _evolved_live(self):
        """True only when a walk-forward champion (or portfolio) is actually
        voting. An empty book is not an arm — leftover +bps from a dead genome
        must not keep 50% of the allocation."""
        if evolution.champions:
            return True
        return any(evolution.champion_portfolios.values())

    def _ml_is_broken(self):
        """Measured directional accuracy at/below a coin flip after warmup.
        Unmeasured (acc is None) is silent for allocation but not a reset —
        we still need samples to get a reading. We also require a minimum count
        of SCORED (non-abstention) predictions before declaring the head broken:
        warmup/just-reset preds are abstentions now, so a near-empty acc_window
        must not trigger a reset that would wipe a head before it has been fairly
        measured (this was the ml-sleeve doom loop)."""
        st = model.stats()
        acc = st["directional_accuracy"]
        n_scored = st.get("acc_samples", 0)
        return (model.n_updates >= 40 and acc is not None
                and n_scored >= 20 and acc <= 0.50)

    def _silent_sleeves(self):
        silent = set()
        if not self._evolved_live():
            silent.add("evolved")
        # ml votes 0 until acc > 50%; don't give it a 4% floor in the meantime
        acc = model.stats()["directional_accuracy"]
        if acc is None or acc <= 0.50:
            silent.add("ml")
        return silent

    def _maybe_reset_broken_ml(self):
        """A head stuck at/below a coin flip is not 'warmed up' — it has fitted
        the wrong mapping. Reset the committee so the next samples train a
        fresh net instead of digging 26% accuracy in further. After reset,
        n=0 so we don't thrash until the new head is measured again."""
        if not self._ml_is_broken():
            return False
        n_before = model.n_updates
        acc_before = model.stats()["directional_accuracy"]
        committee.reset()
        page_hinkley.reset_running()
        self.bandit.drop_strategy("ml")
        self._ml_resets += 1
        db.log_event("learn",
                     f"ML committee RESET (was n={n_before}, "
                     f"dir-acc={'n/a' if acc_before is None else format(acc_before, '.1%')} "
                     f"≤ coin flip; not fitting a broken head) "
                     f"[resets={self._ml_resets}]")
        return True

    def _allocate(self, regime_label, silent):
        """Thompson mix with a hard-zero on silent sleeves (no 4% floor, no
        EMA carry from last cycle's ghost weight). Confirmed in-regime losers
        among LIVE sleeves also lose the exploration floor."""
        for k in silent:
            self.bandit.drop_strategy(k)
        draw = self.bandit.sample_weights(regime_label, temperature=ALLOC_TEMPERATURE)
        base_floor = 0.04
        mixed = {}
        for k in STRATEGIES:
            if k in silent:
                mixed[k] = 0.0
                continue
            w = (WEIGHT_SMOOTH * draw.get(k, 0)
                 + (1 - WEIGHT_SMOOTH) * self.weights.get(k, 0))
            floored = not self.bandit._net_edge_ok(regime_label, k)
            mixed[k] = max(0.0, w) if floored else max(base_floor, w)
        z = sum(mixed.values()) or 1.0
        return {k: round(v / z, 4) for k, v in mixed.items()}

    def _cycle_quality(self, regime_label, silent):
        """Per-cycle snapshot so we can see idle |mean| shrink (decay toward 0)
        and ghost weight leave the book. Improving = live |mean_bps| falling
        while idle, and silent sleeves staying at weight 0."""
        table = self.bandit.table(regime_label)
        live_abs = [abs(v["mean_bps"]) for k, v in table.items()
                    if k not in silent and v.get("mean_bps") is not None]
        mean_abs = round(sum(live_abs) / len(live_abs), 2) if live_abs else None
        prev = self.last_cycle or {}
        delta = None
        if mean_abs is not None and prev.get("mean_abs_bps") is not None:
            delta = round(mean_abs - prev["mean_abs_bps"], 2)
        return {
            "silent": sorted(silent),
            "mean_abs_bps": mean_abs,
            "mean_abs_bps_delta": delta,
            "max_abs_bps": round(max(live_abs), 2) if live_abs else None,
            "ghost_weight": round(sum(self.weights.get(k, 0) for k in silent), 4),
            "trade_attributions": self.trade_attributions,
            "ml_resets": self._ml_resets,
        }

    # ------------------------------------------------ main learning cycle
    def run(self, market, regime=None):
        now = time.time()
        regime_label = (regime or {}).get("label", "unknown")
        self.current_regime_label = regime_label

        # 1. score matured strategy signals.
        # The GROSS (pre-cost) forward return measures a sleeve's DIRECTIONAL
        # skill. We store it for the dashboard AND feed it to the bandit as a
        # separate, heavily-discounted teacher (signal_learn_weight). Rationale:
        # closed-trade PnL is the premium signal but is desperately sparse (a
        # handful of fills/hour across all regime×strategy arms — it can take
        # weeks to fill one arm to NET_EDGE_N). The signal stream is 100-1000x
        # more abundant, so a small amount of it dramatically speeds warm-up on
        # DIRECTION. We deliberately do NOT subtract taker cost here (that made
        # every 5m sleeve a confirmed loser); net-of-cost skill is still taught
        # by on_trade_closed, which keeps the dominant weight.
        matured = db.unscored_signals(now - SIGNAL_EVAL_HORIZON_SEC)
        n_scored, n_retry, n_abandon = 0, 0, 0
        from ..tunables import tv as _tv0
        sig_w = _tv0("signal_learn_weight")
        sig_clip = _tv0("signal_learn_clip")
        for s in matured:
            p0 = self._price_at(s["product"], s["ts"], market)
            p1 = self._price_at(s["product"], s["ts"] + SIGNAL_EVAL_HORIZON_SEC, market) \
                 or market.price(s["product"])
            if p0 and p1:
                fwd = (p1 / p0 - 1) * s["direction"]
                db.score_signal(s["rowid"], fwd)   # store GROSS fwd for the UI
                n_scored += 1
                # feed the bandit a discounted, clipped, confidence-weighted
                # directional lesson (only for arms it actually allocates to)
                if sig_w > 0 and s["strategy"] in STRATEGIES:
                    aligned = max(-sig_clip, min(sig_clip, fwd))
                    reg = s.get("regime") or "unknown"
                    self.bandit.update(reg, s["strategy"],
                                       aligned * s["confidence"] * sig_w)
            elif now - s["ts"] > LOOKUP_GIVE_UP_SEC:
                db.abandon_signal(s["rowid"])
                n_abandon += 1
            else:
                n_retry += 1

        # 1b. COUNTERFACTUAL credit (Phase 3): score actionable conviction
        # signals the book was gated out of with the net-of-cost return they
        # would have made, and teach the bandit — every throttled-but-wanted
        # trade becomes a learning sample instead of a silent lost decision.
        self._score_skips(market)

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

        # 2b. forget old evidence — n, variance, AND mean decay toward 0 so a
        # silent sleeve cannot keep a frozen +8bps reputation at decaying n.
        from ..tunables import tv as _tv
        n_pruned = self.bandit.decay(gamma=_tv("bandit_decay_gamma"))

        # 2c. a broken ML head is RESET before we mix weights or train, so this
        # cycle's samples hit a fresh net instead of 47k updates at 26% acc.
        ml_reset = self._maybe_reset_broken_ml()
        silent = self._silent_sleeves()

        # 3. Thompson-sampled weights. Silent sleeves (no evolved champion, ML
        # ≤ coin-flip) are hard-zeroed — not floored at 4%, not EMA-carried.
        self.weights = self._allocate(regime_label, silent)

        # 4. online model training + drift check
        n_trained = self._train_online_model(market)
        self._check_drift()

        # 5. background genetic evolution
        self.maybe_evolve()

        quality = self._cycle_quality(regime_label, silent)
        quality.update({"n_scored": n_scored, "n_pruned": n_pruned,
                        "n_trained": n_trained, "ml_reset": ml_reset})
        self.last_cycle = quality

        self.last_run = now
        delta = quality.get("mean_abs_bps_delta")
        db.log_event("learn",
                     f"Learning cycle: {n_scored} scored (UI), {n_retry} retry, "
                     f"{n_abandon} abandon, {n_trained} ML updates, "
                     f"pruned={n_pruned}, silent={quality['silent'] or 'none'}, "
                     f"ghost_w={quality['ghost_weight']}, "
                     f"|mean|={quality['mean_abs_bps']} bps "
                     f"(Δ {delta if delta is not None else 'n/a'}), "
                     f"ml_reset={ml_reset}, fills={self.trade_attributions}, "
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
            "skip_attributions": self.skip_attributions,
            "evolution": evolution.stats(),
            "llm_advisor": self._llm_advisor_stats(),
            "model_advisor": self._model_advisor_stats(),
            "exit_advisor": self._exit_advisor_stats(),
            "direction": self._direction_stats(),
            "last_run": self.last_run,
            "last_cycle": self.last_cycle,
        }

    @staticmethod
    def _exit_advisor_stats():
        try:
            from .exit_advisor import exit_advisor
            return exit_advisor.stats()
        except Exception:
            return {}

    @staticmethod
    def _direction_stats():
        try:
            from .direction import direction_learner
            return direction_learner.stats()
        except Exception:
            return {}

    @staticmethod
    def _llm_advisor_stats():
        try:
            from .llm_advisor import advisor
            return advisor.stats()
        except Exception:
            return {"enabled": False}

    @staticmethod
    def _model_advisor_stats():
        try:
            from .model_advisor import advisor
            return advisor.stats()
        except Exception:
            return {"enabled": False}


learner = Learner()
