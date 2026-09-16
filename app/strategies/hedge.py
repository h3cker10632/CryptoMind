"""Market-Neutral Pair Hedge — statistical relative-strength divergence.

Research basis (operator-supplied): market-neutral hedging with strict
statistical entry criteria shows the most consistent documented results
(reported 72% win rate, 3.8% max DD). Entry only when the relative-strength
spread between two CORRELATED coins diverges beyond `hedge_z_entry` standard
deviations (default 2.7); long the laggard, short the leader, equal notional
(net market exposure ~ 0). Exit on mean reversion inside `hedge_z_exit`,
or force-close at max age. All fills go through the same paper broker with
full fees and slippage — no free lunch.
"""
import itertools, statistics, time
from ..tunables import tv
from .. import db


class PairHedger:
    def __init__(self):
        self.active = {}      # "A|B" -> hedge dict
        self.history = []     # closed hedges
        self.last_scan = {}   # pair -> {z, corr} for the dashboard
        self.cooldowns = {}   # "A|B" -> ts the pair last closed (re-entry gate)

    # ---------- statistics ----------
    @staticmethod
    def _rets(closes):
        return [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]

    @staticmethod
    def _corr(x, y):
        n = min(len(x), len(y))
        x, y = x[-n:], y[-n:]
        mx, my = sum(x)/n, sum(y)/n
        sx = (sum((a-mx)**2 for a in x))**0.5
        sy = (sum((b-my)**2 for b in y))**0.5
        if sx == 0 or sy == 0:
            return 0.0
        return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy)

    @staticmethod
    def _pair_cost_viable(ca, cb, z, leg_notional, lookback=96):
        """Cost gate on the PAIR. The expected reversion move (spread going from
        the current |z| back to the exit band) must translate into a dollar
        profit that clears round-trip cost on ALL FOUR fills — open+close on
        BOTH legs — by `hedge_cost_multiple`. A pair whose expected convergence
        can't beat the fee drag on four fills is structurally unprofitable and
        is skipped (the direct analogue of the directional cost-viability gate).
        """
        import math
        n = min(len(ca), len(cb), lookback)
        if n < 40:
            return False
        ratio = [math.log(ca[-n + i] / cb[-n + i]) for i in range(n)]
        sd = statistics.pstdev(ratio)
        if sd < 1e-9:
            return False
        # expected fractional convergence of the log-ratio (entry z -> exit z)
        delta_z = abs(z) - tv("hedge_z_exit")
        if delta_z <= 0:
            return False
        expected_move_frac = delta_z * sd
        expected_pnl = leg_notional * expected_move_frac
        per_fill = tv("fee_rate") + tv("slippage_bps") / 1e4
        cost_four_fills = 4 * per_fill * leg_notional
        return expected_pnl > tv("hedge_cost_multiple") * cost_four_fills

    @staticmethod
    def _attribute(trade, h):
        """Feed a closed hedge leg's NET return to the bandit under the `hedge`
        sleeve. The generic learner.on_trade_closed would DROP hedge legs (they
        carry no directional `votes`), so the hedge book would learn nothing.
        Here we stamp an explicit `hedge` vote + the entry regime and route it
        through the same attribution path so the sleeve is scored like any other.
        """
        try:
            from ..learn.loop import learner
            entry_notional = trade["qty"] * trade["entry"]
            if entry_notional <= 0:
                return
            # a hedge leg's outcome credits the `hedge` sleeve, direction-agnostic
            trade = dict(trade)
            trade["votes"] = {"hedge": 1.0}
            trade.setdefault("regime_at_entry", h.get("regime_at_entry", "unknown"))
            learner.on_trade_closed(trade)
        except Exception:
            pass

    def _spread_z(self, ca, cb, lookback=96):
        """z-score of the log-ratio spread between two close series."""
        import math
        n = min(len(ca), len(cb), lookback)
        if n < 40:
            return None
        ratio = [math.log(ca[-n+i] / cb[-n+i]) for i in range(n)]
        mu = sum(ratio) / n
        sd = statistics.pstdev(ratio)
        if sd < 1e-9:
            return None
        return (ratio[-1] - mu) / sd

    # ---------- main tick ----------
    def tick(self, market, broker, risk):
        from .. import settings as app_settings
        if not app_settings.get("hedge_enabled") or not app_settings.get("allow_shorts"):
            return
        products = [p for p in market.candles if len(market.candles[p]) >= 60]
        closes = {p: [c[4] for c in market.candles[p]] for p in products}

        # --- manage open hedges ---
        for key in list(self.active.keys()):
            h = self.active[key]
            a, b = h["long"], h["short"]
            if a not in closes or b not in closes:
                continue
            z = self._spread_z(closes[h["pair"][0]], closes[h["pair"][1]])
            age_h = (time.time() - h["opened"]) / 3600
            done = reason = None
            if z is not None and abs(z) <= tv("hedge_z_exit"):
                done, reason = True, f"spread reverted (z={z:+.2f})"
            elif age_h > tv("hedge_max_age_hours"):
                done, reason = True, f"max age {age_h:.0f}h"
            elif a not in broker.positions or b not in broker.positions:
                done, reason = True, "leg closed independently (stop/target)"
            if done:
                pnl = 0.0
                for leg in (a, b):
                    if leg in broker.positions:
                        t = broker.sell(leg, market.price(leg), f"hedge exit: {reason}")
                        if t:
                            pnl += t["pnl"]
                            risk.on_trade_closed(t)
                            # attribute the leg's realized PnL to the `hedge`
                            # strategy sleeve (never an empty-votes drop): the
                            # bandit must see how the hedge book actually does.
                            self._attribute(t, h)
                h["closed"] = time.time(); h["pnl"] = round(pnl, 2)
                h["exit_reason"] = reason
                self.history.append(h)
                # start the re-entry cooldown for this pair (prevents churn on a
                # spread that keeps grazing the entry band).
                self.cooldowns[key] = time.time()
                del self.active[key]
                db.log_event("hedge", f"HEDGE CLOSED {key}: {reason} pnl={pnl:+.2f}")

        # --- scan for new entries ---
        if len(self.active) >= 2:          # at most 2 concurrent pair hedges
            return
        # Respect the SAME global risk gates as directional entries: a kill
        # switch, a daily-loss halt, or being at the gross-exposure cap must
        # block new hedge risk too (a "market-neutral" book is still capital and
        # still bleeds fees while a stop is being hunted).
        if risk.killed or risk.halted_today:
            return
        equity = broker.equity(market)
        if equity <= 0:
            return
        if broker.exposure(market) / equity >= tv("max_gross_exposure"):
            return
        leg_notional = equity * tv("hedge_notional_pct")
        if leg_notional < tv("min_notional"):
            return
        # two legs add 2*leg_notional of gross exposure — don't breach the cap
        if (broker.exposure(market) + 2 * leg_notional) / equity > tv("max_gross_exposure"):
            return
        now = time.time()
        cooldown_s = tv("hedge_cooldown_hours") * 3600
        best = None
        for a, b in itertools.combinations(sorted(products), 2):
            if a in broker.positions or b in broker.positions:
                continue
            key = f"{a}|{b}"
            if key in self.active:
                continue
            # re-entry cooldown: skip a pair that closed within the window
            last_close = self.cooldowns.get(key, 0)
            if now - last_close < cooldown_s:
                continue
            corr = self._corr(self._rets(closes[a]), self._rets(closes[b]))
            z = self._spread_z(closes[a], closes[b])
            if z is None:
                continue
            self.last_scan[key] = {"z": round(z, 2), "corr": round(corr, 2)}
            if corr < tv("hedge_corr_min") or abs(z) < tv("hedge_z_entry"):
                continue
            # PAIR cost gate: the expected reversion move must clear round-trip
            # cost on ALL FOUR fills (open+close, both legs) by a healthy margin.
            if not self._pair_cost_viable(closes[a], closes[b], z, leg_notional):
                continue
            if best is None or abs(z) > abs(best[2]):
                best = (a, b, z, corr)
        if not best:
            return
        a, b, z, corr = best
        # z>0: A rich vs B -> short A, long B ; z<0: A cheap -> long A, short B
        long_leg, short_leg = (b, a) if z > 0 else (a, b)
        pa, pb = market.price(long_leg), market.price(short_leg)
        if not pa or not pb:
            return
        fa = market.features(long_leg) or {}
        fb = market.features(short_leg) or {}
        atr_a = fa.get("atr_swing") or fa.get("atr") or pa * 0.02
        atr_b = fb.get("atr_swing") or fb.get("atr") or pb * 0.02
        regime_label = market.regime().get("label", "unknown")
        # BOTH LEGS LIVE OR NEITHER: open the long, then the short; if the short
        # fails to fill, immediately unwind the long so we never carry a naked,
        # directional orphan from what is supposed to be a market-neutral book.
        l = broker.open(long_leg, 1, leg_notional, pa,
                        pa - 4*atr_a, pa + 8*atr_a, f"HEDGE long leg z={z:+.2f}",
                        is_hedge=True, regime_at_entry=regime_label)
        if l is None:
            return
        s = broker.open(short_leg, -1, leg_notional, pb,
                        pb + 4*atr_b, pb - 8*atr_b, f"HEDGE short leg z={z:+.2f}",
                        is_hedge=True, regime_at_entry=regime_label)
        if s is None:
            # unwind the orphaned long and attribute its (tiny) round-trip loss
            t = broker.sell(long_leg, market.price(long_leg) or pa,
                            "hedge abort: short leg failed")
            if t:
                risk.on_trade_closed(t)
                self._attribute(t, {"regime_at_entry": regime_label})
            return
        h = {"pair": (a, b), "long": long_leg, "short": short_leg,
             "z_entry": round(z, 2), "corr": round(corr, 2),
             "regime_at_entry": regime_label,
             "notional": round(leg_notional, 2), "opened": time.time()}
        self.active[f"{a}|{b}"] = h
        db.log_event("hedge", f"HEDGE OPEN {a}|{b}: long {long_leg} / short "
                              f"{short_leg} z={z:+.2f} corr={corr:.2f} "
                              f"${leg_notional:,.0f}/leg")
        from ..alerts import alert
        alert("info", "Pair hedge opened",
              f"Long {long_leg} / short {short_leg}, z={z:+.2f}, corr={corr:.2f}")

    def snapshot(self):
        return {"active": list(self.active.values()),
                "recent": self.history[-10:],
                "scan": dict(sorted(self.last_scan.items(),
                                    key=lambda kv: -abs(kv[1]["z"]))[:8])}


hedger = PairHedger()
