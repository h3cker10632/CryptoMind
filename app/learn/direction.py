"""Direction Learner — improves the long-vs-short decision.

The complaint it addresses: "sometimes it shorted when it should have gone
long." Two mechanisms, both self-correcting:

  1. LEARNED DIRECTIONAL EDGE — per (regime, direction) it tracks the realized,
     after-cost return of closed trades. If longs in bull/normal have paid and
     shorts have consistently lost, the composite is nudged toward the side
     that has actually worked in that regime. This is a *bias*, not an override:
     it shifts the composite and can flip a marginal call, but overwhelming
     signal still wins.

  2. MULTI-TIMEFRAME VETO — a hard sanity check against fighting a strong,
     aligned higher-timeframe trend. Shorting when 15m/1h/4h are all firmly
     bullish (or longing when all firmly bearish) is exactly the "should have
     been a long" mistake; when the requested direction opposes a strongly
     aligned trend, the signal is neutralized (made non-actionable) rather than
     taken. Guarded by the strength of `mtf_align` so it only vetoes clear cases.

Both are learned/derived from data already in the system (closed-trade PnL by
regime+direction, and the multi-timeframe feature added in Tier 2).
"""


class DirectionLearner:
    def __init__(self):
        # (regime, dir) -> [n, mean_net_return]
        self.edge = {}
        self.n_updates = 0
        self.n_flips = 0
        self.n_vetoes = 0
        self.n_conformal_vetoes = 0
        # CONFORMAL direction gate: a calibrated prediction set over {long,short}
        # that replaces the hard mtf_align threshold once it has enough evidence.
        from .direction_conformal import DirectionConformalGate
        self.conformal = DirectionConformalGate()
        # None = follow the evidence gate (app/learn/gate.py); True/False
        # forces it (the learner ablation forces it on for its own copy).
        self.enabled = None

    def _on(self):
        if self.enabled is not None:
            return self.enabled
        from .gate import active
        return active("direction")

    @staticmethod
    def _tv(key, default):
        try:
            from ..tunables import tv
            return tv(key)
        except Exception:
            return default

    # ---------------- learning ----------------
    def on_trade_closed(self, trade):
        """Fold a closed trade's net return into its (regime, direction) cell."""
        regime = trade.get("regime_at_entry", "unknown")
        side = trade.get("side", 1)
        entry_notional = trade.get("qty", 0) * trade.get("entry", 0)
        if entry_notional <= 0:
            return
        net = trade["pnl"] / entry_notional
        key = (regime, 1 if side > 0 else -1)
        n, mean = self.edge.get(key, (0, 0.0))
        n += 1
        a = 0.2 if n > 10 else 1.0 / n
        mean = mean + a * (net - mean)
        self.edge[key] = (n, mean)
        self.n_updates += 1
        # feed the conformal direction gate with (mtf_at_entry, favored side)
        try:
            self.conformal.observe_trade(trade)
        except Exception:
            pass

    def edge_for(self, regime, direction):
        rec = self.edge.get((regime, direction))
        return rec[1] if rec else 0.0

    def _confidence(self, regime, direction):
        rec = self.edge.get((regime, direction))
        return min(1.0, rec[0] / 15.0) if rec else 0.0

    # ---------------- bias applied at signal time ----------------
    def bias(self, regime, composite):
        """Return an additive bias for the composite in this regime, based on
        which direction has historically paid. Positive pushes long, negative
        pushes short. Bounded so it can only tip marginal calls."""
        long_edge = self.edge_for(regime, 1)
        short_edge = self.edge_for(regime, -1)
        conf = min(self._confidence(regime, 1), self._confidence(regime, -1))
        if conf <= 0:
            return 0.0
        gain = self._tv("direction_bias_gain", 8.0)
        cap = self._tv("direction_bias_cap", 0.25)
        # difference in realized edge between the two sides, scaled to signal units
        raw = (long_edge - short_edge) * gain * conf
        return max(-cap, min(cap, raw))

    def adjust(self, regime, composite, mtf_align):
        """Apply the learned bias to a composite and report whether the sign
        flipped. Returns (new_composite, flipped)."""
        if not self._on():
            return composite, False
        b = self.bias(regime, composite)
        if b == 0.0:
            return composite, False
        new = composite + b
        flipped = (new > 0) != (composite > 0) and abs(composite) > 1e-9
        if flipped:
            self.n_flips += 1
        return new, flipped

    # ---------------- multi-timeframe veto ----------------
    def veto(self, direction, mtf_align):
        """True if `direction` fights the higher-timeframe trend and should NOT
        be taken.

        Once the CONFORMAL direction gate is calibrated it drives the decision:
        veto only when the proposed side is confidently EXCLUDED from the
        prediction set (a coverage-backed judgement) instead of a hand-picked
        alignment constant. Until then — or when the gate abstains on an
        ambiguous set — we fall back to the original hard-threshold check so
        behaviour is unchanged cold-start."""
        try:
            cv, cwhy = (self.conformal.veto(direction, mtf_align) if self._on()
                        else (False, ""))
        except Exception:
            cv, cwhy = False, ""
        if cv:
            self.n_vetoes += 1
            self.n_conformal_vetoes += 1
            return True, cwhy
        # legacy hard-threshold fallback (also active while the gate warms up)
        gate = self._tv("mtf_veto_align", 0.75)
        if direction < 0 and mtf_align >= gate:
            self.n_vetoes += 1
            return True, f"short vetoed: HTF strongly bullish (align={mtf_align:+.2f})"
        if direction > 0 and mtf_align <= -gate:
            self.n_vetoes += 1
            return True, f"long vetoed: HTF strongly bearish (align={mtf_align:+.2f})"
        return False, ""

    # ---------------- reporting / persistence ----------------
    def stats(self):
        table = {}
        for (regime, d), (n, mean) in sorted(self.edge.items()):
            table.setdefault(regime, {})["long" if d > 0 else "short"] = {
                "n": n, "mean_net": round(mean, 5)}
        return {
            "updates": self.n_updates,
            "flips": self.n_flips,
            "vetoes": self.n_vetoes,
            "conformal_vetoes": self.n_conformal_vetoes,
            "conformal": self.conformal.stats(),
            "regime_edge": table,
        }

    def capture(self):
        return {
            "edge": {f"{r}||{d}": [n, m] for (r, d), (n, m) in self.edge.items()},
            "n_updates": self.n_updates, "n_flips": self.n_flips,
            "n_vetoes": self.n_vetoes,
            "n_conformal_vetoes": self.n_conformal_vetoes,
            "conformal": self.conformal.to_dict(),
        }

    def restore(self, d):
        if not d:
            return
        try:
            out = {}
            for k, v in (d.get("edge") or {}).items():
                r, ds = k.rsplit("||", 1)
                out[(r, int(ds))] = (int(v[0]), float(v[1]))
            self.edge = out
            self.n_updates = d.get("n_updates", 0)
            self.n_flips = d.get("n_flips", 0)
            self.n_vetoes = d.get("n_vetoes", 0)
            self.n_conformal_vetoes = d.get("n_conformal_vetoes", 0)
            self.conformal.load_dict(d.get("conformal"))
        except Exception:
            pass


direction_learner = DirectionLearner()
