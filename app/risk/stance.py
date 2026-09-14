"""Trading stance — Passive / Auto / Aggressive.

The stance scales HOW HARD the system trades without touching WHAT it
believes. It adjusts: risk per trade, position cap, the confidence gate
for full-size entries, and how often exploration probes fire.

Modes (settings "trade_mode"):
  passive     — fixed defensive multipliers
  aggressive  — fixed offensive multipliers
  auto        — score current conditions and interpolate between the two

Auto scoring inputs (all already tracked elsewhere in the system):
  + recent win rate above 50%          (evidence the edge is working)
  + calm volatility regime             (friendlier conditions)
  - drawdown from peak                 (capital protection first)
  - consecutive losses                 (streak = something is off)
  - feature drift (PSI)                (models may be stale)
  - macro blackout upcoming/active     (event risk)
"""
import time

def PASSIVE_SET():
    from ..tunables import tv
    return {"risk": tv("stance_passive_risk"), "gate": tv("stance_passive_gate"),
            "pos": 0.5, "explore": 0.5}

def NEUTRAL_SET():
    from ..tunables import tv
    return {"risk": 1.0, "gate": tv("min_confidence"), "pos": 1.0, "explore": 1.0}

def AGGRESSIVE_SET():
    from ..tunables import tv
    return {"risk": tv("stance_aggr_risk"), "gate": tv("stance_aggr_gate"),
            "pos": 1.5, "explore": 2.0}


def _lerp(a, b, t):
    return a + (b - a) * t


def _blend(t):
    """t in [0,1]: 0 = fully passive, 0.5 = neutral, 1 = fully aggressive."""
    P, N, A = PASSIVE_SET(), NEUTRAL_SET(), AGGRESSIVE_SET()
    lo, hi = (P, N) if t < 0.5 else (N, A)
    u = t * 2 if t < 0.5 else (t - 0.5) * 2
    return {k: _lerp(lo[k], hi[k], u) for k in N}


class Stance:
    def __init__(self):
        self.last = None          # cached dict for snapshot/UI
        self._ts = 0.0

    def _auto_score(self):
        """Return (t in [0,1], reasons list). 0.5 = neutral baseline."""
        reasons = []
        t = 0.5

        # recent performance (last 20 closed trades)
        from ..execution.paper import broker
        recent = broker.closed_trades[-20:]
        if len(recent) >= 8:
            wr = sum(1 for x in recent if x["pnl"] > 0) / len(recent)
            pnl = sum(x["pnl"] for x in recent)
            if wr >= 0.55 and pnl > 0:
                t += 0.20; reasons.append(f"hot hand: {wr:.0%} win rate, recent PnL {pnl:+,.0f}")
            elif wr <= 0.35 or pnl < 0:
                t -= 0.15; reasons.append(f"cold streak: {wr:.0%} win rate, recent PnL {pnl:+,.0f}")
        else:
            reasons.append("thin trade history — staying near neutral")

        # drawdown / loss streak
        from .manager import risk
        rs = getattr(risk, "_last_status", None) or {}
        dd = rs.get("drawdown", 0.0)
        if dd > 0.05:
            t -= 0.25; reasons.append(f"drawdown {dd:.1%} — defense first")
        elif dd > 0.02:
            t -= 0.10; reasons.append(f"mild drawdown {dd:.1%}")
        if risk.consecutive_losses >= 3:
            t -= 0.15; reasons.append(f"{risk.consecutive_losses} consecutive losses")

        # feature drift
        try:
            from ..learn.loop import learner
            if (learner.drift_state or {}).get("drifting"):
                t -= 0.15; reasons.append("feature drift detected — models may be stale")
        except Exception:
            pass

        # volatility regime
        try:
            from ..data.market import market
            reg = market.regime()
            if reg.get("vol_state") == "high-vol":
                t -= 0.10; reasons.append("high-volatility regime")
            elif reg.get("vol_state") == "low-vol" and reg.get("trend") == "up":
                t += 0.10; reasons.append("calm uptrend — conditions favorable")
        except Exception:
            pass

        # macro blackout
        try:
            from ..data.calendar import calendar
            blackout, ev = calendar.blackout()
            if blackout:
                t -= 0.20; reasons.append(f"macro blackout: {ev['title']}")
        except Exception:
            pass

        return max(0.0, min(1.0, t)), reasons

    def current(self):
        """Compute (cached 10s) the active multiplier set."""
        now = time.time()
        if self.last and now - self._ts < 10:
            return self.last
        from .. import settings as app_settings
        mode = app_settings.get("trade_mode")
        if mode == "passive":
            m, t, reasons = PASSIVE_SET(), 0.0, ["manual: passive mode"]
        elif mode == "aggressive":
            m, t, reasons = AGGRESSIVE_SET(), 1.0, ["manual: aggressive mode"]
        else:
            t, reasons = self._auto_score()
            m = _blend(t)
        label = ("passive" if t < 0.35 else
                 "aggressive" if t > 0.65 else "neutral")
        self.last = {
            "mode": mode, "label": label, "score": round(t, 3),
            "risk_mult": round(m["risk"], 3),
            "conf_gate": round(m["gate"], 3),
            "max_pos_mult": round(m["pos"], 3),
            "explore_mult": round(m["explore"], 3),
            "reasons": reasons,
        }
        self._ts = now
        return self.last


stance = Stance()
