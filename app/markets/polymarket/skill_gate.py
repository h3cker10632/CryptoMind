"""Evidence gate for Polymarket bets — the same rule as everything else in
the system: nothing drives trading until its record shows it helps.

The forecast ledger (db.pm_forecasts) stores one point-in-time forecast per
market and time-to-resolution bucket, whether or not a bet was placed, and
scores it once the market resolves. Betting against the market price only
makes sense if the bot's probabilities are BETTER than the market's, so:

  * per resolved market: the bot's Brier score minus the market price's
    (averaged over that market's buckets — they share one outcome, so they
    are one piece of evidence, not several);
  * the gate opens when >= `pm_gate_min_markets` markets are resolved and the
    bot is better with t <= -`pm_gate_min_t` (one-sided).

`pm_trade_mode`: "off", "on", or "auto" (default) = the gate decides. While
closed the engine keeps forecasting and recording would-open decisions, so
evidence keeps accumulating.
"""
from __future__ import annotations
import math
import time

_cache = {"ts": 0.0, "result": None}
TTL = 600


def evaluate(rows, min_markets=100, min_t=2.0):
    """{"open": bool, "markets": n, "mean_brier_diff": ..., "t": ..., "why": str}
    from forecast rows (bot minus market Brier; negative = bot better)."""
    per = {}
    for r in rows:
        y = r.get("resolved_outcome0")
        if y not in (0, 1):
            continue
        try:
            b = max(0.0, min(1.0, float(r["predicted_p0"])))
            m = max(0.0, min(1.0, float(r["market_p0"])))
        except (KeyError, TypeError, ValueError):
            continue
        per.setdefault(r.get("condition_id"), []).append((b - y) ** 2 - (m - y) ** 2)
    d = [sum(v) / len(v) for v in per.values()]
    n = len(d)
    out = {"open": False, "markets": n, "mean_brier_diff": None, "t": None}
    if n < 2:
        out["why"] = f"{n} resolved markets (need {min_markets})"
        return out
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    if var > 0:
        t = mean / math.sqrt(var / n)
    else:                                   # identical differences: as sure as it gets
        t = math.copysign(1e6, mean) if mean else 0.0
    out.update(mean_brier_diff=round(mean, 5), t=round(t, 2))
    if n < min_markets:
        out["why"] = f"{n} resolved markets (need {min_markets})"
    elif t > -min_t:
        out["why"] = (f"bot vs market Brier {mean:+.4f} (t {t:+.2f}) over {n} markets: "
                      f"not better than the market price")
    else:
        out["open"] = True
        out["why"] = f"bot beats the market price: Brier {mean:+.4f} (t {t:+.2f}) over {n} markets"
    return out


def status(now=None):
    """Cached gate state from the ledger, honoring `pm_trade_mode`."""
    from ... import settings
    try:
        mode = settings.get("pm_trade_mode")
    except Exception:
        mode = "auto"
    mode = mode if mode in ("off", "on", "auto") else "auto"
    if mode != "auto":
        return {"mode": mode, "open": mode == "on", "why": f"pm_trade_mode = {mode}"}
    now = now or time.time()
    if _cache["result"] is None or now - _cache["ts"] > TTL:
        from ... import db
        try:
            min_m = int(settings.get("pm_gate_min_markets"))
        except Exception:
            min_m = 100
        try:
            rows = db.pm_forecast_rows(limit=100000)
        except Exception:
            rows = []
        _cache.update(ts=now, result=evaluate(rows, min_markets=min_m))
    return dict(_cache["result"], mode="auto")
