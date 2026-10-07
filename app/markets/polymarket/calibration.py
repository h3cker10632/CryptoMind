"""Learned calibration for Polymarket probabilities — fitted from the
forecast ledger instead of hand-tuned gains.

The bot's probability is the market price nudged by hand-tuned strategy
gains (momentum, mean reversion, longshot fade, ...). Whether those nudges
help is exactly what resolved markets can tell us — each one is an
independent outcome with the market price as the benchmark. This fits

    P(outcome 0) = logistic(a + b * logit(market p) + c * (logit(bot p) - logit(market p))
                            + per-strategy vote terms)

walk-forward on the ledger: a forecast made at time f is scored by a model
trained only on markets RESOLVED before f. Those out-of-sample probabilities
are judged like the raw bot (skill_gate.evaluate: Brier vs the market price,
per market). Only if they beat the market does the engine use the calibrated
probability to pick the side and the edge (`adjust`) — the ledger keeps the
raw forecasts either way.
"""
from __future__ import annotations
import math
import time

import numpy as np

MIN_TRAIN = 100            # resolved markets before a model is fitted
REFIT_EVERY = 50           # forecasts per walk-forward block
_cache = {"ts": 0.0, "model": None, "report": None}
TTL = 600


def _logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _strategies():
    try:
        from .signals import STRATEGIES
        return list(STRATEGIES)
    except Exception:
        return []


def features(market_p0, bot_p0, votes, strategies=None):
    strategies = strategies if strategies is not None else _strategies()
    lm = _logit(market_p0)
    x = [lm, _logit(bot_p0) - lm]
    votes = votes or {}            # per-strategy leans toward outcome 0 (the ledger's
    x += [float(votes.get(s, 0.0) or 0.0) for s in strategies]   # "votes" = signal["leans"])
    return x


def _rows_xy(rows, strategies):
    X, y, f_ts, r_ts, cid, mkt = [], [], [], [], [], []
    for r in rows:
        if r.get("resolved_outcome0") not in (0, 1):
            continue
        try:
            X.append(features(r["market_p0"], r["predicted_p0"], r.get("votes"), strategies))
            y.append(float(r["resolved_outcome0"]))
            f_ts.append(float(r["forecast_ts"]))
            r_ts.append(float(r.get("resolved_ts") or r["forecast_ts"]))
            cid.append(r.get("condition_id"))
            mkt.append(float(r["market_p0"]))
        except (KeyError, TypeError, ValueError):
            continue
    return (np.array(X), np.array(y), np.array(f_ts), np.array(r_ts), cid, np.array(mkt))


def _fit(X, y):
    from ...ml.models import Logistic
    return Logistic(alpha=5.0).fit(X, y, np.ones(len(y)))


def walk_forward(rows, strategies=None):
    """Out-of-sample calibrated probabilities as ledger-like rows
    ({condition_id, predicted_p0 (calibrated), market_p0, resolved_outcome0})."""
    strategies = strategies if strategies is not None else _strategies()
    X, y, f_ts, r_ts, cid, mkt = _rows_xy(rows, strategies)
    if len(y) < MIN_TRAIN + 10:
        return []
    order = np.argsort(f_ts)
    out = []
    for b in range(0, len(order), REFIT_EVERY):
        block = order[b:b + REFIT_EVERY]
        cutoff = f_ts[block[0]]
        train = np.flatnonzero(r_ts < cutoff)
        if len(set(cid[i] for i in train)) < MIN_TRAIN:
            continue
        mdl = _fit(X[train], y[train])
        p = mdl.predict(X[block])
        for i, pi in zip(block, p):
            out.append({"condition_id": cid[i], "predicted_p0": float(pi),
                        "market_p0": float(mkt[i]), "resolved_outcome0": int(y[i])})
    return out


def status(rows=None, now=None, min_markets=100):
    """{"model": fitted on everything resolved (or None), "report": the
    walk-forward evaluation}. Cached for TTL seconds."""
    now = now or time.time()
    if rows is None and _cache["report"] is not None and now - _cache["ts"] < TTL:
        return {"model": _cache["model"], "report": _cache["report"]}
    if rows is None:
        from ... import db
        try:
            rows = db.pm_forecast_rows(limit=100000)
        except Exception:
            rows = []
    from .skill_gate import evaluate
    strategies = _strategies()
    oos = walk_forward(rows, strategies)
    report = evaluate(oos, min_markets=min_markets)
    X, y, *_ = _rows_xy(rows, strategies)
    model = _fit(X, y) if len(y) >= MIN_TRAIN else None
    _cache.update(ts=now, model=model, report=report)
    return {"model": model, "report": report}


def adjust(market, sig, model, strategies=None):
    """Re-derive side / price / fair / edge from the calibrated probability of
    outcome 0 (same rules as signals.evaluate)."""
    if model is None or sig.get("fair_p0") is None:
        return sig
    p0 = market["prices"][0]
    x = np.array([features(p0, sig["fair_p0"], sig.get("leans"), strategies)])
    q0 = float(model.predict(x)[0])
    q0 = min(max(q0, 0.001), 0.999)
    out = dict(sig)
    if abs(q0 - p0) < 1e-6:
        out["outcome_index"] = None
    elif q0 > p0:
        out.update(outcome_index=0, price=p0, fair=round(q0, 4))
    else:
        out.update(outcome_index=1, price=market["prices"][1], fair=round(1 - q0, 4))
    idx = out.get("outcome_index")
    if idx is not None:
        from ...tunables import tv
        out["outcome"] = (market.get("outcomes") or ["A", "B"])[idx]
        # confidence is the calibrated model's own conviction — its edge over
        # the market price on the engine's edge scale — whichever way the
        # strategies leaned (the model already weighed their leans)
        out["confidence"] = round(min(1.0, abs(q0 - p0) / max(tv("pm_edge_scale"), 1e-6)), 3)
        if idx != sig.get("outcome_index"):
            # calibration bought the side the strategies did not lean to: sign
            # the votes toward it (the bandit credits by them)
            sign = 1.0 if idx == 0 else -1.0
            out["votes"] = {s: round(v * sign, 3) for s, v in (sig.get("leans") or {}).items()}
    out["edge"] = round(out["fair"] - out["price"], 4) if out.get("outcome_index") is not None \
        else 0.0
    out["fair_p0"] = round(q0, 4)
    out["calibrated"] = True
    return out
