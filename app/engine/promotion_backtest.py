"""Backtest the RESEARCH LOOP itself: would its promotion rule have helped?

The live loop makes one promotion decision every so often, so judging the rule
on live results alone would take years. This replays it over history: on each
evaluation date it sees only data up to that date, applies the same rule as
live (evidence.promotion_decision: both-halves backtest gate, deflated Sharpe,
paired always-valid forward test since the replay began), and the replayed
"core" holds whatever is champion until the next evaluation. The result is
the return of FOLLOWING THE PROCESS, next to: never switching (the default
champion), holding the coins, and the best candidate in hindsight (a bound
nobody could have had).

Every candidate is a causal function of the panel (row t uses rows <= t —
checked for the engine strategies in tests/test_engine.py), so each one's
daily returns are computed ONCE over the whole panel and sliced at each
evaluation date — identical to re-running on truncated data, and fast.
"""
from __future__ import annotations
import datetime as dt

import numpy as np

from . import backtest as B, evidence as E


def run(panel, candidates, default_champion, backtest_from="2019-06-01",
        replay_from=None, eval_every=30, cost=0.006, band_rel=0.2, min_dsr=0.95,
        min_forward_days=30, max_forward_days=180, alpha=0.05, tau=0.1,
        weights_fn=None, progress=None):
    """Returns a JSON-able report. `replay_from` (ISO date) is when the
    replayed process starts tracking forward evidence (default: two years of
    backtest after `backtest_from`)."""
    say = progress or (lambda m: None)
    if weights_fn is None:
        from .challengers import weights as cw, vol_forecast_for

        def weights_fn(cfg, panel, start=0):
            return cw(cfg, panel, start=start, vol=vol_forecast_for(cfg, panel))
    R = panel.returns()
    epoch = dt.date(1970, 1, 1)
    d0 = (dt.date.fromisoformat(backtest_from) - epoch).days
    start = int((panel.days < d0).sum())
    rets = {}
    for name, cfg in candidates.items():
        W = weights_fn(cfg, panel, start=start)
        rets[name], _, _ = B.simulate_drift(W, R, cost, start=start, band_rel=band_rel)
        say(f"{name}: {len(rets[name])} days")
    n = min(len(r) for r in rets.values())
    rets = {k: v[:n] for k, v in rets.items()}
    day_of = panel.days[start:start + n]                 # day each return is DECIDED on
    if replay_from:
        s0 = int((day_of < (dt.date.fromisoformat(replay_from) - epoch).days).sum())
    else:
        s0 = min(n - 1, 730)
    n_trials = len(candidates)
    champ = default_champion
    path = np.zeros(0)
    promotions, retired = [], set()
    for e in range(s0, n, eval_every):
        bt = {k: v[:e] for k, v in rets.items()}
        fwd = {k: v[s0:e] for k, v in rets.items()}
        srs = [v.mean() / v.std() for v in bt.values() if len(v) > 1 and v.std() > 0]
        sd = float(np.std(srs)) if len(srs) >= 2 else None
        verdicts, best = E.promotion_decision(
            bt, fwd, champ, n_trials, sd, min_dsr=min_dsr,
            min_forward_days=min_forward_days, max_forward_days=max_forward_days,
            alpha=alpha, tau=tau, retired=retired)
        retired |= {k for k, v in verdicts.items() if v["retired"]}
        if best:
            promotions.append({"day": str(epoch + dt.timedelta(days=int(day_of[e]))),
                               "from": champ, "to": best,
                               "z": verdicts[best]["forward_test"].get("z")})
            champ = best
        path = np.concatenate([path, rets[champ][e:min(e + eval_every, n)]])
    window = slice(s0, n)

    def summ(r):
        st = B.report(r)
        return {"full": st["full"], "first_half": st["first_half"],
                "second_half": st["second_half"], "cagr_pct": st.get("cagr_pct")}
    hindsight = max(rets, key=lambda k: B.stats(rets[k][window])["sharpe"] or -9)
    out = {"from": str(epoch + dt.timedelta(days=int(day_of[s0]))) if n else None,
           "days": int(n - s0), "eval_every_days": eval_every, "candidates": len(candidates),
           "process": summ(path), "promotions": promotions,
           "never_switch": summ(rets[default_champion][window]),
           "best_in_hindsight": {"name": hindsight, **summ(rets[hindsight][window])}}
    if "btc_eth_hold" in rets:
        out["hold_btc_eth"] = summ(rets["btc_eth_hold"][window])
    p, ns = out["process"]["full"]["sharpe"], out["never_switch"]["full"]["sharpe"]
    out["verdict"] = ("promotion process beat never switching" if p is not None and ns is not None
                      and p > ns else "promotion process did NOT beat never switching")
    return out
