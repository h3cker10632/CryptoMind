"""Champion / challenger loop — how the system learns WITHOUT fooling itself.

Every candidate is a frozen config. Each run:
  1. BACKTEST every candidate on the data store (honest universe, drift
     simulation, real costs) and log it to the experiment registry;
  2. FORWARD-TRACK every candidate from the day its config was registered:
     data after registration was never seen when the config was chosen, so
     this is genuine out-of-sample evidence (changing a config re-registers
     it and restarts its clock);
  3. PROMOTE a challenger only when ALL hold:
       * >= `min_forward_days` of forward tracking, with a higher forward
         Sharpe than the champion over the same days;
       * its backtest beats the champion's on Sharpe in both halves;
       * deflated Sharpe >= `min_dsr` counting every variant ever tried.
The champion is persisted in reports/champion.json; the core follows it when
`core_strategy` is "champion" (app/strategies/core.py).

Candidates that need many coins (alt sleeves) are scored like the rest.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import time

import numpy as np

from . import panel as P, strategies as S, backtest as B, registry as Rg, universe as U
from . import risk_models as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(ROOT, "reports", "challengers.json")
CHAMPION = os.path.join(ROOT, "reports", "champion.json")
FAMILY = "challengers"
RESEARCH_FAMILIES = ("core_v2", "core_v2_long", "core_v3", FAMILY)
MAJORS = ("BTC-USD", "ETH-USD")

# The candidate set. Adding one is a new trial (and the registry counts it).
CANDIDATES = {
    "btc_eth_trend": {"assets": list(MAJORS), "selection": "trend", "sizing": "equal",
                      "sma": 125, "hysteresis": 0.02},
    "btc_eth_trend_volf": {"assets": list(MAJORS), "selection": "trend",
                           "sizing": "vol_forecast", "vol_target": 0.6, "sma": 125,
                           "hysteresis": 0.02},
    "btc_eth_trend_fast": {"assets": list(MAJORS), "selection": "trend", "sizing": "equal",
                           "sma": 75, "hysteresis": 0.02},
    "btc_eth_trend_slow": {"assets": list(MAJORS), "selection": "trend", "sizing": "equal",
                           "sma": 200, "hysteresis": 0.02},
    "btc_eth_hold": {"assets": list(MAJORS), "selection": "hold"},
    "alt_momentum_top10": {"universe_top": 10, "selection": "momentum",
                           "sizing": "vol_target", "vol_target": 0.5, "sma": 100,
                           "top_k": 5, "tranches": 7},
}
DEFAULT_CHAMPION = "btc_eth_trend"


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def weights(cfg, panel, start=0, vol=None):
    """(T, N) target weights of a candidate on a panel of any coins."""
    if cfg["selection"] == "hold":
        W = np.zeros((panel.T, panel.N))
        idx = [panel.coins.index(a) for a in cfg["assets"] if a in panel.coins]
        live = ~np.isnan(panel.close[:, idx])
        W[:, idx] = np.where(live, 1.0, 0.0) / np.maximum(live.sum(axis=1, keepdims=True), 1)
        return W
    if "assets" in cfg:
        uni = np.zeros((panel.T, panel.N), bool)
        uni[:, [panel.coins.index(a) for a in cfg["assets"] if a in panel.coins]] = True
    else:
        uni = U.liquid_mask(panel, cfg["universe_top"])
    if cfg.get("sizing") == "vol_forecast" and vol is None:
        vol = M.har_vol_forecast(panel)
    return S.trend_portfolio(panel, cfg["selection"], cfg.get("sizing", "equal"),
                             sma=cfg.get("sma", 100), top_k=cfg.get("top_k", 5),
                             vol_target=cfg.get("vol_target", 0.5), start=start,
                             tranches=cfg.get("tranches", 1), universe=uni,
                             hysteresis=cfg.get("hysteresis", 0.0), vol=vol)


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def champion():
    """(name, config) of the current champion."""
    c = _load(CHAMPION, {})
    name = c.get("name") if c.get("name") in CANDIDATES else DEFAULT_CHAMPION
    return name, CANDIDATES[name]


def run(panel=None, backtest_from="2019-06-01", cost=0.006, band_rel=0.2,
        min_forward_days=90, min_dsr=0.95, today=None, progress=None):
    """One loop iteration. Returns the report (also saved to STATE)."""
    say = progress or (lambda m: None)
    panel = panel or P.from_store()
    R = panel.returns()
    today = today if today is not None else int(panel.days[-1])
    d0 = (dt.date.fromisoformat(backtest_from) - dt.date(1970, 1, 1)).days
    start = int((panel.days < d0).sum())
    state = _load(STATE, {"candidates": {}})
    champ_name, _ = champion()
    vol = (M.har_vol_forecast(panel) if any(c.get("sizing") == "vol_forecast"
                                            for c in CANDIDATES.values()) else None)
    out = {}
    for name, cfg in CANDIDATES.items():
        h = config_hash(cfg)
        reg = state["candidates"].get(name)
        if not reg or reg.get("config_hash") != h:          # new / changed: (re)start clock
            reg = {"config_hash": h, "registered_day": today}
        W = weights(cfg, panel, start=start, vol=vol)
        r, inv, turn = B.simulate_drift(W, R, cost, start=start, band_rel=band_rel)
        Rg.log(FAMILY, name, cfg, panel.data_version, r, {}, start_day=d0)
        fs = int((panel.days < reg["registered_day"]).sum())
        fwd, _, _ = B.simulate_drift(W, R, cost, start=max(fs, start), band_rel=band_rel)
        out[name] = {"config": cfg, "registered_day": reg["registered_day"],
                     "backtest_rets": r, "forward_rets": fwd,
                     "turnover_per_year": round(float(turn.sum()) / max(1, len(turn)) * 365, 1)}
        state["candidates"][name] = reg
        say(f"{name}: backtest {len(r)} days, forward {len(fwd)} days")
    n = sum(Rg.n_trials(f) for f in RESEARCH_FAMILIES)
    sd = Rg.trial_sharpe_std(FAMILY)
    c_bt = out[champ_name]["backtest_rets"]
    h = len(c_bt) // 2
    report = {"ran_at": time.time(), "data_version": panel.data_version, "trials": n,
              "champion": champ_name, "candidates": {}}
    for name, o in out.items():
        rep = B.report(o["backtest_rets"], n_trials=n, trial_sr_std=sd)
        bt = o["backtest_rets"]
        beats_bt = name != champ_name and all(
            B.stats(bt[sl])["sharpe"] > B.stats(c_bt[sl])["sharpe"]
            for sl in (slice(0, h), slice(h, None)))
        fwd = o["forward_rets"]
        cf = out[champ_name]["forward_rets"]
        k = min(len(fwd), len(cf))
        f_sh = B.stats(fwd[-k:])["sharpe"] if k > 1 else None
        c_sh = B.stats(cf[-k:])["sharpe"] if k > 1 else None
        eligible = (name != champ_name and beats_bt and rep["deflated_sharpe"] >= min_dsr
                    and k >= min_forward_days and f_sh is not None and f_sh > c_sh)
        report["candidates"][name] = {
            "config": o["config"], "registered_day": o["registered_day"],
            "backtest": {k2: rep[k2] for k2 in ("full", "first_half", "second_half",
                                                "cagr_pct", "deflated_sharpe")},
            "turnover_per_year": o["turnover_per_year"],
            "beats_champion_backtest_both_halves": bool(beats_bt),
            "forward_days": int(len(fwd)), "forward": B.stats(fwd),
            "forward_sharpe_vs_champion": [f_sh, c_sh] if k > 1 else None,
            "eligible_for_promotion": bool(eligible)}
    winners = [n_ for n_, c in report["candidates"].items() if c["eligible_for_promotion"]]
    if winners:
        best = max(winners, key=lambda n_: report["candidates"][n_]["forward"]["sharpe"])
        _save(CHAMPION, {"name": best, "config": CANDIDATES[best], "promoted_at": time.time(),
                         "replaced": champ_name})
        report["promoted"] = best
        report["champion"] = best
    elif not os.path.exists(CHAMPION):
        _save(CHAMPION, {"name": champ_name, "config": CANDIDATES[champ_name],
                         "promoted_at": time.time(), "replaced": None})
    state["last_report"] = report
    _save(STATE, state)
    return report
