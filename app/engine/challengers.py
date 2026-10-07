"""Champion / challenger loop — how the system learns WITHOUT fooling itself.

Every candidate is a frozen config. Each run:
  1. BACKTEST every candidate on the data store (honest universe, drift
     simulation, real costs) and log it to the experiment registry;
  2. FORWARD-TRACK every candidate from the day its config was registered:
     data after registration was never seen when the config was chosen, so
     this is genuine out-of-sample evidence (changing a config re-registers
     it and restarts its clock);
  3. PROMOTE a challenger only when ALL hold (app/engine/evidence.py):
       * its backtest beats the champion's on Sharpe in both halves;
       * deflated Sharpe >= `min_dsr` counting every variant ever tried;
       * >= `min_forward_days` of forward tracking AND a paired, always-valid
         sequential test vs the champion on the same days says 'better' (or,
         undecided after `max_forward_days`, the paired difference is positive).
     A 'worse' verdict RETIRES a challenger early (it stays reported, never
     promoted, until its config changes).
The champion is persisted in reports/champion.json; the core follows it when
`core_strategy` is "champion" (app/strategies/core.py).

Besides the built-in CANDIDATES, frozen configs can be QUEUED without a code
change (`register`, reports/candidate_queue.json) — e.g. by the ML lab when a
model passes its own out-of-sample gate — so their forward clocks start the
day they exist. Every queued config is one more trial for the deflated Sharpe.

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
from . import evidence as E

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(ROOT, "reports", "challengers.json")
CHAMPION = os.path.join(ROOT, "reports", "champion.json")
QUEUE = os.path.join(ROOT, "reports", "candidate_queue.json")
MAX_QUEUED = 20          # active queued candidates (each one is a counted trial)
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


SELECTIONS = ("hold", "trend", "momentum", "ml_rank", "trend_meta")
SIZINGS = ("equal", "inverse_vol", "vol_target", "vol_forecast")
_BOUNDS = {"sma": (10, 400), "hysteresis": (0.0, 0.2), "top_k": (1, 20),
           "vol_target": (0.05, 2.0), "tranches": (1, 7), "universe_top": (2, 60),
           "horizon": (1, 60), "refit_days": (7, 365)}


def validate_config(cfg):
    """A candidate config is data — check it before it can run. Returns an
    error string, or None if valid."""
    if not isinstance(cfg, dict):
        return "config must be an object"
    if cfg.get("selection") not in SELECTIONS:
        return f"selection must be one of {SELECTIONS}"
    if cfg.get("sizing", "equal") not in SIZINGS:
        return f"sizing must be one of {SIZINGS}"
    if ("assets" in cfg) == ("universe_top" in cfg):
        return "exactly one of assets / universe_top"
    if "assets" in cfg:
        a = cfg["assets"]
        if (not isinstance(a, list) or not a or len(a) > 20
                or not all(isinstance(x, str) and x.endswith("-USD") for x in a)):
            return "assets must be 1-20 '<COIN>-USD' strings"
    for k, (lo, hi) in _BOUNDS.items():
        if k in cfg:
            v = cfg[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
                return f"{k} must be in [{lo}, {hi}]"
    allowed = {"assets", "selection", "sizing", "model", "features"} | set(_BOUNDS)
    extra = set(cfg) - allowed
    if extra:
        return f"unknown keys {sorted(extra)}"
    return None


def candidates():
    """Built-in candidates plus the active queued ones (built-ins win a
    name clash)."""
    q = _load(QUEUE, {}).get("candidates", {})
    out = dict(CANDIDATES)
    for name, e in q.items():
        if name not in out and not e.get("removed") and validate_config(e.get("config")) is None:
            out[name] = e["config"]
    return out


def register(name, cfg, source="manual", now=None):
    """Queue a frozen config as a challenger. Its forward clock starts at the
    next research-loop run. Returns (ok, message)."""
    err = validate_config(cfg)
    if err:
        return False, err
    if not isinstance(name, str) or not name.replace("_", "").isalnum() or len(name) > 48:
        return False, "name must be letters, digits and _ (<= 48 chars)"
    if name in CANDIDATES:
        return False, f"{name} is a built-in candidate"
    q = _load(QUEUE, {"candidates": {}})
    active = [n for n, e in q["candidates"].items() if not e.get("removed")]
    old = q["candidates"].get(name)
    if old and not old.get("removed") and config_hash(old["config"]) == config_hash(cfg):
        return True, "already queued"
    if name not in active and len(active) >= MAX_QUEUED:
        return False, f"queue full ({MAX_QUEUED} active); remove one first"
    q["candidates"][name] = {"config": cfg, "source": source, "added_at": now or time.time()}
    _save(QUEUE, q)
    return True, "queued"


def unregister(name):
    q = _load(QUEUE, {"candidates": {}})
    if name not in q["candidates"]:
        return False
    q["candidates"][name]["removed"] = True
    _save(QUEUE, q)
    return True


def weights(cfg, panel, start=0, vol=None):
    """(T, N) target weights of a candidate on a panel of any coins."""
    if cfg["selection"] in ("ml_rank", "trend_meta"):
        from ..ml import strategies as MLS
        return MLS.weights(cfg, panel, start=start)
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


def vol_forecast_for(cfg, panel):
    """The HAR vol forecast a `vol_forecast` candidate sizes with — fitted on
    the SAME coins the live core loads for it (`P.from_store(cfg["assets"])`:
    those coins only, from the first day any of them has a bar), placed back
    into `panel`'s columns. HAR is pooled over the coins it is fitted on and
    refits on row indices, so fitting it on all ~400 store coins (as this loop
    used to) gave the backtest a different forecast from the one live trading
    would use — and took ~100x longer. None for other sizings."""
    if cfg.get("sizing") != "vol_forecast":
        return None
    if "assets" not in cfg:
        return M.har_vol_forecast(panel)
    sub = panel.subset(cfg["assets"])
    rows = np.flatnonzero((~np.isnan(sub.close)).any(axis=1))
    out = np.full((panel.T, panel.N), np.nan)
    if not len(rows):
        return out
    lo = int(rows[0])
    trimmed = P.Panel(sub.days[lo:], sub.coins, sub.close[lo:], sub.volume[lo:],
                      sub.data_version,
                      None if sub.high is None else sub.high[lo:],
                      None if sub.low is None else sub.low[lo:], sub.bar_sec)
    v = M.har_vol_forecast(trimmed)
    for j, c in enumerate(sub.coins):
        out[lo:, panel.coins.index(c)] = v[:, j]
    return out


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
    """(name, config) of the current champion. A promoted QUEUED candidate
    stays champion from its stored config even if it leaves the queue."""
    c = _load(CHAMPION, {})
    name = c.get("name")
    cands = candidates()
    if name in cands:
        return name, cands[name]
    if name and validate_config(c.get("config")) is None:
        return name, c["config"]
    return DEFAULT_CHAMPION, CANDIDATES[DEFAULT_CHAMPION]


def run(panel=None, backtest_from="2019-06-01", cost=0.006, band_rel=0.2,
        min_forward_days=90, min_dsr=0.95, today=None, progress=None,
        max_forward_days=180, alpha=0.05, tau=0.1):
    """One loop iteration. Returns the report (also saved to STATE)."""
    say = progress or (lambda m: None)
    panel = panel or P.from_store()
    R = panel.returns()
    today = today if today is not None else int(panel.days[-1])
    d0 = (dt.date.fromisoformat(backtest_from) - dt.date(1970, 1, 1)).days
    start = int((panel.days < d0).sum())
    state = _load(STATE, {"candidates": {}})
    champ_name, champ_cfg = champion()
    cands = candidates()
    cands.setdefault(champ_name, champ_cfg)
    out = {}
    for name, cfg in cands.items():
        h = config_hash(cfg)
        reg = state["candidates"].get(name)
        if not reg or reg.get("config_hash") != h:          # new / changed: (re)start clock
            reg = {"config_hash": h, "registered_day": today}
        W = weights(cfg, panel, start=start, vol=vol_forecast_for(cfg, panel))
        r, inv, turn = B.simulate_drift(W, R, cost, start=start, band_rel=band_rel)
        Rg.log(FAMILY, name, cfg, panel.data_version, r, {}, start_day=d0)
        fs = int((panel.days < reg["registered_day"]).sum())
        fwd, _, _ = B.simulate_drift(W, R, cost, start=max(fs, start), band_rel=band_rel)
        out[name] = {"config": cfg, "registered_day": reg["registered_day"],
                     "backtest_rets": r, "forward_rets": fwd, "W": W,
                     "turnover_per_year": round(float(turn.sum()) / max(1, len(turn)) * 365, 1)}
        state["candidates"][name] = reg
        say(f"{name}: backtest {len(r)} days, forward {len(fwd)} days")
    n = sum(Rg.n_trials(f) for f in RESEARCH_FAMILIES)
    sd = Rg.trial_sharpe_std(FAMILY)
    retired = {nm for nm, reg in state["candidates"].items()
               if reg.get("retired") and nm in out}
    verdicts, best = E.promotion_decision(
        {nm: o["backtest_rets"] for nm, o in out.items()},
        {nm: o["forward_rets"] for nm, o in out.items()},
        champ_name, n, sd, min_dsr=min_dsr, min_forward_days=min_forward_days,
        max_forward_days=max_forward_days, alpha=alpha, tau=tau, retired=retired)
    report = {"ran_at": time.time(), "data_version": panel.data_version, "trials": n,
              "champion": champ_name, "candidates": {}}
    for name, o in out.items():
        v = verdicts[name]
        if v["retired"] and not state["candidates"][name].get("retired"):
            state["candidates"][name]["retired"] = {
                "day": today, "why": f"forward test vs {champ_name}: {v['forward_test']}"}
        report["candidates"][name] = dict(
            config=o["config"], registered_day=o["registered_day"],
            turnover_per_year=o["turnover_per_year"], **v)
    try:
        from . import costs_tax as CT
        report["costs_taxes"] = CT.scenarios(out[champ_name]["W"], panel, start=start,
                                             band_rel=band_rel, base_cost=cost)
    except Exception as e:                   # a report extra never blocks the loop
        report["costs_taxes"] = {"error": str(e)}
    if best:
        _save(CHAMPION, {"name": best, "config": cands[best], "promoted_at": time.time(),
                         "replaced": champ_name})
        report["promoted"] = best
        report["champion"] = best
    elif not os.path.exists(CHAMPION):
        _save(CHAMPION, {"name": champ_name, "config": champ_cfg,
                         "promoted_at": time.time(), "replaced": None})
    state["last_report"] = report
    _save(STATE, state)
    return report
