"""ML candidate strategies for the research loop (selection "ml_rank" /
"trend_meta" in app/engine/challengers.py). Panel -> (T, N) weights, causal,
walk-forward — scored exactly like every other candidate.

  ml_rank     among coins in their trend (price above the `sma` average,
              point-in-time top-`universe_top` universe), hold the `top_k`
              the pooled model ranks highest; same sizing options and
              tranching as the momentum rule it competes with.
  trend_meta  the trend rule on `assets` (e.g. the BTC/ETH champion's), each
              position SIZED by a pooled meta-model's probability that a coin
              in its trend gains over the next h days — trained on the
              `train_top` liquid universe, so BTC/ETH borrow thousands of trend
              episodes instead of their own ~90. Size = clip(2p - 0.5, 0, 1).

Live: a walk-forward fit takes minutes, so the research loop (a daily
background process) saves each ML candidate's weights (`save_latest`) and
the live core reads the newest saved row (`latest_saved`) — stale (> 3 days)
means hold, like stale price data.
"""
from __future__ import annotations
import json
import os

import numpy as np

from ..engine import strategies as S, universe as U, features as F
from . import dataset as D, models as Mo

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "reports", "ml_weights")


def _cfg_key(cfg):
    import hashlib
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def rank_predictions(cfg, panel, start=0):
    top = int(cfg.get("universe_top", 20))
    h = int(cfg.get("horizon", 7))
    mask = U.liquid_mask(panel, top)
    X, y, w, _, _ = D.build(panel, mask, h=h)
    return Mo.walk_forward(X, y, w, model=cfg.get("model", "ensemble"), h=h,
                           refit_days=int(cfg.get("refit_days", 30))), mask


def meta_probabilities(cfg, panel, start=0):
    h = int(cfg.get("horizon", 7))
    sma = int(cfg.get("sma", 125))
    mask = U.liquid_mask(panel, int(cfg.get("train_top", 30)))
    X, _, _, _, _ = D.build(panel, mask, h=h)
    gap = F.sma_gap(panel.close, sma)
    held_any = ~np.isnan(panel.close) & (np.nan_to_num(gap, nan=-1) > 0)
    lab = D.meta_labels(panel, held_any & mask, h=h)
    ok = np.isfinite(lab) & np.isfinite(X).all(axis=2)
    n_t = ok.sum(axis=1, keepdims=True)
    w = np.where(ok, 1.0 / (h * np.maximum(n_t, 1)), 0.0)
    return Mo.walk_forward(X, np.nan_to_num(lab), w, model=cfg.get("model", "ridge"), h=h,
                           refit_days=int(cfg.get("refit_days", 30)), classify=True)


def weights(cfg, panel, start=0):
    if cfg["selection"] == "ml_rank":
        pred, mask = rank_predictions(cfg, panel, start)
        W = S.trend_portfolio(panel, "score", cfg.get("sizing", "equal"),
                              sma=int(cfg.get("sma", 100)), top_k=int(cfg.get("top_k", 5)),
                              vol_target=float(cfg.get("vol_target", 0.5)), score=pred,
                              start=start, tranches=int(cfg.get("tranches", 1)),
                              universe=mask, hysteresis=float(cfg.get("hysteresis", 0.0)))
    elif cfg["selection"] == "trend_meta":
        uni = np.zeros((panel.T, panel.N), bool)
        uni[:, [panel.coins.index(a) for a in cfg["assets"] if a in panel.coins]] = True
        base = S.trend_portfolio(panel, "trend", cfg.get("sizing", "equal"),
                                 sma=int(cfg.get("sma", 125)), start=start, universe=uni,
                                 hysteresis=float(cfg.get("hysteresis", 0.0)),
                                 vol_target=float(cfg.get("vol_target", 0.5)))
        p = meta_probabilities(cfg, panel, start)
        scale = np.clip(2 * np.nan_to_num(p, nan=0.75) - 0.5, 0.0, 1.0)  # no model yet: full size
        W = base * scale
    else:
        raise ValueError(cfg["selection"])
    return W


def save_latest(name, cfg, panel, W, keep_days=10):
    """Save the last `keep_days` rows of a candidate's weights for live use."""
    os.makedirs(DIR, exist_ok=True)
    rows = W[-keep_days:]
    doc = {"name": name, "cfg_key": _cfg_key(cfg), "data_version": panel.data_version,
           "days": [int(d) for d in panel.days[-keep_days:]],
           "weights": [{c: float(v) for c, v in zip(panel.coins, r) if v > 0} for r in rows]}
    tmp = os.path.join(DIR, f"{_cfg_key(cfg)}.json.tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, os.path.join(DIR, f"{_cfg_key(cfg)}.json"))


def latest_saved(cfg, today, max_age_days=3):
    """(day, {coin: weight}) from the newest saved row, or None if missing or
    older than `max_age_days`."""
    try:
        with open(os.path.join(DIR, f"{_cfg_key(cfg)}.json")) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    if not doc.get("days"):
        return None
    day = doc["days"][-1]
    if today - day > max_age_days:
        return None
    return day, doc["weights"][-1]
