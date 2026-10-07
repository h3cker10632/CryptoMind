"""Judge out-of-sample predictions the way the signal screen judges a
feature — and against what the model has to beat.

  ranking   daily rank correlation (IC) of the prediction with the realized
            label across the universe, Newey-West t with h lags, both halves;
            the SAME statistic for the baseline (30-day momentum rank) on the
            same days; the model must beat it in both halves.
  calibration  mean realized label per prediction decile (should rise).
  meta      Brier score of the probability vs the base rate, both halves.
"""
from __future__ import annotations
import numpy as np

from ..engine import screen as S
from ..engine.evidence import nw_tstat


def _halves_mean(s):
    s = s[np.isfinite(s)]
    h = len(s) // 2
    return [float(s[:h].mean()) if h else float("nan"),
            float(s[h:].mean()) if len(s) - h else float("nan")]


def ranking(pred, label, mask, baseline, h=7):
    ok_days = np.isfinite(pred).any(axis=1)
    ic = S.ic_series(np.where(ok_days[:, None], pred, np.nan), label, mask)
    ic_b = S.ic_series(np.where(ok_days[:, None], baseline, np.nan), label, mask)
    both = np.isfinite(ic) & np.isfinite(ic_b)
    ic, ic_b = np.where(both, ic, np.nan), np.where(both, ic_b, np.nan)
    m, mb = _halves_mean(ic), _halves_mean(ic_b)
    t = nw_tstat(ic, lags=h)
    t_diff = nw_tstat(ic - ic_b, lags=h)
    out = {"days": int(both.sum()),
           "ic": round(float(np.nanmean(ic)), 4) if both.any() else None,
           "ic_t": None if t != t else round(t, 2),
           "ic_halves": [round(x, 4) for x in m],
           "baseline_ic": round(float(np.nanmean(ic_b)), 4) if both.any() else None,
           "baseline_ic_halves": [round(x, 4) for x in mb],
           "beats_baseline_t": None if t_diff != t_diff else round(t_diff, 2)}
    out["passes"] = bool(t == t and t >= 2.0 and all(a > 0 for a in m)
                         and all(a > b for a, b in zip(m, mb)))
    return out


def calibration(pred, label, mask, bins=10):
    ok = mask & np.isfinite(pred) & np.isfinite(label)
    p, y = pred[ok], label[ok]
    if len(p) < bins * 20:
        return None
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [round(float(y[idx == b].mean()), 4) if (idx == b).any() else None
            for b in range(bins)]


def meta(prob, label):
    ok = np.isfinite(prob) & np.isfinite(label)
    rows = np.flatnonzero(ok.any(axis=1))
    if not len(rows):
        return {"days": 0, "passes": False}
    half = rows[len(rows) // 2]
    res = {}
    for name, sl in (("first_half", slice(0, half)), ("second_half", slice(half, None))):
        o = ok[sl]
        p, y = prob[sl][o], label[sl][o]
        base = y.mean() if len(y) else float("nan")
        res[name] = {"brier": round(float(((p - y) ** 2).mean()), 5) if len(y) else None,
                     "base_rate_brier": round(float(((base - y) ** 2).mean()), 5)
                     if len(y) else None, "n": int(len(y))}
    res["days"] = int(len(rows))
    res["passes"] = all(r["brier"] is not None and r["brier"] < r["base_rate_brier"]
                        for k, r in res.items() if k.endswith("half"))
    return res
