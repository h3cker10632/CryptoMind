"""Daily lab — selection and sizing for the CORE holding, tested on years of
daily history before any of it touches the live core.

Selection (which coins the core holds):
  trend     every coin above its `sma_days` average (the current core rule)
  momentum  the top `top_k` of those by 30-day return, re-picked weekly
  rank      the top `top_k` of those by a gradient-boosted model's predicted
            next-week return relative to the other coins, re-picked weekly.
            The model is trained WALK-FORWARD: refit every `refit_days` on
            samples whose label had fully happened before the refit day, so
            every prediction scored here is out-of-sample.
Sizing (how much of each):
  equal       each held coin gets 1/N of the core (N = coins in the universe)
  inverse_vol held coins weighted by 1/volatility, same total as `equal`
  vol_target  `equal` weights scaled so the portfolio's estimated volatility
              (60-day covariance) is `vol_target`, never above 100% invested

Every variant is simulated day by day with the core's real cost per side, and
judged on SHARPE (return per unit of risk; sizing variants change how much is
invested, so raw return isn't comparable) in BOTH halves of the out-of-sample
period. A variant is only eligible to drive the live core if it beats the
current rule (trend + equal) in both halves; `rank` must also beat plain
`momentum` in both, or the model adds nothing over a one-line rule.

The live core (app/strategies/core.py) calls `target_weights` — the same
function the simulation uses — so what was tested is what runs.

Caveat: the universe is TODAY's coins, so history carries survivorship bias
(coins that died aren't in it). It biases every variant the same way.
"""
from __future__ import annotations
import math
import pickle
import statistics
import time

from ..data.features import stdev

H = 7                      # label horizon (days) = selection holding period
REBAL_DAYS = 7             # momentum / rank re-pick cadence
FEATURES = ("r7", "r14", "r30", "r60", "r90", "vol30", "d50", "d100", "d200",
            "vol_ratio", "dd30", "xs_r30", "btc_r30", "btc_d100")
SELECTIONS = ("trend", "momentum", "rank")
SIZINGS = ("equal", "inverse_vol", "vol_target")


# ------------------------------------------------------------------ features
def coin_features(closes, volumes=None, sma_days=100):
    """Features of one coin at its LAST bar, from its daily history up to and
    including that bar (oldest first). None if under 30 days of history.
    `d_core` is the gap to the core's own `sma_days` average (trend gate)."""
    n = len(closes)
    if n < 31 or closes[-1] <= 0:
        return None
    c = closes[-1]

    def ret(k):
        return c / closes[-1 - k] - 1 if n > k and closes[-1 - k] > 0 else float("nan")

    def sma_gap(k):
        return c / (sum(closes[-k:]) / k) - 1 if n >= k else float("nan")

    lr = [math.log(closes[i] / closes[i - 1]) for i in range(max(1, n - 60), n)
          if closes[i] > 0 and closes[i - 1] > 0]
    vol30 = stdev(lr[-30:], ddof=0) if len(lr) >= 20 else float("nan")
    vr = float("nan")
    if volumes is not None and len(volumes) >= 30:
        m30 = sum(volumes[-30:]) / 30
        vr = (sum(volumes[-7:]) / 7) / m30 if m30 > 0 else float("nan")
    return {"r7": ret(7), "r14": ret(14), "r30": ret(30), "r60": ret(60), "r90": ret(90),
            "vol30": vol30, "d50": sma_gap(50), "d100": sma_gap(100), "d200": sma_gap(200),
            "vol_ratio": vr, "dd30": c / max(closes[-30:]) - 1,
            "d_core": sma_gap(sma_days),
            "rets60": lr[-60:], "n_days": n, "close": c}


def add_cross_features(feats, btc="BTC-USD"):
    """Cross-coin features for one day: 30-day return rank among the coins,
    and BTC's trend as a market factor. Mutates and returns `feats`."""
    r30 = sorted((f["r30"], p) for p, f in feats.items() if not math.isnan(f["r30"]))
    rank = {p: (i + 0.5) / len(r30) for i, (_, p) in enumerate(r30)}
    b = feats.get(btc) or {}
    for p, f in feats.items():
        f["xs_r30"] = rank.get(p, float("nan"))
        f["btc_r30"] = b.get("r30", float("nan"))
        f["btc_d100"] = b.get("d100", float("nan"))
    return feats


def x_row(f):
    return [f.get(k, float("nan")) for k in FEATURES]


# ------------------------------------------------------------------ weights
def target_weights(feats, selection="trend", sizing="equal", preds=None, prev=None,
                   day=0, sma_days=100, top_k=5, vol_target=0.5):
    """{coin: fraction of the core allocation} for one day.

    feats  — {coin: coin_features(...)} with cross features added
    preds  — {coin: model score} (rank selection only)
    prev   — {"day": d, "picks": [...]} the last weekly pick (momentum/rank);
             returned updated as the second value."""
    elig = [p for p, f in feats.items() if f["n_days"] >= sma_days]
    if not elig:
        return {}, prev
    held = [p for p in elig if not math.isnan(feats[p]["d_core"]) and feats[p]["d_core"] > 0]
    n_slots = len(elig)
    if selection in ("momentum", "rank"):
        if prev and day - prev.get("day", -1e9) < REBAL_DAYS:
            held = [p for p in prev["picks"] if p in feats]
        else:
            if selection == "momentum":
                score = {p: feats[p]["r30"] for p in held}
            else:
                score = {p: (preds or {}).get(p) for p in held}
            ranked = sorted((p for p in held if score.get(p) is not None
                             and not math.isnan(score[p])), key=lambda p: -score[p])
            held = ranked[:top_k]
            prev = {"day": day, "picks": held}
        n_slots = top_k
    if not held:
        return {}, prev
    if sizing == "inverse_vol":
        iv = {p: 1.0 / max(feats[p]["vol30"], 1e-4) for p in held
              if not math.isnan(feats[p]["vol30"])}
        tot = sum(iv.values()) or 1.0
        gross = len(held) / n_slots
        return {p: gross * v / tot for p, v in iv.items()}, prev
    w = {p: 1.0 / n_slots for p in held}
    if sizing == "vol_target":
        pv = _port_vol(w, feats)
        if pv and pv > 0:
            s = vol_target / pv
            s = min(s, 1.0 / sum(w.values()))            # never above 100% invested
            w = {p: v * s for p, v in w.items()}
    return w, prev


def _port_vol(w, feats):
    """Annualized portfolio volatility from each coin's last 60 daily log
    returns (aligned on the most recent common days)."""
    import numpy as np
    series = {p: feats[p]["rets60"] for p in w if len(feats[p]["rets60"]) >= 20}
    if not series:
        return None
    m = min(len(s) for s in series.values())
    ps = list(series)
    R = np.array([series[p][-m:] for p in ps])
    cov = np.atleast_2d(np.cov(R))
    wv = np.array([w[p] for p in ps])
    return math.sqrt(max(float(wv @ cov @ wv), 0.0) * 365)


# ------------------------------------------------------------------ data
def daily_panel(candles_by_product, now=None):
    """(days, {coin: {day: (close, volume)}}) from daily candles, dropping the
    still-open day."""
    now = now or time.time()
    panel = {}
    for p, rows in candles_by_product.items():
        d = {}
        for r in rows or []:
            try:
                t, c, v = int(r[0]), float(r[4]), float(r[5])
            except (TypeError, ValueError, IndexError):
                continue
            if c > 0 and t + 86400 <= now:
                d[t // 86400] = (c, v)
        if len(d) > 31:
            panel[p] = d
    days = sorted({k for d in panel.values() for k in d})
    return days, panel


def _features_by_day(days, panel, sma_days=100):
    """[(day, {coin: feats})] for every day, each from history up to that day."""
    hist = {p: ([], []) for p in panel}
    out = []
    for k in days:
        feats = {}
        for p, d in panel.items():
            if k in d:
                hist[p][0].append(d[k][0])
                hist[p][1].append(d[k][1])
                f = coin_features(hist[p][0], hist[p][1], sma_days)
                if f:
                    feats[p] = f
        out.append((k, add_cross_features(feats)))
    return out


def _labels(days, panel):
    """{(day, coin): next-H-day log return minus that day's cross-coin mean}."""
    raw = {}
    for k in days:
        rs = {}
        for p, d in panel.items():
            if k in d and k + H in d:
                rs[p] = math.log(d[k + H][0] / d[k][0])
        if len(rs) >= 3:
            m = sum(rs.values()) / len(rs)
            for p, r in rs.items():
                raw[(k, p)] = r - m
    return raw


# ------------------------------------------------------------------ model
def _new_model():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                             min_samples_leaf=100, l2_regularization=1.0,
                                             random_state=0)
    except ImportError:
        return None


def fit_model(X, y):
    m = _new_model()
    if m is None or len(X) < 500:
        return None
    m.fit(X, y)
    return m


def train_set(samples, labels, k):
    """[(x, y)] usable for a refit on day k: only samples whose H-day label
    window ended before k, so nothing after the refit leaks in."""
    return [(x, labels[(j, p)]) for j, p, x in samples if j + H < k]


def walk_forward_preds(fbd, labels, min_train_days=365, refit_days=30, progress=None):
    """{(day, coin): out-of-sample score}. A training sample (day j) is used
    for a refit on day k only if its label window ended before k (j + H < k)."""
    preds = {}
    first = fbd[0][0] if fbd else 0
    samples = [(k, p, x_row(f)) for k, feats in fbd for p, f in feats.items()
               if (k, p) in labels]
    model, next_refit = None, first + min_train_days
    for i, (k, feats) in enumerate(fbd):
        if k >= next_refit:
            tr = train_set(samples, labels, k)
            model = fit_model([x for x, _ in tr], [y for _, y in tr])
            next_refit = k + refit_days
            if progress:
                progress(f"rank model refit on day {k - first} ({len(tr)} samples)")
        if model is not None and feats:
            ps = list(feats)
            for p, s in zip(ps, model.predict([x_row(feats[p]) for p in ps])):
                preds[(k, p)] = float(s)
    return preds


# ------------------------------------------------------------------ simulation
def simulate(fbd, panel, selection, sizing, preds=None, start_day=None, cost_side=0.006,
             **kw):
    """Daily returns of one variant: weights chosen at day k's close earn the
    k -> k+1 move; turnover pays `cost_side` per unit traded."""
    rets, gross, w_prev, prev = [], [], {}, None
    for i in range(len(fbd) - 1):
        k, feats = fbd[i]
        if start_day is not None and k < start_day:
            continue
        k1 = fbd[i + 1][0]
        pk = {p: preds.get((k, p)) for p in feats} if preds else None
        w, prev = target_weights(feats, selection, sizing, preds=pk, prev=prev, day=k, **kw)
        turn = sum(abs(w.get(p, 0) - w_prev.get(p, 0)) for p in set(w) | set(w_prev))
        r = 0.0
        for p, wp in w.items():
            c0, c1 = panel[p].get(k), panel[p].get(k1)
            if c0 and c1:
                r += wp * (c1[0] / c0[0] - 1)
        rets.append(r - turn * cost_side)
        gross.append(sum(w.values()))
        w_prev = w
    return rets, gross


def _stats(rets):
    if not rets:
        return {"return_pct": None, "sharpe": None, "max_drawdown_pct": None, "days": 0}
    eq, peak, dd = 1.0, 1.0, 0.0
    for r in rets:
        eq *= 1 + r
        peak = max(peak, eq)
        dd = min(dd, eq / peak - 1)
    sd = statistics.pstdev(rets)
    return {"return_pct": round((eq - 1) * 100, 1),
            "sharpe": round(statistics.mean(rets) / sd * math.sqrt(365), 2) if sd > 0 else 0.0,
            "max_drawdown_pct": round(dd * 100, 1), "days": len(rets)}


def run_lab(candles_by_product, sma_days=100, top_k=5, vol_target=0.5, cost_side=0.006,
            min_train_days=365, progress=None):
    say = progress or (lambda m: None)
    days, panel = daily_panel(candles_by_product)
    if len(panel) < 5 or len(days) < min_train_days + 200:
        return {"ok": False, "error": f"need >=5 coins and {min_train_days + 200} days of "
                                      f"daily history, have {len(panel)} / {len(days)}"}
    say(f"features: {len(panel)} coins x {len(days)} days")
    fbd = _features_by_day(days, panel, sma_days)
    labels = _labels(days, panel)
    say("walk-forward rank model")
    preds = walk_forward_preds(fbd, labels, min_train_days=min_train_days, progress=say)
    oos = sorted({k for k, _ in preds})
    if not oos:
        return {"ok": False, "error": "rank model never trained (too little history)"}
    start = oos[0]
    kw = dict(sma_days=sma_days, top_k=top_k, vol_target=vol_target)
    from ..data.store import fingerprint
    out = {"ok": True, "ran_at": time.time(), "data_version": fingerprint(candles_by_product),
           "coins": len(panel),
           "from_day": start, "to_day": days[-1],
           "oos_days": len([k for k in days if k >= start]),
           "params": dict(kw, cost_side=cost_side, horizon_days=H,
                          rebalance_days=REBAL_DAYS, min_train_days=min_train_days),
           "variants": {}}
    for sel in SELECTIONS:
        for siz in SIZINGS:
            say(f"variant {sel}+{siz}")
            rets, gross = simulate(fbd, panel, sel, siz, preds if sel == "rank" else None,
                                   start_day=start, cost_side=cost_side, **kw)
            h = len(rets) // 2
            out["variants"][f"{sel}+{siz}"] = {
                "full": _stats(rets), "first_half": _stats(rets[:h]),
                "second_half": _stats(rets[h:]),
                "avg_invested_pct": round(100 * statistics.mean(gross), 1) if gross else 0}
    bh, _ = _buy_hold(fbd, panel, start, cost_side)
    h = len(bh) // 2
    out["buy_hold_equal"] = {"full": _stats(bh), "first_half": _stats(bh[:h]),
                             "second_half": _stats(bh[h:])}
    out["decision"] = decide(out["variants"])
    return out


def _buy_hold(fbd, panel, start, cost_side):
    rets, w_prev = [], {}
    for i in range(len(fbd) - 1):
        k, feats = fbd[i]
        if k < start:
            continue
        k1 = fbd[i + 1][0]
        live = [p for p in feats if p in panel and k1 in panel[p]]
        w = {p: 1.0 / len(live) for p in live} if live else {}
        turn = sum(abs(w.get(p, 0) - w_prev.get(p, 0)) for p in set(w) | set(w_prev))
        rets.append(sum(wp * (panel[p][k1][0] / panel[p][k][0] - 1) for p, wp in w.items())
                    - turn * cost_side)
        w_prev = w
    return rets, None


def _beats(a, b):
    return all(a[h]["sharpe"] is not None and b[h]["sharpe"] is not None
               and a[h]["sharpe"] > b[h]["sharpe"] for h in ("first_half", "second_half"))


def decide(variants, base="trend+equal"):
    """The variant the live core should use in "auto": the best-Sharpe one
    that beats the current rule in both halves (rank must also beat momentum
    with the same sizing); else the current rule."""
    b = variants[base]
    ok = []
    for name, v in variants.items():
        if name == base or not _beats(v, b):
            continue
        sel, siz = name.split("+")
        if sel == "rank" and not _beats(v, variants[f"momentum+{siz}"]):
            continue
        ok.append(name)
    pick = max(ok, key=lambda n: variants[n]["full"]["sharpe"]) if ok else base
    sel, siz = pick.split("+")
    return {"selection": sel, "sizing": siz, "variant": pick,
            "passed": sorted(ok), "why": ("beats trend+equal on Sharpe in both halves"
                                          if ok else "nothing beat trend+equal in both halves")}


# ------------------------------------------------------------------ live model
def train_live_model(candles_by_product):
    """Fit the rank model on ALL labelled daily history (for live use)."""
    days, panel = daily_panel(candles_by_product)
    fbd = _features_by_day(days, panel)
    labels = _labels(days, panel)
    X = [x_row(f) for k, feats in fbd for p, f in feats.items() if (k, p) in labels]
    y = [labels[(k, p)] for k, feats in fbd for p in feats if (k, p) in labels]
    return fit_model(X, y)


def save_model(model, path):
    with open(path + ".tmp", "wb") as f:
        pickle.dump({"model": model, "features": FEATURES, "saved_at": time.time()}, f)
    import os
    os.replace(path + ".tmp", path)


def load_model(path):
    """Load a model this lab saved (a local file only this app writes)."""
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception:
        return None
    return d.get("model") if d.get("features") == FEATURES else None
