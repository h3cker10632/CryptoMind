"""Strategy replay — bar-by-bar backtest of the LIVE signal engine.

Unlike `composite.py` (which reconstructs an older 3-sleeve ensemble), this
drives `app.signals.engine.SignalEngine.compute` itself in OFFLINE mode over
hourly history, one bar at a time with no look-ahead, and simulates execution
with the paper broker's rules: fee + slippage on both fills, ATR stop / target
/ trailing stop, peak-giveback, minimum hold, re-entry cooldown, entry caps,
position / exposure caps and the cost-viability gate. So when the live
strategy, its tunables or the disabled-sleeve list change, the replay tests
exactly that.

Live-only sleeves (llm, ml, model, evolved, discovered, derivatives) are
neutral here: their only inputs are present-day opinions, which would leak the
future into an old bar. The report says so.

Used by
  * the orchestrator's `replay_loop` (automatic, daily by default), and
  * tools/replay_backtest.py (manual CLI, with what-if overrides).
"""
from __future__ import annotations
import glob
import json
import os
import statistics
import time
from collections import Counter

# sleeves whose vote is a pure function of OHLCV history (safe to replay)
HISTORY_SLEEVES = ("trend_slow", "breakout_slow", "xsmom",
                   "trend", "breakout", "meanrev")

# tunables the simulation reads (all overridable for what-if runs)
_PARAM_KEYS = ("fee_rate", "slippage_bps", "stop_atr_mult", "take_profit_atr_mult",
               "trail_atr_mult", "min_confidence", "min_hold_sec", "cooldown_sec",
               "trail_giveback_arm_pct", "trail_giveback_pct", "max_open_positions",
               "max_entries_per_hour", "min_price", "max_gross_exposure",
               "cost_multiple", "risk_per_trade", "max_position_pct",
               "mtf_veto_align", "maker_fee_rate", "maker_timeout_sec",
               "chop_er_min", "chop_lookback_bars", "filter_strictness")

START_CASH = 100_000.0
MIN_BARS = 100            # warm-up bars before the first decision


def _params(overrides=None):
    from ..tunables import tv
    p = {k: tv(k) for k in _PARAM_KEYS}
    p.update({k: v for k, v in (overrides or {}).items() if k in p})
    return p


def load_history(granularity=3600, days=None, products=None, as_of=None, root=None):
    """({product: candles}, data_version) for offline backtests.

    Reads the versioned data store (app/data/store.py) — closed bars only,
    reproducible via `as_of` and the returned fingerprint. Defaults match what
    the live replay tests: the coins the bot has traded (its legacy cache
    list) over the last `replay_history_days`. Falls back to the legacy JSON
    cache when the store is empty."""
    from ..data import store
    if products is None:
        products = _legacy_products(root, granularity) or None
    if days is None:
        try:
            from .. import settings
            days = int(settings.get("replay_history_days"))
        except Exception:
            days = 365
    end = as_of if as_of is not None else time.time()
    start = None if days <= 0 else int(end - days * 86400) // granularity * granularity
    data, ver = store.load_candles(granularity, products, start=start, as_of=as_of)
    if data:
        return data, ver
    legacy = load_cached_history(root, granularity, _store=False)
    return legacy, store.fingerprint(legacy)


def _legacy_products(root, granularity):
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return sorted({os.path.basename(f).split("_")[0] for f in
                   glob.glob(os.path.join(root, ".cache", "history", f"*_{granularity}_*.json"))})


def load_cached_history(root=None, granularity=3600, _store=True):
    """{product: candles}: the data store when it has data (see
    `load_history`), else the on-disk legacy cache (.cache/history)."""
    if _store:
        data, _ = load_history(granularity, root=root)
        if data:
            return data
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = {}
    for f in glob.glob(os.path.join(root, ".cache", "history", f"*_{granularity}_*.json")):
        p = os.path.basename(f).split("_")[0]
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        if len(rows) > len(out.get(p, [])):
            out[p] = rows
    return out


def _clean(candles_by_product):
    out = {}
    for p, rows in candles_by_product.items():
        dedup = {}
        for r in rows or []:
            try:
                r = [float(x) for x in r[:6]]
            except (TypeError, ValueError):
                continue
            if r[4] > 0:
                dedup[int(r[0])] = r
        if dedup:
            out[p] = [dedup[t] for t in sorted(dedup)]
    return out


class _HistMarket:
    """Minimal market view at one historical bar: the same features the live
    feed computes (shared OHLCV core + native ATR + multi-timeframe trend +
    cross-sectional momentum), built only from candles up to that bar."""

    def __init__(self, C, pos_of, history_bars):
        self.C, self.pos_of, self.n = C, pos_of, history_bars
        self.candles, self._cache, self._xs = {}, {}, {}

    def set_time(self, t):
        from ..data.market import MarketData
        self.candles, self._cache = {}, {}
        for p, rows in self.C.items():
            i = self.pos_of[p].get(t)
            if i is not None:
                self.candles[p] = rows[max(0, i - self.n + 1): i + 1]
        self._xs = MarketData.xs_momentum(self)

    def xs_momentum(self, lookback=72):
        return self._xs

    def features(self, p):
        if p in self._cache:
            return self._cache[p]
        from ..data.features import features_from_ohlcv
        from ..data.market import MarketData
        cs = self.candles.get(p)
        f = None
        if cs and len(cs) >= 60:
            cl = [c[4] for c in cs]; hi = [c[2] for c in cs]
            lo = [c[1] for c in cs]; vo = [c[5] for c in cs]
            f = features_from_ohlcv(cl, hi, lo, vo)
            if f is not None:
                f["atr_swing"] = MarketData._swing_atr(hi, lo, cl)
                f["mtf_align"] = MarketData._mtf(hi, lo, cl)["align"]
                f["xs_mom"] = self._xs.get(p, 0.0)
                f["patterns"] = None
        self._cache[p] = f
        return f

    def regime(self):
        f = self.features("BTC-USD")
        if not f:
            return {"label": "unknown", "vol": 0}
        trend = ("bull" if f["sma20"] > f["sma50"] * 1.002 else
                 "bear" if f["sma20"] < f["sma50"] * 0.998 else "sideways")
        return {"label": f"{trend}/normal", "trend": trend}


class _NoNLP:
    market_sentiment = 0.0

    def asset_score(self, p):
        return (0.0, 0)


def run_replay(candles_by_product, start=0.0, end=1.0, overrides=None,
               shorts=None, min_coverage=0.7, history_bars=None,
               maker=None, chop=False, cache=None, filt=None, hooks=None,
               extras=False, persist=False):
    """Replay the live engine over [start, end) of the common history window
    (fractions of it). Returns a JSON-able result dict; never raises on bad
    data (returns {"ok": False, "error": ...}).

    maker — simulate post-only LIMIT entries (default: the live
            `entry_order_type` setting). A limit at the signal bar's close fills
            only if a LATER bar trades strictly through it within
            `maker_timeout_sec`; unfilled orders are dropped (missed trade, no
            fee). Maker-entered positions take profit at the target with the
            maker fee; every other exit is taker (fee + slippage).
    chop  — apply the chop filter (no new entries while market trendiness is
            below `chop_er_min`).
    filt  — optional {(bar_ts, product): (p_win, base_rate)} from the trade
            filter's walk-forward (out-of-sample) predictions: skip a signal
            whose p_win < base_rate x `filter_strictness`.
    cache — optional dict shared between calls on the SAME candles and signal
            parameters: per-bar signals don't depend on the portfolio, so
            run_report computes them once and reuses them for every window
            and A/B variant (~5x faster on a year of history).
    hooks — optional learner hooks (see app/backtest/ablation.py): re-mix the
            per-bar signals, add an exit, scale size, and learn from bars and
            closed trades as the replay walks forward. None = unchanged replay.
            With hooks the result also carries `_equity` / `_ts` (per bar).
    extras — also cache each bar's raw per-sleeve votes, regime, HTF alignment
            and online-model input vectors (needed by the learner hooks).
    persist — reuse per-bar data computed by earlier runs from the on-disk
            cache (app/backtest/bar_cache.py) wherever the candle rows a bar
            reads are unchanged, and save this run's bars for the next one.
            The result is identical to a fresh run (tests/test_speed_parity.py);
            only the bars that are new or whose inputs changed are computed."""
    from ..config import DISABLED_STRATEGIES, CANDLE_HISTORY
    from ..signals import engine as eng
    from .. import settings as app_settings
    from ..data.market import trendiness
    P = _params(overrides)
    if maker is None:
        try:
            maker = app_settings.get("entry_order_type") == "maker"
        except Exception:
            maker = False
    if shorts is None:
        try:
            shorts = bool(app_settings.get("allow_shorts"))
        except Exception:
            shorts = True
    only = tuple(s for s in HISTORY_SLEEVES if s not in DISABLED_STRATEGIES)
    offline = {"only": only, "gate": P["min_confidence"], "shorts": shorts,
               "veto_align": P["mtf_veto_align"]}
    chop_n = int(P["chop_lookback_bars"])
    hist_n = history_bars or CANDLE_HISTORY
    key = (id(candles_by_product), only, P["min_confidence"], bool(shorts),
           P["mtf_veto_align"], chop_n, hist_n, bool(extras))
    if cache is not None and cache.get("_key") == key:
        C, ts_all, pos_of, bar_cache = cache["C"], cache["ts_all"], cache["pos_of"], cache["bars"]
        i0 = max(MIN_BARS, int(len(ts_all) * start))
        i1 = max(i0 + 1, int(len(ts_all) * end))
        return _simulate(C, ts_all, pos_of, bar_cache, filt, i0, i1, P, maker, chop, offline,
                         chop_n, hist_n, only, hooks, extras)
    C = _clean(candles_by_product)
    if len(C) < 5:
        return {"ok": False, "error": f"need >=5 products with history, have {len(C)}"}
    # timestamps present for most products (drops each coin's ragged edges)
    cnt = Counter(t for rows in C.values() for t in (int(r[0]) for r in rows))
    need = max(5, int(min_coverage * len(C)))
    ts_all = sorted(t for t, k in cnt.items() if k >= need)
    if len(ts_all) < MIN_BARS + 48:
        return {"ok": False, "error": f"only {len(ts_all)} common bars of history"}
    pos_of = {p: {int(r[0]): i for i, r in enumerate(rows)} for p, rows in C.items()}
    i0 = max(MIN_BARS, int(len(ts_all) * start))
    i1 = max(i0 + 1, int(len(ts_all) * end))
    bar_cache, disk = {}, None
    if persist:
        from . import bar_cache as BC
        try:
            dkey = BC.cache_key(offline, chop_n, hist_n, extras)
            fps = BC.fingerprints(C, pos_of, ts_all[i0:i1], hist_n)
            bar_cache.update(BC.load(dkey, fps))
            disk = (dkey, fps, len(bar_cache))
        except Exception:
            bar_cache, disk = {}, None          # a cache problem never blocks a replay
    if cache is not None:
        cache.clear()
        cache.update(_key=key, C=C, ts_all=ts_all, pos_of=pos_of, bars=bar_cache)
    res = _simulate(C, ts_all, pos_of, bar_cache, filt, i0, i1, P, maker, chop, offline,
                    chop_n, hist_n, only, hooks, extras)
    if disk is not None:
        dkey, fps, reused = disk
        try:
            BC.save(dkey, bar_cache, fps)
        except Exception:
            pass
        if cache is not None:
            cache["disk"] = {"reused_bars": reused, "computed_bars": len(bar_cache) - reused}
    return res


def _bar_data(t, C, pos_of, mkt, E, nlp, offline, chop_n, extras=False):
    """Portfolio-independent data for bar t: ordered signal list, OHLC of the
    bar for every coin present, market trendiness, the trade-filter feature
    vector of every actionable signal and (with `extras`) the per-coin raw
    sleeve votes, regime, HTF alignment and online-model input vector."""
    from ..data.market import trendiness
    from ..learn.trade_filter import features as filter_features
    mkt.set_time(t)
    sigs = E.compute(mkt, nlp, record=False, offline=offline)
    sl = [(p, s["direction"], s["confidence"], s["actionable"], s["price"], s["atr"])
          for p, s in sigs.items()]
    bars = {p: (cs[-1][1], cs[-1][2], cs[-1][4]) for p, cs in mkt.candles.items()
            if int(cs[-1][0]) == t}
    trend = trendiness(mkt.candles, chop_n)
    btc = mkt.features("BTC-USD")
    fx = {p: filter_features(mkt.features(p), s["direction"], s["confidence"], btc, trend)
          for p, s in sigs.items() if s["actionable"]}
    ext = None
    if extras:
        from ..learn.online_model import build_x
        reg = mkt.regime()
        ext = {"regime": reg.get("label", "unknown"), "trend": reg.get("trend", "sideways"),
               "raw": {p: {k: v for k, v in E.per_strategy.get(p, {}).items() if v}
                       for p in sigs},
               "mtf": {p: s["mtf_align"] for p, s in sigs.items()},
               "x": {p: tuple(build_x(mkt.features(p), 0.0, 0.0, None)) for p in sigs}}
    return sl, bars, trend, fx, ext


def _simulate(C, ts_all, pos_of, bar_cache, filt, i0, i1, P, maker, chop, offline,
              chop_n, hist_n, only, hooks=None, extras=False):
    from ..signals import engine as eng
    mkt = _HistMarket(C, pos_of, hist_n)
    E = eng.SignalEngine()
    nlp = _NoNLP()
    saved_products = list(eng.PRODUCTS)
    eng.PRODUCTS = sorted(C)       # replay the whole cached universe
    taker_cost = P["fee_rate"] + P["slippage_bps"] / 1e4
    maker_cost = P["maker_fee_rate"]
    entry_cost = maker_cost if maker else taker_cost
    timeout_bars = max(1, int(round(P["maker_timeout_sec"] / 3600)))
    cash, book, trades, eq, eq_ts = START_CASH, {}, [], [], []
    last_exit, entries, pending = {}, [], {}
    n_limits = n_filled = chop_blocked_bars = n_filtered = 0
    try:
        for t in ts_all[i0:i1]:
            bd = bar_cache.get(t)
            if bd is None:
                bd = bar_cache[t] = _bar_data(t, C, pos_of, mkt, E, nlp, offline, chop_n,
                                              extras)
            sl, bars, trend_now = bd[0], bd[1], bd[2]
            if hooks is not None:
                sl = hooks.signals(t, bd)
            sigs = {x[0]: x for x in sl}
            px = {p: b[2] for p, b in bars.items()}
            # ---- resting limits: fill if THIS bar traded strictly through
            for p, o in list(pending.items()):
                bar = bars.get(p)
                if bar and (
                        (o["side"] > 0 and bar[0] < o["e"]) or
                        (o["side"] < 0 and bar[1] > o["e"])):
                    del pending[p]
                    if p in book or o["n"] > cash:
                        continue
                    cash -= o["n"]
                    n_filled += 1
                    book[p] = {"e": o["e"], "n": o["n"], "side": o["side"],
                               "w": o["e"], "t": t, "stop": o["stop"],
                               "tp": o["tp"], "maker": True, "meta": o.get("meta")}
                elif t - o["placed"] >= timeout_bars * 3600:
                    del pending[p]
            # ---- manage: hard stop / target / trail / giveback, then flips
            for p, b in list(book.items()):
                if p not in px:
                    continue
                x, s = px[p], b["side"]
                sg = sigs.get(p)
                atr = sg[5] if sg else None
                b["w"] = max(b["w"], x) if s > 0 else min(b["w"], x)
                if atr:
                    tr = b["w"] - s * P["trail_atr_mult"] * atr
                    b["stop"] = max(b["stop"], tr) if s > 0 else min(b["stop"], tr)
                held = t - b["t"]
                why = None
                if (x - b["stop"]) * s <= 0:
                    why = "stop"
                elif (x - b["tp"]) * s >= 0:
                    why = "take-profit"
                else:
                    peak = (b["w"] / b["e"] - 1) * s
                    cur = (x / b["e"] - 1) * s
                    if (held >= P["min_hold_sec"] and P["trail_giveback_pct"] > 0
                            and peak >= P["trail_giveback_arm_pct"]
                            and cur <= peak * (1 - P["trail_giveback_pct"])):
                        why = "peak-giveback"
                    if (not why and sg and held >= P["min_hold_sec"]
                            and sg[1] * s < 0 and sg[2] > 0.5):
                        why = "signal flip"
                    if not why and hooks is not None and held >= P["min_hold_sec"]:
                        why = hooks.exit(t, p, b, x, atr)
                if why:
                    if why == "take-profit" and b.get("maker"):
                        exit_px, exit_cost = b["tp"], maker_cost   # resting limit
                    else:
                        exit_px, exit_cost = x, taker_cost
                    net = (exit_px / b["e"] - 1) * s - (b.get("ecost", entry_cost) + exit_cost)
                    cash += b["n"] * (1 + net)
                    trades.append({"product": p, "side": s, "net": net,
                                   "usd": b["n"] * net, "exit": why,
                                   "hold_h": held / 3600})
                    if hooks is not None:
                        hooks.on_close(t, p, b, net, why)
                    del book[p]
                    last_exit[p] = t
            eqv = cash + sum(b["n"] * (1 + (px.get(p, b["e"]) / b["e"] - 1) * b["side"])
                             for p, b in book.items())
            eq.append(eqv)
            eq_ts.append(t)
            if hooks is not None:
                hooks.on_bar(t, eqv, book)
            # ---- entries
            entries = [e for e in entries if t - e < 3600]
            if chop and trend_now is not None and trend_now < P["chop_er_min"]:
                chop_blocked_bars += 1
                continue
            for sg in sorted((x for x in sl if x[3]), key=lambda x: -x[2]):
                p, s_dir, s_conf, _, s_price, s_atr = sg
                if (p in book or p in pending
                        or len(book) + len(pending) >= P["max_open_positions"]
                        or t - last_exit.get(p, -1e12) < P["cooldown_sec"]):
                    continue
                if len(entries) >= P["max_entries_per_hour"]:
                    break
                if s_price < P["min_price"]:
                    continue
                if filt is not None:
                    pr = filt.get((t, p))
                    if pr is not None and pr[0] < pr[1] * P["filter_strictness"]:
                        n_filtered += 1
                        continue
                if sum(b["n"] for b in book.values()) / eqv >= P["max_gross_exposure"]:
                    break
                atr, x = s_atr, s_price
                if not atr or atr <= 0:
                    continue
                if P["take_profit_atr_mult"] * atr < (entry_cost + taker_cost) * P["cost_multiple"] * x:
                    continue                              # live cost-viability gate
                mult = hooks.size_mult(t) if hooks is not None else 1.0
                n = min(eqv * P["risk_per_trade"] * (0.5 + s_conf / 2) * mult
                        / (P["stop_atr_mult"] * atr / x),
                        eqv * P["max_position_pct"])
                if n < 10 or n > cash:
                    continue
                s = s_dir
                entries.append(t)
                meta = hooks.on_entry(t, p, s) if hooks is not None else None
                stop = x - s * P["stop_atr_mult"] * atr
                tp = x + s * P["take_profit_atr_mult"] * atr
                if maker:
                    n_limits += 1
                    pending[p] = {"e": x, "n": n, "side": s, "stop": stop,
                                  "tp": tp, "placed": t, "meta": meta}
                    continue
                cash -= n
                book[p] = {"e": x, "n": n, "side": s, "w": x, "t": t,
                           "stop": stop, "tp": tp, "ecost": taker_cost, "meta": meta}
    finally:
        eng.PRODUCTS = saved_products

    final = eq[-1] if eq else START_CASH
    peak, maxdd = 0.0, 0.0
    for v in eq:
        peak = max(peak, v)
        maxdd = min(maxdd, v / peak - 1)
    bh = [C[p][pos_of[p][ts_all[i1 - 1]]][4] / C[p][pos_of[p][ts_all[i0]]][4] - 1
          for p in C if ts_all[i0] in pos_of[p] and ts_all[i1 - 1] in pos_of[p]]
    n = len(trades)
    extra = {"_equity": eq, "_ts": eq_ts} if hooks is not None else {}
    exits = {}
    for k in sorted({x["exit"] for x in trades}):
        sub = [x["net"] for x in trades if x["exit"] == k]
        exits[k] = {"n": len(sub), "avg_net_bps": round(1e4 * statistics.mean(sub), 1)}
    return {
        "ok": True,
        "from_ts": ts_all[i0], "to_ts": ts_all[i1 - 1],
        "days": round((ts_all[i1 - 1] - ts_all[i0]) / 86400, 1),
        "products": len(C),
        "return_pct": round((final / START_CASH - 1) * 100, 2),
        "max_drawdown_pct": round(maxdd * 100, 1),
        "trades": n,
        "win_rate_pct": round(100 * sum(1 for x in trades if x["net"] > 0) / max(n, 1), 1),
        "avg_net_bps": round(1e4 * sum(x["net"] for x in trades) / max(n, 1), 1),
        "median_hold_h": round(statistics.median([x["hold_h"] for x in trades]), 1) if trades else 0,
        "longs": sum(1 for x in trades if x["side"] > 0),
        "shorts": sum(1 for x in trades if x["side"] < 0),
        "exits": exits,
        "buy_hold_equal_weight_pct": round(100 * statistics.mean(bh), 1) if bh else None,
        "entry_orders": "limit (maker)" if maker else "market (taker)",
        "round_trip_cost_pct": round(100 * (entry_cost + taker_cost), 2),
        "limit_orders": n_limits, "limit_fill_rate_pct":
            round(100 * n_filled / n_limits, 1) if n_limits else None,
        "chop_filter": bool(chop),
        "trade_filter": filt is not None, "filtered_signals": n_filtered,
        "chop_blocked_pct": round(100 * chop_blocked_bars / max(1, i1 - i0), 1),
        "sleeves_replayed": list(only),
        "note": ("Live-only sleeves (llm, ml, model, evolved, discovered, "
                 "derivatives) are neutral in replay."),
        "_trades": trades,
        **extra,
    }


def filter_samples(cache, overrides=None, maker=None):
    """Trade-filter training samples from a populated replay cache: one per
    (bar, coin) actionable signal, labelled with the NET return the bot's own
    exits would have produced (see trade_filter.triple_barrier_net)."""
    from ..learn.trade_filter import triple_barrier_net
    from .. import settings as app_settings
    P = _params(overrides)
    if maker is None:
        try:
            maker = app_settings.get("entry_order_type") == "maker"
        except Exception:
            maker = False
    taker = P["fee_rate"] + P["slippage_bps"] / 1e4
    entry = P["maker_fee_rate"] if maker else taker
    tp_cost = P["maker_fee_rate"] if maker else taker
    C, pos_of = cache["C"], cache["pos_of"]
    closes = {p: [r[4] for r in rows] for p, rows in C.items()}
    out = []
    for t, bd in cache["bars"].items():
        sl, fx = bd[0], bd[3]
        sig = {x[0]: x for x in sl}
        for p, x in fx.items():
            sg = sig.get(p)
            i = pos_of[p].get(t)
            if sg is None or i is None:
                continue
            net, held = triple_barrier_net(closes[p], i, sg[1], sg[5],
                                           P["stop_atr_mult"], P["take_profit_atr_mult"],
                                           P["trail_atr_mult"], entry, taker, tp_cost)
            if net is None:
                continue
            out.append({"t": t, "product": p, "x": x, "net": net,
                        "exit_t": C[p][i + held][0]})
    return out


def run_report(candles_by_product, overrides=None, shorts=None, chop=False,
               chop_ab=True, filt_on=False, filter_ab=True):
    """Full window plus first/second-half split (a strategy that only works in
    one half is regime luck, not an edge), per-coin attribution, and A/B
    comparisons that let the bot decide from evidence whether an optional
    layer earns its keep:
      chop_ab   — the same replay with the chop filter flipped;
      filter_ab — the same replay with the trade filter flipped, using
                  walk-forward OUT-OF-SAMPLE filter probabilities.
    `_filter_samples` (popped by the caller) carries the labelled history the
    live trade filter is trained on."""
    from ..learn.trade_filter import walk_forward_probs
    windows = (("full", (0.0, 1.0)), ("first_half", (0.0, 0.55)),
               ("second_half", (0.45, 1.0)))
    from ..data.store import fingerprint
    out = {"ran_at": time.time(), "data_version": fingerprint(candles_by_product)}
    cache = {}
    # populate the per-bar cache once (signals don't depend on the portfolio)
    warm = run_replay(candles_by_product, 0.0, 1.0, overrides=overrides,
                      shorts=shorts, cache=cache, persist=True)
    probs = None
    if warm.get("ok") and (filt_on or filter_ab):
        samples = filter_samples(cache, overrides)
        pr, bases = walk_forward_probs(samples)
        probs = {k: (v, bases[k]) for k, v in pr.items()}
        out["_filter_samples"] = samples
        wins = sum(1 for x in samples if x["net"] > 0)
        out["filter_data"] = {"samples": len(samples),
                              "base_win_rate": round(wins / len(samples), 3) if samples else None,
                              "out_of_sample_coverage": round(len(probs) / len(samples), 3)
                              if samples else None}

    def windows_for(chop_flag, filt_flag):
        res, trades = {}, []
        for name, (a, b) in windows:
            r = run_replay(candles_by_product, a, b, overrides=overrides, shorts=shorts,
                           chop=chop_flag, cache=cache,
                           filt=probs if (filt_flag and probs is not None) else None)
            tr = r.pop("_trades", None) or []
            if name == "full":
                trades = tr
            res[name] = r
        return res, trades

    base, base_trades = windows_for(chop, filt_on and probs is not None)
    out.update(base)
    out["per_product"] = per_product(base_trades, candles_by_product)
    f, h1, h2 = out["full"], out["first_half"], out["second_half"]
    if f.get("ok"):
        both = h1.get("ok") and h2.get("ok") and h1["return_pct"] > 0 and h2["return_pct"] > 0
        out["verdict"] = ("positive in both halves" if both else
                          "positive overall, but not in both halves" if f["return_pct"] > 0
                          else "negative")
    else:
        out["verdict"] = "no result"

    def ab(on, off):
        def ret(d, k):
            return (d.get(k) or {}).get("return_pct")
        ok = all((on.get(k) or {}).get("ok") and (off.get(k) or {}).get("ok")
                 for k in ("first_half", "second_half"))
        return {"off": {k: ret(off, k) for k, _ in windows},
                "on": {k: ret(on, k) for k, _ in windows},
                "helps_in_both_halves": bool(ok and all(
                    ret(on, k) > ret(off, k) for k in ("first_half", "second_half")))}

    if chop_ab and f.get("ok"):
        other, _ = windows_for(not chop, filt_on and probs is not None)
        out["chop_ab"] = ab(other, base) if not chop else ab(base, other)
    try:
        out["slow_benchmarks"] = slow_benchmarks(candles_by_product, overrides)
    except Exception as e:                       # never let a benchmark break the report
        out["slow_benchmarks"] = {"error": str(e)}
    if filter_ab and f.get("ok") and probs:
        other, _ = windows_for(chop, not filt_on)
        out["filter_ab"] = ab(other, base) if not filt_on else ab(base, other)
        on_full = (other if not filt_on else base)["full"]
        out["filter_ab"]["filtered_signals"] = on_full.get("filtered_signals")
    return out


def per_product(trades, candles_by_product=None):
    """Per-coin attribution of a universe replay: every coin in the universe
    gets a row (0 trades if the strategy never fired on it), plus how many
    hourly bars of history it had."""
    rows = {}
    for p in sorted(candles_by_product or {}):
        rows[p] = {"trades": 0, "net_usd": 0.0, "avg_net_bps": None,
                   "win_rate_pct": None,
                   "history_bars": len(candles_by_product[p] or [])}
    for x in trades:
        r = rows.setdefault(x["product"], {"trades": 0, "net_usd": 0.0,
                                           "history_bars": None})
        r["trades"] += 1
        r["net_usd"] += x["usd"]
        r.setdefault("_nets", []).append(x["net"])
    for r in rows.values():
        nets = r.pop("_nets", [])
        r["net_usd"] = round(r["net_usd"], 2)
        if nets:
            r["avg_net_bps"] = round(1e4 * statistics.mean(nets), 1)
            r["win_rate_pct"] = round(100 * sum(1 for v in nets if v > 0) / len(nets), 1)
    return rows


# ---------------------------------------------------------------- slow rules
def _daily_closes(candles_by_product):
    """{product: {day_index: last close of that UTC day}} from hourly bars,
    dropping the current (incomplete) day."""
    out = {}
    for p, rows in _clean(candles_by_product).items():
        d = {}
        for r in rows:
            d[int(r[0]) // 86400] = r[4]
        if d:
            d.pop(max(d), None)
        out[p] = d
    return out


def _run_daily(days, closes, weight_fn, cost_side, warmup):
    """Daily simulation: weights chosen at day d's close earn day d+1's return;
    turnover pays `cost_side` per unit traded. Returns the daily return list."""
    rets, w_prev = [], {}
    for k in range(warmup, len(days) - 1):
        d, d1 = days[k], days[k + 1]
        w = weight_fn(k)
        turn = sum(abs(w.get(p, 0) - w_prev.get(p, 0)) for p in set(w) | set(w_prev))
        r = 0.0
        for p, wp in w.items():
            c0, c1 = closes[p].get(d), closes[p].get(d1)
            if wp and c0 and c1:
                r += wp * (c1 / c0 - 1)
        rets.append(r - turn * cost_side)
        w_prev = w
    return rets


def _summ(rets):
    def comp(xs):
        e = 1.0
        for x in xs:
            e *= 1 + x
        return round((e - 1) * 100, 1)
    eq, peak, dd = 1.0, 1.0, 0.0
    for x in rets:
        eq *= 1 + x
        peak = max(peak, eq)
        dd = min(dd, eq / peak - 1)
    h = len(rets) // 2
    return {"return_pct": comp(rets), "first_half_pct": comp(rets[:h]),
            "second_half_pct": comp(rets[h:]), "max_drawdown_pct": round(dd * 100, 1)}


def slow_benchmarks(candles_by_product, overrides=None, sma_slow=100, core_sma=None,
                    core_assets=("BTC-USD", "ETH-USD")):
    """The low-turnover daily rules the hourly bot must beat to earn its keep:
      per_coin_trend — hold each coin (equal slot) only while its daily close
                       is above its `sma_slow`-day average, else cash;
      core_trend     — BTC/ETH 50/50, each only while above its
                       `core_sma`-day average (the optional core holding);
      buy_hold       — equal-weight buy & hold of every coin.
    Same market-order cost per side as the bot. Note: the universe is TODAY's,
    so the all-coin rows carry survivorship bias."""
    P = _params(overrides)
    cost_side = P["fee_rate"] + P["slippage_bps"] / 1e4
    if core_sma is None:
        try:
            from .. import settings
            core_sma = int(settings.get("core_sma_days"))
        except Exception:
            core_sma = 50
    closes = _daily_closes(candles_by_product)
    if len(closes) < 3:
        return None
    days = sorted(set(d for c in closes.values() for d in c))
    warm = max(sma_slow, core_sma)
    if len(days) < warm + 30:
        return None
    coins = sorted(closes)
    cache = {}

    def sma(p, k, n):
        key = (p, k, n)
        if key not in cache:
            vals = [closes[p].get(days[j]) for j in range(k - n + 1, k + 1)]
            cache[key] = None if any(v is None for v in vals) else sum(vals) / n
        return cache[key]

    def above(p, k, n):
        c = closes[p].get(days[k])
        m = sma(p, k, n)
        return c is not None and m is not None and c > m

    def per_coin(k):
        return {p: 1.0 / len(coins) for p in coins if above(p, k, sma_slow)}

    core = [a for a in core_assets if a in closes]

    def core_rule(k):
        return {a: 1.0 / len(core) for a in core if above(a, k, core_sma)} if core else {}

    def buy_hold(k):
        live = [p for p in coins if closes[p].get(days[k])]
        return {p: 1.0 / len(live) for p in live} if live else {}

    return {"days": len(days) - warm - 1,
            f"per_coin_trend_{sma_slow}d": _summ(_run_daily(days, closes, per_coin, cost_side, warm)),
            f"btc_eth_core_{core_sma}d": _summ(_run_daily(days, closes, core_rule, cost_side, warm)),
            "buy_hold_equal_weight": _summ(_run_daily(days, closes, buy_hold, cost_side, warm)),
            "note": "today's universe -> all-coin rows are survivorship-biased"}
