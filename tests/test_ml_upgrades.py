"""ML upgrades: learned trade filter (labels, walk-forward, gate), replay
filter A/B, online-model warm start, masked inputs, tradeable-move grading."""
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _candles(n=500, seed=0, drift=0.0004, vol=0.012, t0=1_700_000_000):
    r = random.Random(seed)
    px, out = 100.0, []
    for i in range(n):
        op = px
        px *= math.exp(drift + r.gauss(0, vol))
        out.append([t0 + i * 3600, min(op, px) * 0.996, max(op, px) * 1.004, op, px, 1000.0])
    return out


def _universe(k=7, n=500):
    names = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "DOGE-USD", "XRP-USD"][:k]
    return {p: _candles(n, seed=i, drift=0.0008 * (i - k / 2) / k) for i, p in enumerate(names)}


# ------------------------------------------------------------ labels
def test_triple_barrier_take_profit_stop_and_open():
    from app.learn.trade_filter import triple_barrier_net
    up = [100 + i for i in range(30)]
    net, held = triple_barrier_net(up, 0, 1, 1.0, 3, 6, 3, 0.004, 0.006, 0.004)
    assert held == 6 and abs(net - (0.06 - 0.008)) < 1e-12        # TP at +6 ATR, maker exit
    down = [100 - i for i in range(30)]
    net, held = triple_barrier_net(down, 0, 1, 1.0, 3, 6, 3, 0.004, 0.006, 0.004)
    assert held == 3 and net < -0.03                              # stop
    net, held = triple_barrier_net(down, 0, -1, 1.0, 3, 6, 3, 0.004, 0.006, 0.004)
    assert net > 0                                                # short wins on the drop
    flat = [100.0] * 5
    assert triple_barrier_net(flat, 0, 1, 1.0, 3, 6, 3, 0, 0, 0) == (None, None)  # unresolved


def test_features_are_direction_aligned():
    from app.learn.trade_filter import features, FEATURE_NAMES
    f = {"price": 100, "ema24": 102, "ema96": 100, "mom_72": 0.05, "mom_4h": 0.02,
         "mom_1h": 0.01, "hi48": 99, "lo48": 90, "xs_mom": 1.0, "mtf_align": 1.0,
         "atr_swing": 1.0, "volatility": 0.01, "rsi": 70, "vol_ratio": 1.2}
    L, S = features(f, 1, 0.7), features(f, -1, 0.7)
    assert len(L) == len(FEATURE_NAMES)
    i = FEATURE_NAMES.index("trend_24_96")
    assert L[i] > 0 and S[i] == -L[i]
    j = FEATURE_NAMES.index("atr_pct")
    assert L[j] == S[j]                                           # non-directional


# ------------------------------------------------------------ walk-forward
def _synthetic_samples(n=3000, learnable=True, seed=1):
    r = random.Random(seed)
    out, t0 = [], 1_700_000_000
    for k in range(n):
        x = [r.gauss(0, 1) for _ in range(15)]
        edge = 0.02 * x[2] if learnable else 0.0
        net = edge + r.gauss(-0.005, 0.01)
        t = t0 + k * 900
        out.append({"t": t, "product": "P%d" % (k % 5), "x": x, "net": net,
                    "exit_t": t + 24 * 3600})
    return out


def test_walk_forward_has_no_leakage_and_finds_real_signal():
    from app.learn import trade_filter as tf
    S = _synthetic_samples()
    probs, bases = tf.walk_forward_probs(S, fold_sec=5 * 86400)
    assert probs and set(probs) == set(bases)
    first_t = min(s["t"] for s in S)
    assert all(t >= first_t + 5 * 86400 for t, _ in probs)       # first fold never predicted
    take = [s["net"] for s in S if (s["t"], s["product"]) in probs
            and probs[(s["t"], s["product"])] >= bases[(s["t"], s["product"])]]
    skip = [s["net"] for s in S if (s["t"], s["product"]) in probs
            and probs[(s["t"], s["product"])] < bases[(s["t"], s["product"])]]
    assert sum(take) / len(take) > sum(skip) / len(skip)          # real signal separates


def test_numpy_fallback_model_works():
    from app.learn.trade_filter import _BinnedLogit
    r = random.Random(0)
    X = [[r.gauss(0, 1), r.gauss(0, 1)] for _ in range(2000)]
    y = [1 if x[0] + 0.3 * r.gauss(0, 1) > 0 else 0 for x in X]
    m = _BinnedLogit().fit(X, y)
    p = m.predict_proba([[2.0, 0.0], [-2.0, 0.0]])[:, 1]
    assert p[0] > 0.8 and p[1] < 0.2


def test_live_filter_gate_modes(monkeypatch):
    from app.learn.trade_filter import TradeFilter
    from app import settings
    S = _synthetic_samples()
    tf = TradeFilter()
    assert tf.allows([0.0] * 15) == (True, None)                 # untrained -> allow
    assert tf.fit(S, now=time.time() * 10)
    real = settings.get
    mode = {"v": "auto"}
    monkeypatch.setattr(settings, "get", lambda k: mode["v"] if k == "trade_filter_mode" else real(k))
    assert not tf.active()                                        # auto, not proven yet
    tf.auto_on = True
    assert tf.active()
    good, bad = [0.0] * 15, [0.0] * 15
    good[2], bad[2] = 3.0, -3.0
    assert tf.allows(good)[0] and not tf.allows(bad)[0]
    mode["v"] = "off"
    assert tf.allows(bad) == (True, None)


def test_fit_uses_only_resolved_samples():
    from app.learn.trade_filter import TradeFilter
    S = _synthetic_samples(400)
    tf = TradeFilter()
    cutoff = S[350]["exit_t"]
    tf.fit(S, now=cutoff)
    assert tf.n_train == sum(1 for s in S if s["exit_t"] <= cutoff)


# ------------------------------------------------------------ replay A/B
def test_report_includes_filter_data_and_ab():
    from app.backtest import replay as rp
    rep = rp.run_report(_universe(7, 1600))
    assert rep["filter_data"]["samples"] > 0
    samples = rep.pop("_filter_samples")
    assert all({"t", "product", "x", "net", "exit_t"} <= set(s) for s in samples)
    assert all(s["exit_t"] > s["t"] for s in samples)
    if "filter_ab" in rep:
        ab = rep["filter_ab"]
        assert ab["off"]["full"] == rep["full"]["return_pct"]    # base = filter off
        assert isinstance(ab["helps_in_both_halves"], bool)


def test_replay_filter_skips_low_probability_signals():
    from app.backtest import replay as rp
    U = _universe(7, 600)
    cache = {}
    base = rp.run_replay(U, cache=cache)
    keys = {(t, p) for t, bd in cache["bars"].items() for p in bd[3]}
    block_all = {k: (0.0, 0.5) for k in keys}
    r = rp.run_replay(U, cache=cache, filt=block_all)
    assert r["trades"] == 0 and r["filtered_signals"] > 0 and base["trades"] > 0


# ------------------------------------------------------------ online model
def test_masked_inputs_are_zero_and_width_unchanged():
    from app.learn.online_model import build_x, N_IN, FEAT_NAMES, MASKED_FEATURES
    f = {"rsi": 55, "macd": 0.1, "macd_delta": 0.02, "mom_1h": 0.001, "mom_4h": 0.002,
         "vol_ratio": 1.1, "imbalance": 0.9, "spread_bps": 30, "price": 100, "sma20": 99,
         "sma50": 98, "volatility": 0.003, "atr": 0.5}
    x = build_x(f, 0.8, -0.7, None)
    assert len(x) == N_IN
    for name in MASKED_FEATURES:
        assert x[FEAT_NAMES.index(name)] == 0.0
    assert x[FEAT_NAMES.index("rsi")] != 0.0


def test_accuracy_ignores_untradeable_moves():
    from app.learn.online_model import TinyMLP, N_IN, TARGET_SCALE
    m = TinyMLP()
    x = [0.1] * N_IN
    m.update(x, 0.1 * TARGET_SCALE, pred_at_record=0.5)          # tiny move: not graded
    assert len(m.acc_window) == 0
    m.update(x, 0.8 * TARGET_SCALE, pred_at_record=0.5)          # tradeable: graded
    assert list(m.acc_window) == [1]


def test_warm_start_trains_in_chronological_order():
    from app.learn.pretrain import pretrain_online
    from app.learn.online_model import Committee
    c = Committee(n_members=2)
    n = pretrain_online(_universe(5, 400), max_samples=600, committee=c, model=c.primary)
    assert 0 < n <= 600
    assert c.primary.n_updates == n
