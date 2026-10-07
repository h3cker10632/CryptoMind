"""Pooled ML pipeline: causal features, leak-free walk-forward, an honest
evaluation vs a baseline, candidate strategies, live hand-off, and the
Polymarket calibration."""
import math
import sys
import types

import numpy as np
import pytest

DAY = 86400


@pytest.fixture
def store_stub(monkeypatch):
    """liquid_mask asks app/data/store.py which coins are tradeable; stand in
    when that module isn't present in this checkout."""
    try:
        import app.data.store  # noqa: F401
        return
    except ImportError:
        import app.data
        stub = types.ModuleType("app.data.store")
        stub.is_tradeable_asset = lambda c: True
        monkeypatch.setitem(sys.modules, "app.data.store", stub)
        monkeypatch.setattr(app.data, "store", stub, raising=False)


def _panel(T=900, N=24, seed=0, low_vol_premium=0.0):
    """Coins with different volatilities; optionally low-vol coins earn more
    (a signal momentum doesn't capture)."""
    from app.engine.panel import Panel
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.02, 0.08, N)
    drift = low_vol_premium * (vols.mean() - vols) / vols.std()
    mkt = rng.normal(0.0005, 0.02, T)
    r = mkt[:, None] + drift[None, :] + rng.normal(0, 1, (T, N)) * vols[None, :]
    close = 100 * np.exp(np.cumsum(r, axis=0))
    names = ["BTC-USD", "ETH-USD"] + [f"C{i:02d}-USD" for i in range(N - 2)]
    days = np.arange(18000, 18000 + T)
    return Panel(days, names, close, np.full_like(close, 1e7) * (1 + np.arange(N)), "t",
                 close * 1.01, close * 0.99)


def test_features_are_causal(store_stub):
    from app.ml import dataset as D
    from app.engine import universe as U
    p = _panel()
    mask = U.liquid_mask(p, 20)
    X1, *_ = D.build(p, mask, h=7)
    p.close[600:] *= 1.5                          # rewrite the future
    X2, *_ = D.build(p, U.liquid_mask(p, 20), h=7)
    np.testing.assert_array_equal(np.nan_to_num(X1[:600], nan=-9), np.nan_to_num(X2[:600], nan=-9))


@pytest.mark.parametrize("model", ["ridge", "ensemble"])
def test_walk_forward_has_no_lookahead(store_stub, model):
    from app.ml import dataset as D, models as Mo
    from app.engine import universe as U
    h, refit, min_train = 7, 30, 200
    preds, k = [], None
    for scramble in (False, True):
        q = _panel(T=700)
        if scramble:
            rng = np.random.default_rng(9)
            q.close[k:] *= np.exp(rng.normal(0, 0.2, q.close[k:].shape))
        mask = U.liquid_mask(q, 20)
        X, y, w, _, _ = D.build(q, mask, h=h)
        if k is None:       # the day after a refit r (walk_forward's schedule): a model
            # refit at r that trained on any label ending after r sees the scramble
            first = int(np.flatnonzero((w > 0).any(axis=1))[0]) + min_train + h + 1
            k = first + 3 * refit + 1
        preds.append(Mo.walk_forward(X, y, w, model=model, h=h, refit_days=refit,
                                     min_train_days=min_train))
    a, b = preds
    assert np.isfinite(a[k - 1]).any()              # the refit at k - 1 is in the compared prefix
    np.testing.assert_array_equal(np.nan_to_num(a[:k], nan=-9), np.nan_to_num(b[:k], nan=-9))


def test_model_finds_signal_momentum_misses_and_eval_says_so(store_stub):
    from app.ml import dataset as D, models as Mo, evaluate as Ev
    from app.engine import universe as U, features as F
    p = _panel(T=1100, low_vol_premium=0.004)
    mask = U.liquid_mask(p, 24)
    X, y, w, _, _ = D.build(p, mask, h=7)
    pred = Mo.walk_forward(X, y, w, model="ridge", h=7, min_train_days=250)
    base = F.cross_rank(F.ret_n(p.close, 30))
    rep = Ev.ranking(pred, y, mask, base, h=7)
    assert rep["passes"] and rep["ic"] > rep["baseline_ic"]
    # with no planted signal the gate stays shut
    q = _panel(T=1100, seed=4)
    m2 = U.liquid_mask(q, 24)
    X2, y2, w2, _, _ = D.build(q, m2, h=7)
    rep2 = Ev.ranking(Mo.walk_forward(X2, y2, w2, model="ridge", h=7, min_train_days=250),
                      y2, m2, F.cross_rank(F.ret_n(q.close, 30)), h=7)
    assert not rep2["passes"]


def test_ml_candidate_weights_and_live_handoff(store_stub, tmp_path, monkeypatch):
    from app.ml import strategies as MLS
    from app.engine import challengers as C
    monkeypatch.setattr(MLS, "DIR", str(tmp_path))
    p = _panel(T=700)
    rank_cfg = {"universe_top": 20, "selection": "ml_rank", "sizing": "equal", "sma": 50,
                "top_k": 5, "model": "ridge", "horizon": 7}
    meta_cfg = {"assets": ["BTC-USD", "ETH-USD"], "selection": "trend_meta", "sma": 50,
                "model": "ridge", "horizon": 7, "train_top": 20}
    for cfg in (rank_cfg, meta_cfg):
        assert C.validate_config(cfg) is None
        W = C.weights(cfg, p)
        assert W.shape == (p.T, p.N) and (W >= 0).all() and (W.sum(axis=1) <= 1 + 1e-9).all()
    from app.engine import strategies as S
    uni = np.zeros((p.T, p.N), bool)
    uni[:, :2] = True
    base = S.trend_portfolio(p, "trend", sma=50, universe=uni)
    assert (C.weights(meta_cfg, p) <= base + 1e-12).all()       # meta only scales down
    W = C.weights(rank_cfg, p)
    MLS.save_latest("r", rank_cfg, p, W)
    day, w = MLS.latest_saved(rank_cfg, int(p.days[-1]))
    assert day == int(p.days[-1])
    assert w == {c: float(v) for c, v in zip(p.coins, W[-1]) if v > 0}
    assert MLS.latest_saved(rank_cfg, int(p.days[-1]) + 10) is None     # stale -> hold
    assert C.validate_config({"universe_top": 20, "selection": "ml_rank", "model": "x"})


def test_series_z_is_causal():
    from app.ml.dataset import series_z
    x = np.cumsum(np.random.default_rng(0).normal(size=500))
    a = series_z(x)
    x2 = x.copy()
    x2[300:] += 50
    b = series_z(x2)
    np.testing.assert_array_equal(np.nan_to_num(a[:300]), np.nan_to_num(b[:300]))


# ------------------------------------------------------------------ Polymarket calibration
def _ledger(n=400, seed=0):
    """Markets priced with a longshot bias: the true probability is lower
    than the price for longshots and higher for favourites. The bot just
    copies the market (no nudge)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(rng.uniform(0.05, 0.95))
        lp = math.log(p / (1 - p))
        true = 1 / (1 + math.exp(-1.6 * lp))
        y = int(rng.random() < true)
        rows.append({"condition_id": f"m{i}", "forecast_ts": 1e9 + i * 3600,
                     "resolved_ts": 1e9 + i * 3600 + 1800, "market_p0": p,
                     "predicted_p0": p, "votes": {}, "resolved_outcome0": y})
    return rows


def test_calibration_learns_a_market_bias_and_is_judged_out_of_sample():
    from app.markets.polymarket import calibration as Cal
    from app.markets.polymarket.skill_gate import evaluate
    rows = _ledger(n=1500)
    oos = Cal.walk_forward(rows, strategies=[])
    assert oos and all(r["condition_id"] for r in oos)
    rep = evaluate(oos, min_markets=100)
    assert rep["open"] and rep["mean_brier_diff"] < 0
    raw = evaluate(rows, min_markets=100)
    assert not raw["open"]                         # copying the market is no edge


def test_calibration_adjust_picks_side_from_calibrated_probability():
    from app.markets.polymarket import calibration as Cal

    class Model:
        def predict(self, X):
            return np.array([0.30])
    m = {"prices": [0.20, 0.80], "outcomes": ["Yes", "No"]}
    sig = {"outcome_index": 1, "outcome": "No", "price": 0.80, "fair": 0.85, "fair_p0": 0.15,
           "edge": 0.05, "confidence": 0.5, "leans": {}}
    out = Cal.adjust(m, sig, Model(), strategies=[])
    assert out["outcome_index"] == 0 and out["outcome"] == "Yes"
    assert abs(out["edge"] - 0.10) < 1e-9 and out["calibrated"]
