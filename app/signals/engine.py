"""Signal Engine — ensemble of strategies producing directional signals with
confidence, edge estimate and invalidation levels. Strategy weights adapt via
the self-improvement loop."""
import time
import threading
from ..config import PRODUCTS
from ..tunables import tv
from .. import db

# per-thread context (was module-global mutable cells — not reentrant, so
# concurrent/parallel compute() calls used to cross-contaminate signals).
_ctx = threading.local()


def _cur_product():
    return getattr(_ctx, "product", "")


def _cur_market_sent():
    return getattr(_ctx, "market_sent", 0.0)


class Signal(dict):
    """direction: +1 long / -1 short, confidence in [0,1]. Shorts are
    actionable when allow_shorts is enabled (settings)."""


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


# ---------------- individual strategies ----------------

def strat_trend(f, sent, regime):
    """Trend-following: EMA cross + MACD momentum."""
    score = 0.0
    score += 0.45 if f["ema12"] > f["ema26"] else -0.45
    score += _clip(f["macd_delta"] / (f["atr"] * 0.25 + 1e-9)) * 0.3
    score += _clip(f["mom_4h"] / 0.02) * 0.25
    return score


def strat_meanrev(f, sent, regime):
    """Mean reversion on RSI extremes, damped in trending regimes."""
    if f["rsi"] < 30:
        score = (30 - f["rsi"]) / 30
    elif f["rsi"] > 70:
        score = -(f["rsi"] - 70) / 30
    else:
        score = 0.0
    if regime.get("trend") in ("bull", "bear"):
        score *= 0.5
    return _clip(score * 1.4)


def strat_breakout(f, sent, regime):
    """20-bar breakout with volume confirmation."""
    rng = f["hi20"] - f["lo20"] or 1e-9
    pos = (f["price"] - f["lo20"]) / rng
    score = 0.0
    if pos > 0.985 and f["vol_ratio"] > 1.2:
        score = 0.6 + min(0.4, (f["vol_ratio"] - 1.2) * 0.4)
    elif pos < 0.015 and f["vol_ratio"] > 1.2:
        score = -0.6 - min(0.4, (f["vol_ratio"] - 1.2) * 0.4)
    return _clip(score)


def strat_sentiment(f, sent, regime):
    """Narrative/sentiment driven, confirmed by short-term momentum."""
    s, n_docs = sent
    score = _clip(s * 1.6) * 0.7 + _clip(f["mom_1h"] / 0.01) * 0.3
    if n_docs == 0:
        score *= 0.5
    return _clip(score)


def strat_microstructure(f, sent, regime):
    """Order-book imbalance + spread quality."""
    score = _clip(f["imbalance"] * 2.2)
    if f["spread_bps"] > 8:      # illiquid — distrust
        score *= 0.4
    return score


def strat_derivatives(f, sent, regime):
    """Perp-market positioning: funding extremes (contrarian), OI-confirmed
    trends, long/short crowding, and taker aggression."""
    from ..data.derivatives import derivatives
    d = derivatives.features(_cur_product())
    if not d:
        return 0.0
    score = 0.0
    # contrarian on extreme funding: crowded longs paying big premium = risk
    score += -d["funding_norm"] * 0.35
    # OI expansion in direction of trend = confirmation
    trend_dir = 1.0 if f["ema12"] > f["ema26"] else -1.0
    score += trend_dir * max(0.0, d["oi_change_norm"]) * 0.30
    # crowding contrarian (everyone long → fade)
    score += -d["ls_crowding"] * 0.15
    # taker aggression momentum (buyers lifting offers)
    score += d["taker_aggression"] * 0.20
    return _clip(score)


def strat_ml(f, sent, regime):
    """Online neural network prediction (continual learning). Silent until
    the model has warmed up on enough labeled live samples."""
    from ..learn.online_model import model, build_x
    from ..data.derivatives import derivatives
    if model.n_updates < 40:
        return 0.0
    # Gate the vote on MEASURED directional accuracy BEFORE spending a forward
    # pass. A model at or below a coin flip (or one we haven't measured yet)
    # contributes NOTHING — no credit for being unmeasured, and no partial
    # credit below 50%. Trust ramps in only once it's genuinely better than
    # random, so a broken head (20.7% dir-acc) can no longer earn ensemble weight.
    st = model.stats()
    acc = st["directional_accuracy"]
    if acc is None or acc <= 0.50:
        return 0.0
    trust = _clip((acc - 0.50) / 0.10, 0.0, 1.0)   # 50%→0, 60%→full
    d = derivatives.features(_cur_product())
    pred = model.predict(build_x(f, sent[0], _cur_market_sent(), d))
    return _clip(pred * 1.3) * trust


def strat_evolved(f, sent, regime):
    """GA-evolved champion rule set (promoted only after out-of-sample
    validation). Inactive until evolution has produced a champion."""
    from ..learn.evolution import evolution
    g = evolution.champion_for(_cur_product())
    if not g:
        return 0.0
    score = 0.0
    ema_f = f["ema12"] if g["ema_fast"] <= 16 else f["ema26"]
    trend_ok = f["ema12"] > f["ema26"] if g["ema_fast"] < g["ema_slow"] else True
    if trend_ok:
        score += 0.35
    rng = f["hi20"] - f["lo20"] or 1e-9
    pos = (f["price"] - f["lo20"]) / rng
    if pos > 0.97:                                   # breakout leg
        score += 0.4
    if f["rsi"] < g["rsi_buy"]:                      # dip-buy leg
        score += 0.35
    if g["mom_w"] > 0.5 and f["mom_1h"] <= 0:
        score *= 0.4
    if not trend_ok:
        # downtrend: short-capable genomes vote short on breakdowns/overbought
        if g.get("short_w", 0) > 0.5:
            score = 0.0
            if pos < 0.03:                                   # 20-bar breakdown
                score -= 0.4
            if f["rsi"] > g.get("rsi_sell", 70):             # overbought fade
                score -= 0.35
            score -= 0.35 if f["ema12"] < f["ema26"] else 0  # trend confirm
            if g["mom_w"] > 0.5 and f["mom_1h"] >= 0:
                score *= 0.4
        else:
            score = -0.3 if f["rsi"] > 70 else 0.0
    return _clip(score)


# (per-thread context is set on `_ctx` in compute(); see top of file)

STRATEGIES = {
    "trend": strat_trend,
    "meanrev": strat_meanrev,
    "breakout": strat_breakout,
    "sentiment": strat_sentiment,
    "microstructure": strat_microstructure,
    "derivatives": strat_derivatives,
    "ml": strat_ml,
    "evolved": strat_evolved,
}


class SignalEngine:
    def __init__(self):
        self.weights = {k: 1.0 / len(STRATEGIES) for k in STRATEGIES}
        self.latest = {}          # product -> composite signal
        self.per_strategy = {}    # product -> {strategy: raw score}

    def set_weights(self, w):
        self.weights = w

    def compute(self, market, nlp, record=True):
        regime = market.regime()
        _ctx.market_sent = nlp.market_sentiment
        out = {}
        for p in PRODUCTS:
            f = market.features(p)
            if not f:
                continue
            sent = nlp.asset_score(p)
            _ctx.product = p
            raw = {}
            for name, fn in STRATEGIES.items():
                try:
                    s = _clip(fn(f, sent, regime))
                except Exception:
                    s = 0.0
                raw[name] = s
                if record and abs(s) > 0.3:
                    db.record_signal(name, p, 1 if s > 0 else -1, abs(s),
                                     regime.get("label"))
            self.per_strategy[p] = raw

            # --- active-strategy renormalization ---
            # Strategies with no opinion (|s| <= 0.05, e.g. ml warming up,
            # no breakout setup) must NOT dilute the composite. Weight-average
            # only over strategies that actually voted.
            active = {n: s for n, s in raw.items() if abs(s) > 0.05}
            if active:
                wsum = sum(self.weights[n] for n in active) or 1e-9
                composite = sum(self.weights[n] * s for n, s in active.items()) / wsum
                # agreement: fraction of active voters on the composite's side
                agree = sum(1 for s in active.values()
                            if s * composite > 0) / len(active)
                # breadth: more voters = more evidence (full trust at 3+)
                breadth = min(1.0, len(active) / 3)
                confidence = min(1.0, abs(composite) * (0.4 + 0.6 * agree) * (0.6 + 0.4 * breadth))
            else:
                composite, confidence = 0.0, 0.0
            direction = 1 if composite > 0 else -1
            from .. import settings as app_settings
            from ..risk.stance import stance
            shorts_ok = app_settings.get("allow_shorts")
            dir_ok = direction > 0 or shorts_ok
            gate = stance.current()["conf_gate"]     # stance-adjusted MIN_CONFIDENCE
            out[p] = Signal(
                product=p, direction=direction, confidence=round(confidence, 3),
                composite=round(composite, 3),
                edge_bps=round(composite * 25, 1),
                actionable=confidence >= gate and dir_ok,
                explorable=(tv("explore_min_confidence") <= confidence < gate
                            and dir_ok),
                stop=round(f["price"] - 2 * f["atr"], 6) if direction > 0
                     else round(f["price"] + 2 * f["atr"], 6),
                price=f["price"], atr=f["atr"], rsi=round(f["rsi"], 1),
                sentiment=round(sent[0], 3), regime=regime["label"],
                ts=time.time(),
            )
        self.latest = out
        return out


engine = SignalEngine()
