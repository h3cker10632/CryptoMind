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
    """Trend-following: EMA cross + MACD momentum, confirmed across timeframes.

    Multi-timeframe alignment (`mtf_align`, mean of 15m/1h/4h trend signs) both
    ADDS conviction when the higher timeframes agree with the 5m read and DAMPS
    the score when they conflict — a 5m EMA cross fighting the 1h/4h trend is
    exactly the kind of chop this filter is meant to avoid.
    """
    score = 0.0
    base = 0.45 if f["ema12"] > f["ema26"] else -0.45
    score += base
    score += _clip(f["macd_delta"] / (f["atr"] * 0.25 + 1e-9)) * 0.3
    score += _clip(f["mom_4h"] / 0.02) * 0.25
    align = f.get("mtf_align", 0.0)
    score += _clip(align) * 0.2                       # HTF confirmation vote
    if base * align < 0:                              # 5m fights the HTF trend
        score *= 0.6
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
    from ..learn.online_model import model, committee, build_x
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
    x = build_x(f, sent[0], _cur_market_sent(), d)
    # COMMITTEE + UNCERTAINTY: use the ensemble mean, and scale the vote down
    # when the members disagree or their quantile bands are wide. A confident,
    # agreed-upon signal keeps full weight; an uncertain one is damped toward 0.
    u = committee.predict_with_uncertainty(x)
    # stash the uncertainty for the sizer to read this cycle (see compute()).
    _ctx.ml_uncertainty = u
    return _clip(u["mean"] * 1.3) * trust * u["confidence"]


def strat_llm(f, sent, regime):
    """Optional LLM advisor vote — ONE opinion among many, weighted by the
    bandit like every other sleeve (NOT the driver). Returns the cached
    directional lean in [-1, 1], or 0.0 when the advisor is disabled /
    unconfigured / its opinion has expired — in which case the engine's
    active-strategy renorm simply excludes it. The (latency- and cost-heavy)
    model call happens out of band; this hot path only reads the cache."""
    from ..learn.llm_advisor import advisor
    return _clip(advisor.lean(_cur_product()))


def strat_model(f, sent, regime):
    """Optional ML-model advisor vote — a trained crypto_ml_lab model as ONE
    opinion among many, weighted by the bandit like every other sleeve (NOT the
    driver). Returns the cached directional lean in [-1, 1], or 0.0 when the
    advisor is disabled / no validated artifact is loaded / its prediction has
    expired — in which case the engine's active-strategy renorm excludes it. The
    (latency-heavy) inference happens out of band; this hot path only reads the
    cache."""
    from ..learn.model_advisor import advisor as model_advisor
    return _clip(model_advisor.lean(_cur_product()))


def strat_pattern(f, sent, regime):
    """Chart-pattern sleeve — reads the pre-computed pattern report on the
    features dict (market.features stamps feat["patterns"]) and votes its
    blended directional lean in [-1, 1]. Recognises market structure (HH/HL vs
    LH/LL), support/resistance position, reversal patterns (double top/bottom,
    head-&-shoulders), continuation patterns (triangles/wedges), candlesticks
    and RSI divergence. Weighted by the bandit like every other sleeve; returns
    0.0 when no pattern report is present (short history / backtest)."""
    rep = f.get("patterns") if isinstance(f, dict) else None
    if not rep:
        return 0.0
    return _clip(rep.get("lean", 0.0))


def _evolved_vote(g, f):
    """Score a single evolved genome's rule set on the current features."""
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


def strat_evolved(f, sent, regime):
    """GA-evolved champion rule set (promoted only after purged walk-forward
    validation). Now averages the CHAMPION PORTFOLIO (top-k promoted genomes)
    instead of betting on a single champion: a signal several independently
    validated genomes agree on is far less likely to be an overfit artefact,
    and disagreement naturally shrinks the vote toward zero. Inactive until
    evolution has promoted at least one genome."""
    from ..learn.evolution import evolution
    pop = evolution.portfolio_for(_cur_product())
    if not pop:
        return 0.0
    votes = [_evolved_vote(g, f) for g in pop]
    return _clip(sum(votes) / len(votes))


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
    "llm": strat_llm,
    "model": strat_model,
    "pattern": strat_pattern,
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
            _ctx.ml_uncertainty = None      # strat_ml sets this if it votes
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
            # DIRECTION LEARNER: nudge the composite toward whichever side has
            # actually paid in this regime (learned from realized net PnL), so a
            # regime where shorts keep losing stops producing marginal shorts.
            # A bias can flip only a marginal call; strong signal still wins.
            from ..learn.direction import direction_learner
            mtf_align = f.get("mtf_align", 0.0)
            adj_composite, dir_flipped = direction_learner.adjust(
                regime.get("label", "unknown"), composite, mtf_align)
            composite = adj_composite
            if confidence == 0.0 and composite != 0.0:
                # bias created a lean from a dead-flat composite; give it a small
                # floor confidence so it can be evaluated by the gate normally.
                confidence = min(1.0, abs(composite) * 0.5)
            direction = 1 if composite > 0 else -1
            # MODEL-UNCERTAINTY sizing hint (Phase 3): when the online committee
            # voted, expose its confidence so the risk manager can shrink the
            # position when the model is unsure (wide bands / members disagree)
            # and only press size when it's confident. Defaults to 1.0 (no
            # effect) whenever the ML head didn't participate.
            mlu = getattr(_ctx, "ml_uncertainty", None)
            ml_conf = float(mlu["confidence"]) if mlu else 1.0
            from .. import settings as app_settings
            from ..risk.stance import stance
            shorts_ok = app_settings.get("allow_shorts")
            dir_ok = direction > 0 or shorts_ok
            # MULTI-TIMEFRAME VETO: never open AGAINST a strongly-aligned higher-
            # timeframe trend (the "should have been a long" mistake). Vetoed
            # signals are made non-actionable rather than flipped.
            vetoed, veto_why = direction_learner.veto(direction, mtf_align)
            if vetoed:
                dir_ok = False
            gate = stance.current()["conf_gate"]     # stance-adjusted MIN_CONFIDENCE
            out[p] = Signal(
                product=p, direction=direction, confidence=round(confidence, 3),
                composite=round(composite, 3),
                ml_confidence=round(ml_conf, 3),
                edge_bps=round(composite * 25, 1),
                dir_flipped=dir_flipped, mtf_veto=veto_why,
                actionable=confidence >= gate and dir_ok,
                explorable=(tv("explore_min_confidence") <= confidence < gate
                            and dir_ok),
                # stop/target/sizing use the SWING ATR (higher timeframe) so a
                # trade can clear round-trip costs at its natural horizon — the
                # fix for "target can't beat fees" is the HORIZON, not fees.
                stop=round(f["price"] - 2 * f["atr_swing"], 6) if direction > 0
                     else round(f["price"] + 2 * f["atr_swing"], 6),
                price=f["price"], atr=f["atr_swing"], rsi=round(f["rsi"], 1),
                sentiment=round(sent[0], 3), regime=regime["label"],
                mtf_align=round(mtf_align, 4),
                ts=time.time(),
            )
        self.latest = out
        return out


engine = SignalEngine()
