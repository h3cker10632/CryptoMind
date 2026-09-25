"""Chart-pattern detection engine — pure-Python, zero dependencies.

Reads recent OHLCV history and recognises classical chart structure and
patterns, then emits three things:

  * a directional LEAN in [-1, 1] consumed by the bandit's ``pattern`` sleeve
    (positive = bullish structure, negative = bearish);
  * a FIXED-ORDER numeric feature vector (``PATTERN_FEATURES``) fed to the
    online ML model so the net can learn from chart geometry too;
  * a human-readable list of DETECTED patterns — each ``{name, direction,
    strength, note}`` — surfaced to the dashboard / alerts so the operator can
    literally see which patterns are helping or hurting the book.

Detectors, all built from swing pivots + a light RSI so they cost almost
nothing and never look ahead (they only read closed bars):

  - MARKET STRUCTURE   higher-highs/higher-lows (uptrend) vs lower-highs/lower-lows
  - SUPPORT/RESISTANCE  where price sits between the nearest S/R levels
  - REVERSAL            double top / double bottom, head-&-shoulders / inverse
  - CONTINUATION        ascending / descending triangle, rising / falling wedge
  - CANDLESTICK         bullish/bearish engulfing, hammer, shooting star, doji
  - DIVERGENCE          price vs RSI (regular bullish / bearish)

Everything is best-effort: on too-little history it returns a neutral report,
and it NEVER raises (a bad bar must not take down the signal engine).
"""

# Fixed, order-stable feature names appended to the ML input vector. Changing
# this list changes N_IN — keep additions at the END so migration stays clean.
PATTERN_FEATURES = [
    "pat_structure",     # +1 uptrend structure ... -1 downtrend structure
    "pat_sr",            # +1 sitting on support ... -1 pinned under resistance
    "pat_reversal",      # signed reversal-pattern score (double top/bottom, H&S)
    "pat_continuation",  # signed continuation score (triangles / wedges)
    "pat_candle",        # signed candlestick score (engulfing / hammer / star)
    "pat_divergence",    # signed price-vs-RSI divergence score
]

_NEUTRAL = {
    "lean": 0.0,
    "features": {k: 0.0 for k in PATTERN_FEATURES},
    "detected": [],
    "structure": "unknown",
}

# blend weights for the overall lean (sum need not be 1; result is clamped)
_W = {
    "pat_structure": 0.30,
    "pat_sr": 0.15,
    "pat_reversal": 0.25,
    "pat_continuation": 0.15,
    "pat_candle": 0.10,
    "pat_divergence": 0.20,
}

MIN_BARS = 40          # need at least this much history to say anything
PIVOT_K = 3            # a swing pivot is an extreme over +/- K bars
LEVEL_TOL = 0.02       # two prices within 2% count as "the same level"


def _clip(x, lo=-1.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def _rsi(closes, period=14):
    """Wilder-style RSI series (same value list length as closes; leading
    entries are 50 until enough data)."""
    n = len(closes)
    out = [50.0] * n
    if n < period + 1:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        ch = closes[i] - closes[i - 1]
        gains += max(ch, 0.0)
        losses += max(-ch, 0.0)
    avg_g, avg_l = gains / period, losses / period
    for i in range(period, n):
        if i > period:
            ch = closes[i] - closes[i - 1]
            avg_g = (avg_g * (period - 1) + max(ch, 0.0)) / period
            avg_l = (avg_l * (period - 1) + max(-ch, 0.0)) / period
        rs = avg_g / avg_l if avg_l > 1e-12 else 999.0
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _pivots(highs, lows, k=PIVOT_K):
    """Return (pivot_highs, pivot_lows) as lists of (index, price). A pivot high
    at i is the strict-ish max of highs[i-k .. i+k]; pivot low mirrors it."""
    ph, pl = [], []
    n = len(highs)
    for i in range(k, n - k):
        seg_h = highs[i - k:i + k + 1]
        seg_l = lows[i - k:i + k + 1]
        if highs[i] >= max(seg_h):
            ph.append((i, highs[i]))
        if lows[i] <= min(seg_l):
            pl.append((i, lows[i]))
    return ph, pl


def _close(a, b, tol=LEVEL_TOL):
    """True when two prices are within `tol` fractional distance."""
    m = (abs(a) + abs(b)) / 2 or 1e-9
    return abs(a - b) / m <= tol


# --------------------------------------------------------------------------
# individual detectors — each returns (score in [-1,1], [detected dicts])
# --------------------------------------------------------------------------

def _detect_structure(ph, pl):
    """Market structure from the last two swing highs and lows."""
    if len(ph) < 2 or len(pl) < 2:
        return 0.0, []
    h1, h2 = ph[-2][1], ph[-1][1]      # older, newer high
    l1, l2 = pl[-2][1], pl[-1][1]
    hh, hl = h2 > h1, l2 > l1
    lh, ll = h2 < h1, l2 < l1
    if hh and hl:
        return 1.0, [{"name": "Uptrend structure (HH/HL)", "direction": "bullish",
                      "strength": 0.7, "note": "higher highs & higher lows — helps longs"}]
    if lh and ll:
        return -1.0, [{"name": "Downtrend structure (LH/LL)", "direction": "bearish",
                       "strength": 0.7, "note": "lower highs & lower lows — helps shorts / hurts longs"}]
    return 0.0, [{"name": "Range / no clear structure", "direction": "neutral",
                  "strength": 0.3, "note": "mixed swings — trend entries are lower quality"}]


def _detect_sr(price, ph, pl):
    """Where price sits between the nearest support and resistance."""
    res = [p for _, p in ph if p > price]
    sup = [p for _, p in pl if p < price]
    if not res or not sup:
        return 0.0, []
    resistance = min(res)
    support = max(sup)
    band = resistance - support
    if band <= 1e-9:
        return 0.0, []
    pos = (price - support) / band            # 0 at support, 1 at resistance
    score = _clip(1.0 - 2.0 * pos)            # +1 at support, -1 at resistance
    detected = []
    if pos <= 0.15:
        detected.append({"name": "Testing support", "direction": "bullish",
                         "strength": 0.5, "note": "near support — bounce zone, hurts fresh shorts"})
    elif pos >= 0.85:
        detected.append({"name": "Testing resistance", "direction": "bearish",
                         "strength": 0.5, "note": "near resistance — rejection zone, hurts fresh longs"})
    return score, detected


def _detect_double(ph, pl, price):
    """Double top (bearish) / double bottom (bullish)."""
    out = []
    score = 0.0
    if len(ph) >= 2:
        (i1, p1), (i2, p2) = ph[-2], ph[-1]
        if _close(p1, p2) and i2 - i1 >= PIVOT_K and price < min(p1, p2):
            score -= 0.9
            out.append({"name": "Double top", "direction": "bearish", "strength": 0.8,
                        "note": "twin peaks rejected — classic top, warns against longs"})
    if len(pl) >= 2:
        (i1, p1), (i2, p2) = pl[-2], pl[-1]
        if _close(p1, p2) and i2 - i1 >= PIVOT_K and price > max(p1, p2):
            score += 0.9
            out.append({"name": "Double bottom", "direction": "bullish", "strength": 0.8,
                        "note": "twin troughs held — classic bottom, supports longs"})
    return _clip(score), out


def _detect_hns(ph, pl, price):
    """Head-and-shoulders (bearish) and inverse H&S (bullish)."""
    out = []
    score = 0.0
    if len(ph) >= 3:
        (_, l), (_, h), (_, r) = ph[-3], ph[-2], ph[-1]
        if h > l and h > r and _close(l, r, tol=LEVEL_TOL * 1.5):
            score -= 0.85
            out.append({"name": "Head & shoulders", "direction": "bearish", "strength": 0.75,
                        "note": "head above even shoulders — distribution top, warns against longs"})
    if len(pl) >= 3:
        (_, l), (_, h), (_, r) = pl[-3], pl[-2], pl[-1]
        if h < l and h < r and _close(l, r, tol=LEVEL_TOL * 1.5):
            score += 0.85
            out.append({"name": "Inverse head & shoulders", "direction": "bullish", "strength": 0.75,
                        "note": "trough below even shoulders — accumulation bottom, supports longs"})
    return _clip(score), out


def _slope(pivots):
    """Sign of the linear slope through a list of (idx, price) pivots."""
    if len(pivots) < 2:
        return 0.0
    xs = [i for i, _ in pivots]
    ys = [p for _, p in pivots]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    return num / den


def _detect_triangle_wedge(ph, pl):
    """Ascending/descending triangles and rising/falling wedges from the slope
    of the recent resistance (pivot-high) and support (pivot-low) lines."""
    hh = ph[-3:]
    ll = pl[-3:]
    if len(hh) < 2 or len(ll) < 2:
        return 0.0, []
    top = _slope(hh)
    bot = _slope(ll)
    # normalise slopes by average price to a per-bar fraction
    avg = (sum(p for _, p in hh) + sum(p for _, p in ll)) / (len(hh) + len(ll)) or 1e-9
    top_f, bot_f = top / avg, bot / avg
    flat = 3e-4                       # |slope/price/bar| below this is "flat"
    out, score = [], 0.0
    if abs(top_f) < flat and bot_f > flat:
        score += 0.6
        out.append({"name": "Ascending triangle", "direction": "bullish", "strength": 0.6,
                    "note": "flat resistance, rising support — bullish continuation"})
    elif abs(bot_f) < flat and top_f < -flat:
        score -= 0.6
        out.append({"name": "Descending triangle", "direction": "bearish", "strength": 0.6,
                    "note": "flat support, falling resistance — bearish continuation"})
    elif top_f > flat and bot_f > flat and bot_f > top_f:
        score -= 0.5
        out.append({"name": "Rising wedge", "direction": "bearish", "strength": 0.5,
                    "note": "converging up — bearish, momentum fading"})
    elif top_f < -flat and bot_f < -flat and bot_f > top_f:
        score += 0.5
        out.append({"name": "Falling wedge", "direction": "bullish", "strength": 0.5,
                    "note": "converging down — bullish, selling exhausting"})
    return _clip(score), out


def _body(o, c):
    return abs(c - o)


def _detect_candles(opens, highs, lows, closes, structure):
    """Last-bar candlestick reads, contextualised by structure."""
    if len(closes) < 2:
        return 0.0, []
    o, c = opens[-1], closes[-1]
    po, pc = opens[-2], closes[-2]
    h, l = highs[-1], lows[-1]
    rng = (h - l) or 1e-9
    body = _body(o, c)
    upper = h - max(o, c)
    lower = min(o, c) - l
    out, score = [], 0.0
    # engulfing
    if pc < po and c > o and c >= po and o <= pc:
        score += 0.7
        out.append({"name": "Bullish engulfing", "direction": "bullish", "strength": 0.6,
                    "note": "up-bar swallows prior down-bar — supports longs"})
    elif pc > po and c < o and c <= po and o >= pc:
        score -= 0.7
        out.append({"name": "Bearish engulfing", "direction": "bearish", "strength": 0.6,
                    "note": "down-bar swallows prior up-bar — warns against longs"})
    # hammer (bullish, in a downtrend)
    if body < 0.35 * rng and lower > 2 * body and upper < body and structure <= 0:
        score += 0.5
        out.append({"name": "Hammer", "direction": "bullish", "strength": 0.5,
                    "note": "long lower wick after weakness — reversal up"})
    # shooting star (bearish, in an uptrend)
    if body < 0.35 * rng and upper > 2 * body and lower < body and structure >= 0:
        score -= 0.5
        out.append({"name": "Shooting star", "direction": "bearish", "strength": 0.5,
                    "note": "long upper wick after strength — reversal down"})
    # doji — indecision (no directional score, but worth surfacing)
    if body < 0.1 * rng:
        out.append({"name": "Doji", "direction": "neutral", "strength": 0.3,
                    "note": "indecision — trend continuation less reliable"})
    return _clip(score), out


def _detect_divergence(closes, rsi, pl, ph):
    """Regular price-vs-RSI divergence at the last two swing pivots."""
    out, score = [], 0.0
    if len(pl) >= 2:
        (i1, p1), (i2, p2) = pl[-2], pl[-1]
        if p2 < p1 and rsi[i2] > rsi[i1] + 2:       # lower low, higher RSI
            score += 0.7
            out.append({"name": "Bullish RSI divergence", "direction": "bullish", "strength": 0.6,
                        "note": "price lower low but RSI higher — selling losing steam"})
    if len(ph) >= 2:
        (i1, p1), (i2, p2) = ph[-2], ph[-1]
        if p2 > p1 and rsi[i2] < rsi[i1] - 2:       # higher high, lower RSI
            score -= 0.7
            out.append({"name": "Bearish RSI divergence", "direction": "bearish", "strength": 0.6,
                        "note": "price higher high but RSI lower — buying losing steam"})
    return _clip(score), out


def analyze(highs, lows, closes, vols=None, opens=None):
    """Run every detector over recent OHLC history and return a report dict:

        {"lean": float in [-1,1],
         "features": {name: score, ...} for PATTERN_FEATURES,
         "detected": [{name, direction, strength, note}, ...],
         "structure": "uptrend"|"downtrend"|"range"|"unknown"}

    Never raises; returns a neutral report on bad / short input.
    """
    try:
        n = len(closes)
        if n < MIN_BARS or len(highs) != n or len(lows) != n:
            return dict(_NEUTRAL, features=dict(_NEUTRAL["features"]))
        opens = opens if (opens and len(opens) == n) else \
            [closes[i - 1] if i > 0 else closes[0] for i in range(n)]
        price = closes[-1]
        ph, pl = _pivots(highs, lows)
        rsi = _rsi(closes)

        s_struct, d_struct = _detect_structure(ph, pl)
        s_sr, d_sr = _detect_sr(price, ph, pl)
        s_dbl, d_dbl = _detect_double(ph, pl, price)
        s_hns, d_hns = _detect_hns(ph, pl, price)
        s_rev = _clip(s_dbl + s_hns)
        s_cont, d_cont = _detect_triangle_wedge(ph, pl)
        s_candle, d_candle = _detect_candles(opens, highs, lows, closes, s_struct)
        s_div, d_div = _detect_divergence(closes, rsi, pl, ph)

        feats = {
            "pat_structure": s_struct,
            "pat_sr": s_sr,
            "pat_reversal": s_rev,
            "pat_continuation": s_cont,
            "pat_candle": s_candle,
            "pat_divergence": s_div,
        }
        lean = _clip(sum(_W[k] * v for k, v in feats.items()))
        structure = ("uptrend" if s_struct > 0.5 else
                     "downtrend" if s_struct < -0.5 else "range")
        detected = (d_struct + d_sr + d_dbl + d_hns + d_cont + d_candle + d_div)
        # strongest / most decision-relevant first
        detected.sort(key=lambda d: -d.get("strength", 0.0))
        return {"lean": lean, "features": feats,
                "detected": detected, "structure": structure}
    except Exception:
        return dict(_NEUTRAL, features=dict(_NEUTRAL["features"]))


def feature_vector(report):
    """Extract the fixed-order pattern feature list from a report (or a neutral
    zero-vector), for appending to the ML input vector."""
    f = (report or {}).get("features", {}) if isinstance(report, dict) else {}
    return [float(f.get(k, 0.0)) for k in PATTERN_FEATURES]
