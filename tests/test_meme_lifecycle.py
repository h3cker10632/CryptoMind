"""Meme hype-lifecycle classifier: phases + lean signs from price/volume/hype,
and the strat_meme sleeve stays inert for non-memes / when disabled."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.memes import Memes, memes


def test_markup_is_bullish():
    closes = [100 * (1.004 ** i) for i in range(30)]        # steady rise
    vols = [1000] * 24 + [3000, 3200, 3400, 3600, 3800, 4000]
    phase, lean, detail = Memes.classify(closes, vols, hype_vel=0.2)
    assert phase == "markup"
    assert lean > 0
    assert detail["vol_exp"] > 1.3


def test_blowoff_is_faded():
    # flat then vertical + volume climax = parabolic blow-off top -> fade
    closes = [100] * 18 + [100 * (1.03 ** i) for i in range(1, 13)]
    vols = [1000] * 24 + [5000, 6000, 7000, 8000, 9000, 10000]
    phase, lean, detail = Memes.classify(closes, vols, hype_vel=0.5)
    assert phase == "blowoff"
    assert lean < 0                       # do NOT chase the climax


def test_decay_is_bearish():
    closes = [130 - i for i in range(30)]
    vols = [3000] * 24 + [800, 700, 600, 500, 400, 300]
    phase, lean, _ = Memes.classify(closes, vols, hype_vel=-0.3)
    assert phase == "decay"
    assert lean <= 0


def test_distribution_rolls_over_near_highs():
    # steep rise decelerating to a crawl near the highs, on still-heavy volume:
    # recent momentum is small-positive but strongly DECELERATING (accel << 0).
    up = [100 + i * 4 for i in range(18)]                    # steep markup
    shallow = [168 + i * 0.3 for i in range(1, 13)]         # near-flat topping
    closes = up + shallow
    vols = [1000] * 24 + [2000, 2100, 2200, 2000, 1900, 1800]
    phase, lean, detail = Memes.classify(closes, vols, hype_vel=0.0)
    assert phase == "distribution"
    assert lean < 0
    assert detail["accel"] < 0


def test_dormant_when_flat_and_quiet():
    phase, lean, _ = Memes.classify([100] * 30, [1000] * 30, hype_vel=0.0)
    assert phase == "dormant"
    assert lean == 0.0


def test_short_history_neutral():
    phase, lean, _ = Memes.classify([100, 101, 102], [1000, 1000, 1000])
    assert phase == "dormant" and lean == 0.0


def test_hype_velocity_from_history():
    m = Memes()
    # two spaced samples: heat rose 10 -> 15 => +0.5 fractional
    m._record_hype("PEPE", 10.0, now=1000.0)
    m._record_hype("PEPE", 15.0, now=1100.0)
    assert abs(m._hype_velocity("PEPE") - 0.5) < 1e-6
    # a single sample => zero velocity
    m2 = Memes()
    m2._record_hype("WIF", 5.0, now=1000.0)
    assert m2._hype_velocity("WIF") == 0.0


def test_strat_meme_inert_for_non_meme_and_disabled():
    from app.signals.engine import strat_meme, STRATEGIES, _ctx
    assert "meme" in STRATEGIES
    # no product in context -> non-meme -> 0.0 (also safe when disabled)
    _ctx.product = "BTC-USD"
    assert strat_meme({}, 0.0, {}) == 0.0


def test_lean_returns_zero_without_market_history():
    class _M:
        candles = {}
    # PEPE is a seed meme; with no candles the lean must be a safe 0.0
    assert memes.lean("PEPE-USD", _M()) == 0.0
