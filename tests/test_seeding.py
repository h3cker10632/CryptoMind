"""Central validation seeding: reproducible + independent per context."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backtest import seeding
from app import config


def test_derive_is_deterministic():
    a = seeding.derive("ga", "BTC-USD")
    b = seeding.derive("ga", "BTC-USD")
    assert a == b and a is not None


def test_derive_differs_by_context():
    assert seeding.derive("ga", "BTC-USD") != seeding.derive("ga", "ETH-USD")
    assert seeding.derive("ga", "BTC-USD") != seeding.derive("bootstrap", "BTC-USD")


def test_derive_changes_with_base_seed(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_SEED", 1337)
    s1 = seeding.derive("x")
    monkeypatch.setattr(config, "VALIDATION_SEED", 9999)
    s2 = seeding.derive("x")
    assert s1 != s2


def test_negative_seed_is_nondeterministic(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_SEED", -1)
    assert seeding.derive("x") is None
    assert seeding.base_seed() is None
    # rng still returns a usable Random even in nondeterministic mode
    r = seeding.rng("x")
    assert 0.0 <= r.random() <= 1.0


def test_rng_reproducible_in_deterministic_mode(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_SEED", 1337)
    seq1 = [seeding.rng("ga", "BTC-USD").random() for _ in range(1)]
    seq2 = [seeding.rng("ga", "BTC-USD").random() for _ in range(1)]
    assert seq1 == seq2


def test_two_negative_rngs_differ(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_SEED", -1)
    a = [seeding.rng("x").random() for _ in range(5)]
    b = [seeding.rng("x").random() for _ in range(5)]
    assert a != b            # OS entropy -> different streams


def test_ga_run_is_reproducible(monkeypatch):
    """Two GA runs on identical candles with the same base seed produce the same
    champion report (the whole point: an auditable promotion decision)."""
    monkeypatch.setattr(config, "VALIDATION_SEED", 4242)
    import random
    from app.learn.evolution import Evolution

    rng = random.Random(0)
    px = 100.0
    candles = []
    t = 0
    for _ in range(700):
        t += 3600
        op = px
        px *= (1 + rng.gauss(0.0004, 0.012))
        cl = px
        hi = max(op, cl) * (1 + abs(rng.gauss(0, 0.003)))
        lo = min(op, cl) * (1 - abs(rng.gauss(0, 0.003)))
        candles.append([t, lo, hi, op, cl, rng.uniform(100, 200)])

    e1 = Evolution(pop_size=8, generations=3)
    r1 = e1.evolve([c[:] for c in candles], product="BTC-USD")
    e2 = Evolution(pop_size=8, generations=3)
    r2 = e2.evolve([c[:] for c in candles], product="BTC-USD")
    # the full per-generation search trajectory must match bit-for-bit — a far
    # stronger check than the (possibly None) promoted genome on random data.
    assert e1.history == e2.history
    assert len(e1.history) == 3
    assert r1.get("genome") == r2.get("genome")
    assert r1.get("promoted") == r2.get("promoted")


def test_ga_differs_across_products(monkeypatch):
    """Independent streams: same seed, different product -> different search."""
    monkeypatch.setattr(config, "VALIDATION_SEED", 4242)
    import random
    from app.learn.evolution import Evolution

    rng = random.Random(0)
    px = 100.0
    candles = []
    t = 0
    for _ in range(700):
        t += 3600
        op = px
        px *= (1 + rng.gauss(0.0004, 0.012))
        cl = px
        hi = max(op, cl) * (1 + abs(rng.gauss(0, 0.003)))
        lo = min(op, cl) * (1 - abs(rng.gauss(0, 0.003)))
        candles.append([t, lo, hi, op, cl, rng.uniform(100, 200)])

    a = Evolution(pop_size=8, generations=3)
    a.evolve([c[:] for c in candles], product="BTC-USD")
    b = Evolution(pop_size=8, generations=3)
    b.evolve([c[:] for c in candles], product="ETH-USD")
    assert a.history != b.history        # uncorrelated per-product streams
