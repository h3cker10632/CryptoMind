"""Central, deterministic seeding for the VALIDATION path.

One base seed (config.VALIDATION_SEED) fans out into stable per-context sub-seeds
so that independent stochastic steps — the GA search for a product, a bootstrap
CI, a DSR/PBO estimate — are each reproducible AND independent of one another.
Deriving sub-seeds from a hash of (base, context) means:

  * the same (base, context) always yields the same seed -> reproducible runs,
  * different contexts get uncorrelated streams -> the GA for BTC-USD doesn't
    share a stream with the ETH-USD run or with a bootstrap CI.

Set config.VALIDATION_SEED to a NEGATIVE number to opt out (each call then gets
fresh OS entropy, i.e. nondeterministic — useful when you WANT run-to-run
variation to gauge stability).
"""
from __future__ import annotations
import hashlib
import os
import random


def base_seed() -> int | None:
    """Current base validation seed, or None for nondeterministic mode."""
    from .. import config
    s = getattr(config, "VALIDATION_SEED", 1337)
    return None if s is None or s < 0 else int(s)


def derive(*context) -> int | None:
    """A stable 63-bit sub-seed from (base_seed, *context), or None in
    nondeterministic mode. `context` is any hashable-as-string label, e.g.
    derive("ga", "BTC-USD") or derive("bootstrap", "expectancy")."""
    b = base_seed()
    if b is None:
        return None
    key = "|".join([str(b), *(str(c) for c in context)]).encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def rng(*context) -> random.Random:
    """A random.Random seeded from derive(*context). In nondeterministic mode it
    is seeded from OS entropy (os.urandom) so runs vary."""
    seed = derive(*context)
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big")
    return random.Random(seed)
