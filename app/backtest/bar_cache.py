"""On-disk cache of the replay's per-bar signal data, so the daily replay and
the weekly learner ablation only compute bars they have not seen.

Per-bar data (signals, OHLC, trendiness, filter features, learner extras) does
not depend on the portfolio — `run_report` already exploits that within one
run. Across runs it stays valid as long as nothing it was computed from
changed. Each cached bar therefore carries a FINGERPRINT of exactly the candle
rows it read: for every coin with a bar at t, the trailing `hist_n` rows the
replay's market view hands the signal engine (`_HistMarket.set_time`). A
revised bar, a coin's history shifting (rows dropped at the front of a rolling
window) or a coin appearing invalidates exactly the bars whose window saw it.

Everything else that shapes the data is in the cache KEY: the source of the
modules that compute it, the current tunables, the offline signal settings,
the window lengths and whether learner extras are included. Any change there
starts a fresh cache file.

The file is a pickle the app writes and reads itself (like the daily lab's
rank model); it lives under .cache/ (gitignored).
"""
from __future__ import annotations
import hashlib
import json
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, ".cache", "replay_bars")

# modules whose code decides the per-bar data
_SOURCES = ("app/signals/engine.py", "app/signals/patterns.py", "app/data/features.py",
            "app/data/market.py", "app/backtest/replay.py", "app/learn/trade_filter.py",
            "app/learn/online_model.py", "app/config.py", "app/tunables.py")

# odd 64-bit multipliers, one per candle column, to hash a row into 64 bits
_K = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
               0xD6E8FEB86659FD93, 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53],
              dtype=np.uint64)


def code_version():
    h = hashlib.sha256()
    for rel in _SOURCES:
        try:
            with open(os.path.join(ROOT, rel), "rb") as f:
                h.update(rel.encode() + b"\0" + f.read())
        except OSError:
            h.update(rel.encode() + b"\0missing")
    return h.hexdigest()[:16]


def cache_key(offline, chop_n, hist_n, extras):
    from ..tunables import values
    blob = json.dumps({"code": code_version(), "tunables": values(),
                       "offline": {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in offline.items()},
                       "chop_n": chop_n, "hist_n": hist_n, "extras": bool(extras)},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _row_hashes(rows):
    """(n,) uint64 hash of each [ts, low, high, open, close, volume] row."""
    a = np.ascontiguousarray(np.asarray([r[:6] for r in rows], dtype=np.float64))
    u = a.view(np.uint64)
    with np.errstate(over="ignore"):
        h = (u * _K[: u.shape[1]]).sum(axis=1, dtype=np.uint64)
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return h


def fingerprints(C, pos_of, ts, hist_n):
    """{t: fingerprint} of the candle windows bar t reads, for every t in `ts`."""
    pref = {}
    for p, rows in C.items():
        if not rows:
            continue
        h = _row_hashes(rows)
        cs = np.zeros(len(h) + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            np.cumsum(h, dtype=np.uint64, out=cs[1:])
        pref[p] = (cs, rows)
    out = {}
    coins = sorted(pref)
    for t in ts:
        d = hashlib.blake2b(digest_size=16)
        for p in coins:
            i = pos_of[p].get(t)
            if i is None:
                continue
            cs, rows = pref[p]
            lo = max(0, i - hist_n + 1)
            with np.errstate(over="ignore"):
                s = int(cs[i + 1] - cs[lo])
            d.update(f"{p}|{i - lo + 1}|{int(rows[lo][0])}|{s};".encode())
        out[t] = d.hexdigest()
    return out


def _path(key):
    return os.path.join(DIR, f"{key}.pkl")


def load(key, fps):
    """Bars from the cache whose fingerprint still matches `fps` ({t: fp})."""
    try:
        with open(_path(key), "rb") as f:
            saved = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
        return {}
    return {t: bd for t, (fp, bd) in saved.items() if fps.get(t) == fp}


KEEP_FILES = 4      # the daily replay and the ablation each keep their own key


def save(key, bars, fps):
    """Persist every bar we have data for (atomic write). Only the
    `KEEP_FILES` most recently written configurations stay on disk."""
    os.makedirs(DIR, exist_ok=True)
    keep = {t: (fps[t], bd) for t, bd in bars.items() if t in fps}
    tmp = _path(key) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _path(key))
    files = sorted((os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(".pkl")),
                   key=os.path.getmtime, reverse=True)
    for f in files[KEEP_FILES:]:
        try:
            os.remove(f)
        except OSError:
            pass
