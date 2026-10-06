"""Experiment registry: every backtest is logged — parameters, data version,
metrics and its daily return series — so that

  * any result can be traced to its exact data and settings, and
  * multiple-testing corrections (deflated Sharpe, PBO) count EVERY variant
    ever tried in a family, not just the ones that made it into a report.
"""
from __future__ import annotations
import hashlib
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.environ.get("CRYPTOMIND_EXPERIMENTS", os.path.join(ROOT, "reports", "experiments"))


def _id(family, name, params, data_version):
    blob = json.dumps([family, name, params, data_version], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def log(family, name, params, data_version, rets, metrics, start_day=None):
    """Record one backtest. Re-running the identical experiment overwrites its
    returns but is still ONE trial."""
    os.makedirs(DIR, exist_ok=True)
    eid = _id(family, name, params, data_version)
    np.save(os.path.join(DIR, f"{eid}.npy"), np.asarray(rets, dtype=float))
    entry = {"id": eid, "ts": time.time(), "family": family, "name": name,
             "params": params, "data_version": data_version, "start_day": start_day,
             "metrics": metrics}
    with open(os.path.join(DIR, "index.jsonl"), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return eid


def entries(family=None):
    try:
        with open(os.path.join(DIR, "index.jsonl")) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return []
    latest = {}
    for r in rows:
        if family is None or r["family"] == family:
            latest[r["id"]] = r
    return list(latest.values())


def n_trials(family):
    """Distinct (name, params) variants ever tried in `family` (any data)."""
    return len({json.dumps([e["name"], e["params"]], sort_keys=True, default=str)
                for e in entries(family)}) or 1


def trial_sharpe_std(family):
    """Dispersion of per-period Sharpe across the family's trials (for DSR)."""
    srs = []
    for e in entries(family):
        try:
            r = np.load(os.path.join(DIR, f"{e['id']}.npy"))
        except OSError:
            continue
        if len(r) > 1 and r.std() > 0:
            srs.append(r.mean() / r.std())
    return float(np.std(srs)) if len(srs) >= 2 else None


def returns(eid):
    return np.load(os.path.join(DIR, f"{eid}.npy"))
