"""The improvement pipeline start to finish on a synthetic market
(tools/run_pipeline.py --synthetic): every stage runs as its own process in a
throwaway workspace, a planted cross-coin signal is found by the ML lab,
queued as a candidate, scored by the research loop, and its saved weights
reach the live core when it is champion."""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HANDOFF = """
import sys
from app.engine import challengers as C
from app.strategies.core import CoreBook
from app import settings
name = sys.argv[1]
C._save(C.CHAMPION, {"name": name, "config": C.candidates()[name]})
real = settings.get
settings.get = lambda k: {"core_strategy": "champion", "core_allocation_pct": 50}.get(k, real(k))
core = CoreBook()
w = core._champion_weights()
assert core.enabled() and w and 0 < sum(w.values()) <= 1 + 1e-9, w
assert core.mode_used == ("champion", name), core.mode_used
print("ok", len(w))
"""


def _snapshot():
    """(path, mtime) of everything under this checkout's reports/ and data store."""
    out = set()
    for d in ("reports", os.path.join(".cache", "store")):
        for base, _, files in os.walk(os.path.join(ROOT, d)):
            for f in files:
                fp = os.path.join(base, f)
                out.add((fp, os.path.getmtime(fp)))
    return out


def test_pipeline_runs_start_to_finish_and_hands_a_model_to_the_core():
    before = _snapshot()
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "run_pipeline.py"), "--synthetic",
         "--coins", "24", "--start", "2018-01-01", "--ml-model", "ridge",
         "--plant-signal", "0.003", "--json"],
        capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    run = json.loads(p.stdout)
    ws = run["workspace"]
    try:
        assert os.path.basename(ws).startswith("cryptomind-synthetic-") and ws != ROOT
        stages = run["stages"]
        for name in ("synthetic_store", "screen", "ml", "research", "promotion"):
            assert stages[name]["exit"] == 0, (name, stages[name])
        assert stages["sync"]["skipped"]                       # synthetic: no network
        s = run["summary"]
        assert s["screen"]["screened"] > 10
        assert s["ml"]["rank_model"]["passes"]                 # the planted signal is found...
        queued = [q[0] for q in s["ml"]["queued_candidates"] if q[1]]
        assert queued and queued[0].startswith("ml_rank")
        cand = s["research"]["candidates"][queued[0]]          # ...and scored like any candidate
        assert cand["forward_test"] == "undecided" and cand["eligible"] is False
        assert s["research"]["champion"] == "btc_eth_trend"    # no forward evidence yet
        assert "exchange taker after-tax" in s["research"]["champion_costs_taxes"]
        assert s["promotion"]["verdict"]
        # hand-off: as champion, the queued model's saved weights drive the core
        chk = subprocess.run([sys.executable, "-c", HANDOFF, queued[0]], cwd=ws,
                             capture_output=True, text=True, timeout=300)
        assert chk.returncode == 0 and chk.stdout.startswith("ok"), chk.stdout + chk.stderr
        # this checkout's reports and data store were never touched
        assert _snapshot() == before
    finally:
        shutil.rmtree(ws, ignore_errors=True)
