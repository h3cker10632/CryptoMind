"""Validate the Invo-signal harness discriminates real edge from noise.

If these pass, the measurement machinery is trustworthy on real collected data:
it flags a KNOWN embedded relationship as significant, and correctly finds NO
edge when the signal is pure noise. (Synthetic data here is a correctness
fixture only — never real Invo data, never fed to the live model.)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.invo_signal.make_synthetic import generate
from tools.invo_signal import run_study, study


def _report(mode, rho=0.35, n=320, seed=1, tmp_path="/tmp/_invo_t"):
    import json
    snaps, prices = generate(mode, n_snaps=n, rho=rho, seed=seed)
    sp, pp = f"{tmp_path}_s.json", f"{tmp_path}_p.json"
    json.dump(snaps, open(sp, "w")); json.dump(prices, open(pp, "w"))
    return run_study.run(sp, pp, horizon_hours=1.0)


def test_recovers_known_edge():
    rep = _report("edge", rho=0.4)
    p = rep["pooled"]
    assert p["n"] >= 100
    assert p["ic"] > 0.05, f"should detect positive IC, got {p['ic']}"
    assert p["p_value"] < 0.05, f"edge should be significant, got p={p['p_value']}"


def test_rejects_pure_noise():
    rep = _report("null")
    p = rep["pooled"]
    assert p["n"] >= 100
    # no real relationship => IC small and NOT significant
    assert p["p_value"] > 0.05, f"noise flagged as signal! p={p['p_value']}"


def test_poller_append_and_schema_roundtrip(tmp_path):
    """Poller plumbing works offline: atomic append + output parses back through
    the collector schema. No network involved."""
    from tools.invo_signal.poller import InvoPoller, PollerConfig, default_mapper
    from tools.invo_signal.collector import load_snapshots
    import pytest

    out = str(tmp_path / "snaps.json")
    cfg = PollerConfig(base_url="https://example.invalid", token="x",
                       leaderboard_path="/lb", out_path=out)
    p = InvoPoller(cfg)
    # atomic append of two schema-valid snapshots
    for k in range(2):
        n = p.append_snapshot({"ts": 1_700_000_000.0 + k, "traders": [
            {"id": "t1", "rank": 1, "score": 0.7, "positions": [
                {"asset": "BTC", "direction": 1, "notional": 1000.0, "leverage": 3}]}]})
    assert n == 2
    snaps = load_snapshots(out)          # parses back cleanly
    assert len(snaps) == 2
    assert snaps[0].traders[0].positions[0].asset == "BTC"

    # default mapper must refuse to guess the response shape
    with pytest.raises(NotImplementedError):
        default_mapper({"data": []}, None, 25)


def test_stats_primitives():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500); y = 0.6 * x + rng.normal(scale=0.5, size=500)
    assert study.spearman(x, y) > 0.4
    assert study.pearson(x, y) > 0.4
    assert study.mutual_info(x, y) > 0.0
    # partial IC of x vs y controlling for y itself -> ~0
    assert abs(study.partial_ic(x, y, y.reshape(-1, 1))) < 0.2


def test_verdict_pinpoints_zero_positions(tmp_path):
    """A leaderboard with traders but no positions must produce a SPECIFIC
    verdict naming the cause, not an opaque 'NO DATA'."""
    import json
    from tools.invo_signal import run_study
    snaps = [{"ts": 1789435352.0 + i * 60,
              "traders": [{"id": "t1", "rank": 1, "score": 0.5, "positions": []}]}
             for i in range(174)]
    prices = {"BTC": [[1789435352.0 + j * 60, 100 + j * 0.1] for j in range(400)]}
    sp, pp = str(tmp_path / "s.json"), str(tmp_path / "p.json")
    json.dump(snaps, open(sp, "w")); json.dump(prices, open(pp, "w"))
    rep = run_study.run(sp, pp, horizon_hours=1.0)
    assert rep["diagnostics"]["positions_total"] == 0
    assert rep["diagnostics"]["aligned_pairs"] == 0
    v = run_study._verdict(rep["pooled"], rep["diagnostics"])
    assert "ZERO positions" in v


def test_verdict_pinpoints_asset_mismatch(tmp_path):
    import json
    from tools.invo_signal import run_study
    snaps = [{"ts": 1789435352.0 + i * 60,
              "traders": [{"id": "t1", "rank": 1, "score": 0.8,
                           "positions": [{"asset": "ETH", "direction": 1,
                                          "notional": 1000.0}]}]}
             for i in range(50)]
    prices = {"BTC": [[1789435352.0 + j * 60, 100 + j * 0.1] for j in range(400)]}
    sp, pp = str(tmp_path / "s.json"), str(tmp_path / "p.json")
    json.dump(snaps, open(sp, "w")); json.dump(prices, open(pp, "w"))
    rep = run_study.run(sp, pp, horizon_hours=1.0)
    v = run_study._verdict(rep["pooled"], rep["diagnostics"])
    assert "don't match" in v and "ETH" in v
