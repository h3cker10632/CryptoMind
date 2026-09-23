"""Training-grade ML dataset emitter (app/ml_export.py + /api/ml_dataset)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db, ml_export


def _init():
    db.init()


def _seed_scored_signal(ts, product="BTC-USD", strategy="trend", direction=1,
                        conf=0.7, fwd=0.012, regime="trend-up"):
    with db._lock, db._conn() as c:
        c.execute("INSERT INTO signal_scores(ts,strategy,product,direction,"
                  "confidence,fwd_return,scored,regime) VALUES(?,?,?,?,?,?,1,?)",
                  (ts, strategy, product, direction, conf, fwd, regime))
        c.commit()


def _seed_unscored_signal(ts, product="BTC-USD"):
    with db._lock, db._conn() as c:
        c.execute("INSERT INTO signal_scores(ts,strategy,product,direction,"
                  "confidence,fwd_return,scored,regime) VALUES(?,?,?,?,?,NULL,0,?)",
                  (ts, "meanrev", product, -1, 0.4, "trend-up"))
        c.commit()


def _seed_decision(ts, product="BTC-USD"):
    with db._lock, db._conn() as c:
        c.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (ts, 1, product, 1, 0.55, 0.7, 0.6, "enter", "composite>gate",
                   1000, 800, "trend-up", '{"trend":0.5}'))
        c.commit()


def _seed_candles():
    from app.data.market import market
    now = time.time()
    market.candles["BTC-USD"] = [
        [now - 300 * i, 60000 - i, 60100 - i, 60000 - i, 60050 - i, 12.5]
        for i in range(4)][::-1]
    market.books["BTC-USD"] = {"bid_depth": 5e5, "ask_depth": 4.8e5,
                               "imbalance": 0.02, "spread_bps": 1.5}


def test_dataset_has_schema_and_sections():
    _init()
    d = ml_export.build_ml_dataset()
    assert d["schema"] == "cryptomind.ml_dataset.v1"
    assert "events" in d and "labels" in d and "meta" in d
    assert "label_horizon_sec" in d["meta"]


def test_only_labeled_signals_emitted():
    _init()
    now = time.time()
    _seed_scored_signal(now - 4000)
    _seed_unscored_signal(now - 3000)   # must be excluded
    d = ml_export.build_ml_dataset()
    labels = d["labels"]
    assert len(labels) >= 1
    # every emitted row carries a realized label
    assert all(r["y_fwd_return"] is not None for r in labels)
    # the unscored meanrev row is not present
    assert all(r["strategy"] != "meanrev" for r in labels)


def test_label_direction_correctness():
    _init()
    now = time.time()
    _seed_scored_signal(now - 5000, direction=1, fwd=0.02)    # long, up -> correct
    _seed_scored_signal(now - 4000, direction=1, fwd=-0.02,
                        strategy="breakout")                   # long, down -> wrong
    d = ml_export.build_ml_dataset()
    by_strat = {r["strategy"]: r for r in d["labels"]}
    assert by_strat["trend"]["y_direction_correct"] == 1
    assert by_strat["breakout"]["y_direction_correct"] == 0


def test_decision_context_joined_nearest_before():
    _init()
    now = time.time()
    sig = now - 4000
    _seed_decision(sig - 5)          # 5s before the signal
    _seed_scored_signal(sig)
    d = ml_export.build_ml_dataset()
    row = d["labels"][-1]
    assert "context" in row
    assert row["context"]["action"] == "enter"
    assert row["context"]["ctx_lag_sec"] == 5.0
    assert row["context"]["votes"] == {"trend": 0.5}


def test_events_satisfy_lab_contract_and_no_lookahead():
    _init()
    _seed_candles()
    d = ml_export.build_ml_dataset()
    evs = d["events"]
    assert len(evs) >= 3
    REQUIRED = {"ts", "symbol", "price"}
    assert all(REQUIRED <= set(e) for e in evs)
    assert all(e["ts"] and e["symbol"] and e["price"] for e in evs)
    # sorted by (symbol, ts)
    keys = [(e["symbol"], e["ts"]) for e in evs]
    assert keys == sorted(keys)
    # no forward-looking columns leak into events
    assert not any(k.startswith("y_") or "fwd" in k for e in evs for k in e)


def test_book_and_sentiment_only_on_last_bar():
    _init()
    _seed_candles()
    d = ml_export.build_ml_dataset()
    btc = [e for e in d["events"] if e["symbol"] == "BTC-USD"]
    # older bars must not back-fill a live snapshot value
    assert all(e["liquidity"] is None for e in btc[:-1])
    assert btc[-1]["liquidity"] is not None
    assert btc[-1]["spread_bps"] == 1.5


def test_json_serialisable_and_counts():
    _init()
    now = time.time()
    _seed_scored_signal(now - 4000)
    _seed_candles()
    d = ml_export.build_ml_dataset()
    s = json.dumps(d, default=str)
    assert len(s) > 100
    assert d["meta"]["n_labeled_signals"] == len(d["labels"])
    assert d["meta"]["n_events"] == len(d["events"])


def test_cli_split_writes_two_files(tmp_path):
    _init()
    rc = ml_export.main([str(tmp_path), "--split"])
    assert rc == 0
    ev = tmp_path / "events.json"
    lb = tmp_path / "labels.json"
    assert ev.exists() and lb.exists()
    # events.json is directly consumable by crypto_ml io.load_json
    assert "events" in json.loads(ev.read_text())
    assert json.loads(lb.read_text())["schema"] == "cryptomind.ml_dataset.v1"


def test_cli_single_file(tmp_path):
    _init()
    out = tmp_path / "ds.json"
    rc = ml_export.main([str(out), "--compact"])
    assert rc == 0
    txt = out.read_text()
    assert "\n" not in txt.strip()
    assert json.loads(txt)["schema"] == "cryptomind.ml_dataset.v1"


def test_labeled_signals_db_helper_excludes_unscored():
    _init()
    now = time.time()
    _seed_scored_signal(now - 100)
    _seed_unscored_signal(now - 50)
    rows = db.labeled_signals()
    assert all(r["fwd_return"] is not None for r in rows)
    # ascending chronological order
    ts = [r["ts"] for r in rows]
    assert ts == sorted(ts)
