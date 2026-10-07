"""Key-free external series with downloadable HISTORY, so they can be tested
on the past before anything trades on them (app/data/series.py stores them
point-in-time; tools/series_sync.py keeps them current; tools/signal_screen.py
tests them).

Each fetcher returns [(ts, value, known_at)] — `ts` the start of the period
the value describes, `known_at` the earliest moment it could have been known.
Publication lags are set conservatively: a value is never treated as known
before its period has ended (plus the provider's usual delay).

Parsers follow each provider's documented response format; they are tested
against recorded response shapes in tests/test_series.py. Check a first live
run with `python tools/series_sync.py --dry-run`.
"""
from __future__ import annotations
import datetime as dt

DAY = 86400

CATALOG = {
    # Crypto Fear & Greed index (alternative.me), daily since 2018-02. The
    # value stamped at 00:00 UTC is published shortly after.
    "fear_greed": {"fetch": "fear_greed", "lag": 3600},
    # Deribit implied-volatility indices (DVOL), daily candles since 2021-03.
    # A daily close is known when the day ends.
    "dvol_btc": {"fetch": "deribit_dvol", "currency": "BTC", "lag": DAY},
    "dvol_eth": {"fetch": "deribit_dvol", "currency": "ETH", "lag": DAY},
    # Total USD stablecoin supply (DefiLlama), daily. Treated as known the day
    # after its stamp.
    "stablecoin_supply_usd": {"fetch": "defillama_stablecoins", "lag": DAY},
    # Coin Metrics community API (free tier), daily on-chain activity. Their
    # daily values settle with a delay; known two days after the day starts.
    "cm_btc_active_addresses": {"fetch": "coinmetrics", "asset": "btc",
                                "metric": "AdrActCnt", "lag": 2 * DAY},
    "cm_eth_active_addresses": {"fetch": "coinmetrics", "asset": "eth",
                                "metric": "AdrActCnt", "lag": 2 * DAY},
    "cm_btc_tx_count": {"fetch": "coinmetrics", "asset": "btc", "metric": "TxCnt",
                        "lag": 2 * DAY},
}


def fear_greed(client, lag=3600, **_):
    r = client.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"},
                   timeout=30)
    r.raise_for_status()
    out = []
    for d in r.json().get("data") or []:
        try:
            ts = int(d["timestamp"])
            out.append((ts, float(d["value"]), ts + lag))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


def parse_deribit_dvol(payload, lag=DAY):
    res = payload.get("result") or {}
    out = []
    for row in res.get("data") or []:
        try:
            ts = int(row[0]) // 1000
            out.append((ts, float(row[4]), ts + lag))       # [ts_ms, open, high, low, close]
        except (TypeError, ValueError, IndexError):
            continue
    return out, res.get("continuation")


def deribit_dvol(client, currency="BTC", start=None, end=None, lag=DAY, max_pages=20, **_):
    """Daily DVOL closes, paging backwards with Deribit's `continuation`."""
    end_ms = int((end or dt.datetime.now(dt.timezone.utc).timestamp()) * 1000)
    start_ms = int((start or 1_609_459_200) * 1000)            # 2021-01-01
    out = {}
    for _ in range(max_pages):
        r = client.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                       params={"currency": currency, "start_timestamp": start_ms,
                               "end_timestamp": end_ms, "resolution": "1D"}, timeout=30)
        r.raise_for_status()
        rows, cont = parse_deribit_dvol(r.json(), lag)
        for row in rows:
            out[row[0]] = row
        if not cont or int(cont) <= start_ms or int(cont) >= end_ms:
            break
        end_ms = int(cont)
    return sorted(out.values())


def parse_defillama_stablecoins(payload, lag=DAY):
    out = []
    for p in payload or []:
        try:
            ts = int(p["date"])
            tot = p.get("totalCirculatingUSD") or {}
            v = sum(float(x) for x in tot.values())
            if v > 0:
                out.append((ts, v, ts + lag))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


def defillama_stablecoins(client, lag=DAY, **_):
    r = client.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=60)
    r.raise_for_status()
    return parse_defillama_stablecoins(r.json(), lag)


def parse_coinmetrics(payload, metric, lag=2 * DAY):
    out = []
    for row in payload.get("data") or []:
        try:
            t = dt.datetime.strptime(row["time"][:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=dt.timezone.utc)
            ts = int(t.timestamp())
            out.append((ts, float(row[metric]), ts + lag))
        except (KeyError, TypeError, ValueError):
            continue
    return out, payload.get("next_page_url")


def coinmetrics(client, asset="btc", metric="AdrActCnt", start=None, lag=2 * DAY,
                max_pages=50, **_):
    start_s = dt.datetime.fromtimestamp(start or 1_420_070_400, dt.timezone.utc).strftime(
        "%Y-%m-%d")
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {"assets": asset, "metrics": metric, "frequency": "1d", "start_time": start_s,
              "page_size": 10000}
    out = []
    for _ in range(max_pages):
        r = client.get(url, params=params, timeout=60)
        r.raise_for_status()
        rows, nxt = parse_coinmetrics(r.json(), metric, lag)
        out += rows
        if not nxt:
            break
        url, params = nxt, None
    return sorted(set(out))


FETCHERS = {"fear_greed": fear_greed, "deribit_dvol": deribit_dvol,
            "defillama_stablecoins": defillama_stablecoins, "coinmetrics": coinmetrics}


def fetch(name, client, start=None):
    spec = dict(CATALOG[name])
    fn = FETCHERS[spec.pop("fetch")]
    return fn(client, start=start, **spec)
