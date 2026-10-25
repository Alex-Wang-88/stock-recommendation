"""Correctness checks for real-series alignment, missing data, and derived summaries."""

from datetime import UTC, datetime, timedelta

import pytest
import requests

from recommend.complexity import (
    FACTORS,
    SECTORS,
    basket_history,
    build_snapshot,
    clean_rows,
    correlation,
    download_source,
    lagged_changes,
    parse_tencent,
    parse_treasury,
    parse_yahoo,
    percentile,
    previous_complete_day,
    seed_cache,
    source_summary,
    window_network,
)


def sample_source(id="test", kind="index", rows=None):
    return {
        "id": id,
        "label": id,
        "kind": kind,
        "symbol": id,
        "status": "ok",
        "observations": rows or [],
        "fetched_at": "2026-10-19T08:00:00+00:00",
    }


def rows(count=150, offset=0):
    import math

    start = datetime(2026, 1, 1)
    return [
        {
            "date": (start + timedelta(days=i)).date().isoformat(),
            "close": 100 + 0.1 * i + math.sin(i * 0.37 + offset) * 3,
            "amount": 1000.0,
        }
        for i in range(count)
    ]


def test_parser_is_strict_about_prices_dates_and_duplicates():
    assert clean_rows(
        [
            {"date": "2026-01-01", "close": 1},
            {"date": "2026-01-01", "close": 2},
            {"date": "2026-01-02", "close": float("nan")},
            {"date": "2026-01-04", "close": 3},
            {"date": "wrong", "close": 3},
        ],
        "2026-01-03",
    ) == [{"date": "2026-01-01", "close": 2.0, "amount": None}]


def test_tencent_units_and_no_adjustment_fallback():
    raw = {
        "data": {"sh512480": {"day": [["2026-01-01", "1", "2", "3", "1", "10", {}, "1", "12.5"]]}}
    }
    result = parse_tencent(raw, "sh512480", "2026-01-02")
    assert result[0]["close"] == 2
    assert result[0]["amount"] == 125000
    with pytest.raises(ValueError):
        parse_tencent({"data": {"sh512480": {"qfqday": []}}}, "sh512480", "2026-01-02")


def test_lag_alignment_excludes_same_day_and_does_not_repeat_holiday_return():
    source = sample_source(
        rows=[
            {"date": "2026-01-01", "close": 100},
            {"date": "2026-01-02", "close": 110},
            {"date": "2026-01-05", "close": 220},
        ]
    )
    actual = lagged_changes(
        source, ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]
    )
    assert actual["2026-01-03"] == pytest.approx(0.1)
    assert actual["2026-01-04"] == 0
    assert (
        actual["2026-01-05"] == 0
    )  # That day's close cannot enter that day's A-share observation.
    assert actual["2026-01-06"] == 1
    assert lagged_changes(source, ["2026-02-01", "2026-02-02"]) == {}


def test_yield_change_is_bp_and_zero_variance_is_missing_not_zero_correlation():
    src = sample_source(
        kind="yield",
        rows=[{"date": "2026-01-01", "close": 4}, {"date": "2026-01-02", "close": 4.02}],
    )
    assert lagged_changes(src, ["2026-01-02", "2026-01-03"])["2026-01-03"] == pytest.approx(2)
    assert correlation([(1.0, float(i)) for i in range(30)])["rho"] is None
    assert correlation([(float(i), float(i)) for i in range(19)])["rho"] is None
    assert correlation([(float(i), -float(i)) for i in range(20)])["rho"] == pytest.approx(-1)


def test_network_density_and_strongest_derive_from_same_signed_relations():
    source = sample_source("external", rows=rows())
    sector = sample_source("sector", "etf", rows=rows(offset=2))
    days = [r["date"] for r in sector["observations"]]
    result = window_network([source], [sector], days, 60)
    valid = [r for r in result["relations"] if r["rho"] is not None]
    connected = [r for r in valid if r["connected"]]
    assert result["strongest"] == max(valid, key=lambda r: abs(r["rho"]))
    assert result["density"] == len(connected) / len(valid)
    assert result["possible_pairs"] == 1
    assert result["relations"][0]["n"] == 60


def test_complete_basket_formula_percentile_and_missing_fund():
    sources = [sample_source(s["id"], "etf", rows=rows(offset=i)) for i, s in enumerate(SECTORS)]
    history = basket_history(sources)
    assert history[0]["date"] == rows()[19]["date"]
    assert history[-1]["hhi"] == pytest.approx(0.2)
    assert history[-1]["synchrony_pairs"] == 10
    assert 0 <= history[-1]["breadth"] <= 100
    assert history[59]["dispersion_percentile"] is None
    assert history[60]["dispersion_percentile"] is not None
    sources[-1]["observations"] = []
    assert basket_history(sources) == []
    assert percentile(5, [5] * 60) == 50
    assert percentile(100, list(range(60))) == 100


def test_basket_returns_use_identical_periods_when_one_etf_day_is_missing():
    import statistics

    sources = [sample_source(s["id"], "etf", rows=rows(offset=i)) for i, s in enumerate(SECTORS)]
    missing_day = sources[0]["observations"][-2]["date"]
    sources[0]["observations"].pop(-2)
    actual = basket_history(sources)[-1]["dispersion"]
    aligned_changes = []
    for source in sources:
        aligned = [r for r in source["observations"] if r["date"] != missing_day]
        aligned_changes.append(aligned[-1]["close"] / aligned[-2]["close"] - 1)
    assert actual == pytest.approx(statistics.pstdev(aligned_changes) * 100)


def test_complete_session_cutoff_and_yahoo_identity():
    assert previous_complete_day(datetime(2026, 10, 19, 7, 29, tzinfo=UTC)) == "2026-10-18"
    assert previous_complete_day(datetime(2026, 10, 19, 7, 30, tzinfo=UTC)) == "2026-10-19"
    spec = FACTORS[0]
    ts = int(datetime(2026, 10, 19, 13, 30, tzinfo=UTC).timestamp())
    raw = {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "^NDX"},
                    "timestamp": [ts],
                    "indicators": {"quote": [{"close": [20000]}]},
                }
            ]
        }
    }
    assert parse_yahoo(raw, spec, datetime(2026, 10, 19, 18, tzinfo=UTC)) == []
    assert len(parse_yahoo(raw, spec, datetime(2026, 10, 19, 21, tzinfo=UTC))) == 1
    raw["chart"]["result"][0]["meta"]["symbol"] = "OTHER"
    with pytest.raises(ValueError):
        parse_yahoo(raw, spec, datetime(2026, 10, 20, tzinfo=UTC))


def test_xml_and_stale_source_summary():
    xml = (
        b'<feed xmlns:m="urn:m" xmlns:d="urn:d"><m:properties>'
        b"<d:NEW_DATE>2026-10-16T00:00:00</d:NEW_DATE>"
        b"<d:BC_10YEAR>4.2</d:BC_10YEAR></m:properties></feed>"
    )
    assert parse_treasury(xml, "2026-10-19")[0]["close"] == 4.2
    src = sample_source(
        rows=[{"date": "2026-10-15", "close": 4}, {"date": "2026-10-16", "close": 4.2}],
        kind="yield",
    )
    out = source_summary(src, datetime(2026, 10, 22, tzinfo=UTC))
    assert out["stale"] and out["change"] == pytest.approx(20)
    assert out["published_at"] is None


def test_fetch_failure_never_invents_values_and_preserves_actual_cache_time(tmp_path, monkeypatch):
    import json

    def failed(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests.Session, "get", failed)
    spec = FACTORS[0]
    now = datetime(2026, 10, 19, tzinfo=UTC)
    unavailable = download_source(spec, now, tmp_path)
    assert unavailable["status"] == "unavailable" and unavailable["observations"] == []
    cached = {
        **sample_source(spec["id"], rows=rows()),
        "symbol": spec["symbol"],
        "adjustment": "source_close",
        "receipts": [],
    }
    (tmp_path / f"{spec['id']}.json").write_text(json.dumps(cached), encoding="utf8")
    restored = download_source(spec, now, tmp_path)
    assert restored["status"] == "cached"
    assert restored["fetched_at"] == cached["fetched_at"]


def test_snapshot_missing_series_is_not_connected():
    sources = [sample_source(s["id"], s["kind"]) for s in FACTORS]
    sources.extend(
        sample_source(s["id"], "etf", rows=rows(offset=i)) for i, s in enumerate(SECTORS)
    )
    result = build_snapshot(sources, datetime(2026, 10, 19, tzinfo=UTC))
    assert result["basket"]["available"] == 5
    assert result["windows"]["60"]["valid_pairs"] == 0
    assert result["windows"]["60"]["density"] is None
    assert all(
        r["rho"] is None and not r["connected"] for r in result["windows"]["60"]["relations"]
    )


def test_bootstrap_from_real_bundle_retains_time_and_does_not_overwrite_cache(tmp_path):
    import json

    summary = {
        **FACTORS[0],
        "fetched_at": "2026-10-16T08:00:00+00:00",
        "adjustment": "source_close",
    }
    payload = {
        "schema_version": 1,
        "factors": [summary],
        "sectors": [],
        "source_observations": {"ndx": rows()},
    }
    source = tmp_path / "seed.json"
    source.write_text(json.dumps(payload), encoding="utf8")
    cache = tmp_path / "cache"
    seed_cache(cache, source)
    out = json.loads((cache / "ndx.json").read_text(encoding="utf8"))
    assert out["fetched_at"] == summary["fetched_at"]
    assert out["seeded_from"] == "bundled-public-snapshot"
    original = (cache / "ndx.json").read_bytes()
    payload["factors"][0]["fetched_at"] = "2099-01-01T00:00:00+00:00"
    source.write_text(json.dumps(payload), encoding="utf8")
    seed_cache(cache, source)
    assert (cache / "ndx.json").read_bytes() == original
