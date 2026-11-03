"""Offline integration of the new production job, frozen records and static export."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from recommend import daily_job
from recommend.daily_data import atomic_json
from recommend.daily_strategy import canonical_hash
from recommend.market import SHANGHAI_TIMEZONE


def test_real_job_freezes_five_picks_and_static_export_never_leaks_quotes(
    tmp_path,
    parquet_writer,
    monkeypatch,
):
    root = tmp_path
    (root / "config").mkdir()
    source = Path(__file__).parents[1]
    shutil.copy2(source / "config/daily_strategy.json", root / "config/daily_strategy.json")
    (root / "data/reference").mkdir(parents=True)
    (root / "data/market").mkdir()
    dates = pd.bdate_range(end="2026-11-02", periods=270)
    stocks = [f"60000{i}" for i in range(1, 7)]
    rows = []
    for i, symbol in enumerate(stocks):
        for n, day in enumerate(dates):
            close = 10 + n * (0.01 + i * 0.001)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "pre_close": close,
                    "volume": 2e7,
                    "amount": 2e8,
                }
            )
    parquet_writer(pd.DataFrame(rows), root / "data/market/daily.parquet")
    parquet_writer(
        pd.DataFrame(
            {
                "symbol": stocks,
                "sec_name": [f"公司{i}" for i in range(6)],
                "listed_date": pd.to_datetime(["2000-01-01"] * 6),
            }
        ),
        root / "data/reference/instrument_master.parquet",
    )
    pd.DataFrame({"symbol": stocks, "industry": [str(i) for i in range(6)]}).to_csv(
        root / "data/reference/industry_master.csv", index=False
    )

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 2, 17, tzinfo=SHANGHAI_TIMEZONE)

    def quotes(expected, day, current):
        result = {
            "as_of": day,
            "fetched_at": current.isoformat(),
            "coverage": 1,
            "quotes": {
                symbol: {
                    "symbol": symbol,
                    "name": f"公司{i}",
                    "status_date": day,
                    "open": 10,
                    "close": 10,
                    "volume": 10000,
                    "pre_close": 10,
                }
                for i, symbol in enumerate(expected)
            },
        }
        result["sha256"] = canonical_hash(result)
        return result

    monkeypatch.setattr(daily_job, "datetime", Clock)
    monkeypatch.setattr(
        daily_job,
        "advance_bars",
        lambda *_a, **_k: SimpleNamespace(rows=1, up_to_date=True, failed=[], notes=[]),
    )
    monkeypatch.setattr(daily_job, "fetch_quotes", quotes)
    monkeypatch.setattr(daily_job, "refresh_benchmark", lambda *_a: None)
    first = daily_job.run_daily(root)
    assert first["state"] == "live"
    assert len(first["selection"]["picks"]) == 5
    record = next((root / "data/reports/daily-v2/signals").glob("2026-*.json"))
    original = record.read_bytes()
    second = daily_job.run_daily(root, sync=False)
    assert record.read_bytes() == original
    assert second["validation"]["summary"]["signals"] == 1
    public = daily_job.public_daily_payload(root / "data", source / "site", root / "public")
    assert "pool_symbols" not in public["selection"]
    assert all(len(p["history"]) == 60 for p in public["selection"]["picks"])
    assert not (root / "public/quotes").exists()
    assert not (root / "public/archive.duckdb").exists()
    assert json.loads((root / "public/data.json").read_text(encoding="utf-8")) == public
    assert (root / "public/signals" / record.name).read_bytes() == original

    # A complete transport outage produces a dated historical page, never a fake live signal.
    quote_path = root / "data/reports/daily-v2/quotes/2026-11-02.json"
    missing = {
        "as_of": "2026-11-02",
        "fetched_at": Clock.now().isoformat(),
        "coverage": 0,
        "quotes": {},
    }
    missing["sha256"] = canonical_hash(missing)
    atomic_json(quote_path, missing)
    failed = daily_job.run_daily(root, sync=False)
    assert failed["state"] == "historical"
    assert record.read_bytes() == original

    # Restoring an old cloud price artifact cannot regress a newer frozen signal.
    parquet_writer(
        pd.DataFrame(rows).loc[lambda f: f.trade_date <= "2026-10-20"],
        root / "data/market/daily.parquet",
    )
    recovered = daily_job.run_daily(root, sync=False)
    assert recovered["as_of"] == "2026-11-02"
    assert len(recovered["selection"]["picks"]) == 5
    assert record.read_bytes() == original
