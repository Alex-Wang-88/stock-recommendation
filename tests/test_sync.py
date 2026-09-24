"""手动同步编排测试：源失败不应阻断其它阶段，且进度必须单调前进。"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from recommend import sync
from recommend.archive import ArchiveStore, BackfillResult
from recommend.bars import AdvanceResult
from recommend.config import AppConfig, MarketConfig


def test_manual_sync_runs_all_stages_and_persists_local_report(
    tmp_path, bars_factory, parquet_writer, monkeypatch
):
    daily = tmp_path / "daily.parquet"
    parquet_writer(
        bars_factory(n_days=300, symbols=("000001", "000002"), start="2025-01-01"),
        daily,
    )
    config = replace(
        AppConfig(),
        archive_db=tmp_path / "archive.duckdb",
        reports_dir=tmp_path / "reports",
        market=MarketConfig(daily_parquet=daily),
    )

    monkeypatch.setattr(
        sync,
        "advance_bars",
        lambda *_args, **_kwargs: AdvanceResult(
            up_to_date=True, previous_date="2026-02-24", quote_date="2026-02-24"
        ),
    )
    monkeypatch.setattr(
        sync,
        "backfill_themes",
        lambda *_args, **_kwargs: BackfillResult(task="themes", ok=1, attempted=1, rows=1),
    )
    monkeypatch.setattr(
        sync,
        "backfill_dragon_tiger",
        lambda *_args, **_kwargs: BackfillResult(task="dragon-tiger", ok=1, attempted=1, rows=1),
    )
    monkeypatch.setattr(
        sync,
        "backfill_margin_exchange",
        lambda *_args, **_kwargs: BackfillResult(task="margin", ok=1, attempted=1, rows=1),
    )
    monkeypatch.setattr(
        sync,
        "snapshot_all",
        lambda *_args, **_kwargs: [BackfillResult(task="sector", skipped=1)],
    )

    events = []
    result = sync.run_sync_all(config, progress=events.append)

    assert result.ok
    assert [step.key for step in result.steps] == [
        "market",
        "themes",
        "dragon-tiger",
        "margin",
        "sector",
        "recommendation",
        "validation",
    ]
    assert result.screen is not None
    assert result.report_path is not None and result.report_path.exists()
    assert result.csv_path is not None and result.csv_path.exists()
    assert result.forward_record_path is not None and result.forward_record_path.exists()
    assert result.evaluation_path is not None and result.evaluation_path.exists()
    assert result.evaluation_summary_path is not None and result.evaluation_summary_path.exists()
    assert [event.fraction for event in events] == sorted(event.fraction for event in events)


def test_auto_backfill_window_fills_internal_holes_before_tail(tmp_path):
    archive_path = tmp_path / "archive.duckdb"
    with ArchiveStore(archive_path) as archive:
        archive.upsert(
            "dragon_tiger",
            pd.DataFrame(
                [
                    {
                        "trade_date": "2026-09-17",
                        "symbol": "000001",
                        "name": "测试",
                        "explanation": "",
                        "close": 10.0,
                        "change_pct": 1.0,
                        "net_buy": 1.0,
                        "buy_amt": 2.0,
                        "sell_amt": 1.0,
                        "turnover_pct": 3.0,
                    }
                ]
            ),
            (
                "trade_date",
                "symbol",
                "name",
                "explanation",
                "close",
                "change_pct",
                "net_buy",
                "buy_amt",
                "sell_amt",
                "turnover_pct",
            ),
        )

        assert sync._auto_backfill_window(
            archive,
            "dragon_tiger",
            ["2026-09-16", "2026-09-17", "2026-09-18"],
            "2024-09-01",
        ) == ("2026-09-16", 2, "2026-09-17")


def test_auto_backfill_window_uses_fallback_only_for_empty_table(tmp_path):
    archive_path = tmp_path / "archive.duckdb"
    with ArchiveStore(archive_path) as archive:
        assert sync._auto_backfill_window(
            archive,
            "theme_attribution",
            ["2024-08-30", "2024-09-02", "2024-09-03"],
            "2024-09-01",
        ) == ("2024-09-02", 2, None)
