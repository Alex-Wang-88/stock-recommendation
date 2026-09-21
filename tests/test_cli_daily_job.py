"""`daily-job` 编排测试 —— 核心不变量是**步骤之间互不拖累**。

四步的"可恢复性"不同（行情/题材可补、T2 快照**补不回来**），所以：

* 行情或题材失败，**不能**阻止 T2 快照的尝试（T2 错过就永久没了）；
* T2 失败，**不能**把整次任务判为失败（调度器重试也补不回过去的那一天）；
* 筛选留证的 as-of 必须是**行情归档里的交易日**，不是系统日期。
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from recommend import cli
from recommend.archive import ArchiveStore, BackfillResult
from recommend.bars import AdvanceResult
from recommend.config import AppConfig, MarketConfig


def build_config(tmp_path) -> AppConfig:
    return replace(
        AppConfig(),
        archive_db=tmp_path / "archive.duckdb",
        logs_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        market=MarketConfig(daily_parquet=tmp_path / "daily.parquet"),
    )


def seed_market(config: AppConfig, bars_factory, parquet_writer) -> None:
    frame = bars_factory(n_days=6, symbols=("600519", "000001"), start="2026-09-09")
    parquet_writer(frame, config.market.daily_parquet)


def up_to_date_advance(*_a, **_k) -> AdvanceResult:
    """行情已是最新（测试里不碰网络）。"""

    return AdvanceResult(up_to_date=True, previous_date="2026-09-16")


def make_args(**overrides) -> argparse.Namespace:
    base = {
        "spec": None,
        "spec_file": None,
        "preset": "low-position",
        "theme": None,
        "out": None,
        "forward_dir": None,
        "workers": 2,
        "no_screen": False,
        "skip_themes": False,
        "skip_snapshot": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def failing_snapshot(*_a, **_k) -> list[BackfillResult]:
    return [BackfillResult(task="snapshot-all", attempted=1, failed=1)]


def test_daily_job_runs_every_step_and_reports_each_one(
    tmp_path, bars_factory, parquet_writer, monkeypatch
):
    config = build_config(tmp_path)
    seed_market(config, bars_factory, parquet_writer)

    monkeypatch.setattr(
        cli,
        "advance_bars",
        lambda store, workers=8: AdvanceResult(
            quote_date="2026-09-16", previous_date="2026-09-16", up_to_date=True
        ),
    )
    monkeypatch.setattr(cli, "snapshot_all", failing_snapshot)
    monkeypatch.setattr(
        cli,
        "backfill_themes",
        lambda *a, **k: BackfillResult(task="backfill-themes", attempted=3, ok=3, rows=9),
    )
    screened: list[str] = []

    def record(_config, _args, as_of):
        screened.append(as_of)
        return 0

    monkeypatch.setattr(cli, "_screen_and_record", record)

    assert cli.cmd_daily_job(config, make_args()) == 0
    assert screened == ["2026-09-16"]  # as-of 来自归档，不是系统日期


def test_t2_failure_does_not_fail_the_whole_job(
    tmp_path, bars_factory, parquet_writer, monkeypatch
):
    """T2 失败就判失败的话，调度器会一直重试 —— 而重试补不回已经过去的那一天。"""

    config = build_config(tmp_path)
    seed_market(config, bars_factory, parquet_writer)
    monkeypatch.setattr(cli, "advance_bars", up_to_date_advance)
    monkeypatch.setattr(cli, "snapshot_all", failing_snapshot)
    monkeypatch.setattr(cli, "backfill_themes", lambda *a, **k: BackfillResult(task="t", ok=1))
    monkeypatch.setattr(cli, "_screen_and_record", lambda *a, **k: 0)

    assert cli.cmd_daily_job(config, make_args()) == 0


def test_theme_backfill_starts_after_the_last_archived_theme_day(
    tmp_path, bars_factory, parquet_writer, monkeypatch
):
    """题材补抓的区间必须**接着归档最后一天往后**，且只覆盖交易日。

    从固定的"近 N 天"起算会每天重复抓同样的日期；用自然日回推又会撞上节假日。
    """

    config = build_config(tmp_path)
    seed_market(config, bars_factory, parquet_writer)
    store = ArchiveStore(config.archive_db)
    try:
        store.upsert(
            "theme_attribution",
            pd.DataFrame(
                {
                    "trade_date": [pd.Timestamp("2026-09-11").date()],
                    "symbol": ["600519"],
                    "name": ["贵州茅台"],
                    "reason": ["测试"],
                    "source_date": [pd.Timestamp("2026-09-11").date()],
                }
            ),
            ("trade_date", "symbol", "name", "reason", "source_date"),
        )
    finally:
        store.close()

    monkeypatch.setattr(cli, "advance_bars", up_to_date_advance)
    monkeypatch.setattr(cli, "snapshot_all", lambda *a, **k: [])
    monkeypatch.setattr(cli, "_screen_and_record", lambda *a, **k: 0)
    captured: dict[str, str] = {}

    def fake_backfill(_store, _client, start, end, **_kwargs):
        captured["start"] = start
        captured["end"] = end
        return BackfillResult(task="backfill-themes", ok=1)

    monkeypatch.setattr(cli, "backfill_themes", fake_backfill)

    assert cli.cmd_daily_job(config, make_args()) == 0
    assert captured["start"] == "2026-09-14"  # 09-11 之后的下一个**交易日**（跳过周末）
    assert captured["end"] == "2026-09-16"


def test_theme_step_is_skipped_when_already_current(
    tmp_path, bars_factory, parquet_writer, monkeypatch
):
    config = build_config(tmp_path)
    seed_market(config, bars_factory, parquet_writer)
    called: list[bool] = []
    monkeypatch.setattr(cli, "advance_bars", up_to_date_advance)
    monkeypatch.setattr(cli, "snapshot_all", lambda *a, **k: [])
    monkeypatch.setattr(cli, "_screen_and_record", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "backfill_themes", lambda *a, **k: called.append(True))
    # 题材归档已到 09-16（= 行情最新日）⇒ 不该发起任何抓取。
    store = ArchiveStore(config.archive_db)
    try:
        store.upsert(
            "theme_attribution",
            pd.DataFrame(
                {
                    "trade_date": [pd.Timestamp("2026-09-16").date()],
                    "symbol": ["600519"],
                    "name": ["贵州茅台"],
                    "reason": ["测试"],
                    "source_date": [pd.Timestamp("2026-09-16").date()],
                }
            ),
            ("trade_date", "symbol", "name", "reason", "source_date"),
        )
    finally:
        store.close()

    assert cli.cmd_daily_job(config, make_args()) == 0
    assert called == []


def boom(*_args, **_kwargs):
    raise AssertionError("该步骤本应被跳过，不应被调用")


def test_steps_can_be_skipped_explicitly(tmp_path, bars_factory, parquet_writer, monkeypatch):
    config = build_config(tmp_path)
    seed_market(config, bars_factory, parquet_writer)
    monkeypatch.setattr(cli, "advance_bars", up_to_date_advance)
    for name in ("snapshot_all", "backfill_themes", "_screen_and_record"):
        monkeypatch.setattr(cli, name, boom)

    args = make_args(skip_themes=True, skip_snapshot=True, no_screen=True)
    assert cli.cmd_daily_job(config, args) == 0


def test_daily_job_refuses_to_run_without_a_market_archive(tmp_path, monkeypatch):
    config = build_config(tmp_path)  # 不落盘 daily.parquet
    monkeypatch.setattr(cli, "advance_bars", boom)
    assert cli.cmd_daily_job(config, make_args()) == 1
