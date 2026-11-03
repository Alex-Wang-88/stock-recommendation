"""CLI dispatch tests target the actual registered runner; never contact live sources."""

from __future__ import annotations

import argparse
from dataclasses import replace
from types import SimpleNamespace

from recommend import cli
from recommend.config import AppConfig, MarketConfig


def arguments(**overrides):
    values = {
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
    return argparse.Namespace(**{**values, **overrides})


def test_daily_job_dispatches_to_registered_runner(monkeypatch):
    captured = {}

    def run(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(exit_ok=True, steps=[])

    monkeypatch.setattr(cli, "intraday_sync_block_reason", lambda: None)
    monkeypatch.setattr(cli, "run_sync_all", run)
    assert cli.cmd_daily_job(AppConfig(), arguments()) == 0
    assert captured["workers"] == 2
    assert captured["preset"] == "low-position"
    assert captured["skip_sources"] == set()


def test_registered_runner_failure_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(cli, "intraday_sync_block_reason", lambda: None)
    monkeypatch.setattr(
        cli, "run_sync_all", lambda *_a, **_k: SimpleNamespace(exit_ok=False, steps=[])
    )
    assert cli.cmd_daily_job(AppConfig(), arguments()) == 1


def test_skip_options_reach_actual_source_registry(monkeypatch):
    captured = {}

    def run(_config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(exit_ok=True, steps=[])

    monkeypatch.setattr(cli, "intraday_sync_block_reason", lambda: None)
    monkeypatch.setattr(cli, "run_sync_all", run)
    assert (
        cli.cmd_daily_job(
            AppConfig(), arguments(skip_themes=True, skip_snapshot=True, no_screen=True)
        )
        == 0
    )
    assert captured["skip_sources"] == {"themes", "sector"}
    assert captured["skip_recommendation"] and captured["skip_validation"]


def test_intraday_cli_does_not_start_sync(monkeypatch):
    monkeypatch.setattr(cli, "intraday_sync_block_reason", lambda: "market open")

    def forbidden(*_a, **_k):
        raise AssertionError("sync must not run")

    monkeypatch.setattr(cli, "run_sync_all", forbidden)
    assert cli.cmd_daily_job(AppConfig(), arguments()) == 0


def test_missing_market_archive_reports_failure_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "intraday_sync_block_reason", lambda: None)
    config = replace(
        AppConfig(),
        archive_db=tmp_path / "archive.duckdb",
        logs_dir=tmp_path / "logs",
        market=MarketConfig(daily_parquet=tmp_path / "absent.parquet"),
    )
    assert cli.cmd_daily_job(config, arguments()) == 1
