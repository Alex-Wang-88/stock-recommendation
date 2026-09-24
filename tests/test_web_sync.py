"""Web synchronization must not block the Streamlit script runner."""

from __future__ import annotations

import threading
import time

from recommend import web_sync
from recommend.config import AppConfig
from recommend.sync import SyncProgress, SyncRunResult


def _reset_state() -> None:
    with web_sync._STATE.lock:
        web_sync._STATE.startup_started = False
        web_sync._STATE.running = False
        web_sync._STATE.kind = ""
        web_sync._STATE.preset = None
        web_sync._STATE.result = None
        web_sync._STATE.progress = None
        web_sync._STATE.error = None
        web_sync._STATE.last_result_id = None


def _wait_until_finished(timeout: float = 2.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = web_sync.snapshot()
        if not state["running"]:
            return state
        time.sleep(0.01)
    raise AssertionError("background synchronization did not finish")


def test_start_runs_in_background_and_deduplicates(monkeypatch) -> None:
    _reset_state()
    release = threading.Event()

    def fake_sync(config, *, preset, progress):
        del config, preset
        progress(SyncProgress("market", "行情日线", 0.1, "测试中"))
        assert release.wait(timeout=2)
        return SyncRunResult(started_at="now", sync_run_id="test-run", finished_at="done")

    monkeypatch.setattr(web_sync, "run_sync_all", fake_sync)
    try:
        assert web_sync.start(AppConfig(), "low-position", startup=True)
        assert web_sync.start(AppConfig(), "low-position", startup=True) is False
        assert web_sync.snapshot()["running"] is True
        release.set()
        state = _wait_until_finished()
        assert state["result"].sync_run_id == "test-run"
        assert state["error"] is None
    finally:
        _reset_state()
