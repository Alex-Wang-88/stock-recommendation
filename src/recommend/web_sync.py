"""Non-blocking synchronization state for the Streamlit workbench."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig
from .sync import SyncProgress, SyncRunResult, run_sync_all


@dataclass
class SyncJobState:
    lock: threading.RLock = field(default_factory=threading.RLock)
    startup_started: bool = False
    running: bool = False
    kind: str = ""
    preset: str | None = None
    result: SyncRunResult | None = None
    progress: SyncProgress | None = None
    error: str | None = None
    last_result_id: str | None = None


_STATE = SyncJobState()


def snapshot() -> dict[str, Any]:
    """Return a thread-safe copy of the current web synchronization state."""

    with _STATE.lock:
        return {
            "running": _STATE.running,
            "kind": _STATE.kind,
            "preset": _STATE.preset,
            "result": _STATE.result,
            "progress": _STATE.progress,
            "error": _STATE.error,
        }


def _worker(config: AppConfig, preset: str) -> None:
    def on_progress(event: SyncProgress) -> None:
        with _STATE.lock:
            _STATE.progress = event

    try:
        result = run_sync_all(config, preset=preset, progress=on_progress)
    except Exception as error:  # noqa: BLE001 - show worker failures in the page
        with _STATE.lock:
            _STATE.running = False
            _STATE.error = f"{type(error).__name__}: {error}"
        return

    with _STATE.lock:
        _STATE.result = result
        _STATE.running = False
        _STATE.error = None
        _STATE.progress = SyncProgress(
            "sync-done",
            "同步流程结束",
            1.0,
            "同步流程结束",
            "done" if result.ok else "failed",
        )


def start(config: AppConfig, preset: str, *, startup: bool = False) -> bool:
    """Start one background synchronization, unless one is already running."""

    with _STATE.lock:
        if startup and _STATE.startup_started:
            return False
        if _STATE.running:
            return False
        if startup:
            _STATE.startup_started = True
        _STATE.running = True
        _STATE.kind = "启动时自动同步" if startup else "手动同步"
        _STATE.preset = preset
        _STATE.result = None
        _STATE.error = None
        _STATE.progress = SyncProgress(
            "sync-start",
            _STATE.kind,
            0.0,
            "后台同步任务已启动，等待数据源响应…",
        )
        thread = threading.Thread(
            target=_worker,
            args=(config, preset),
            daemon=True,
            name="recommend-sync-worker",
        )
        try:
            thread.start()
        except Exception as error:  # noqa: BLE001 - do not leave the page in a running state
            _STATE.running = False
            _STATE.error = f"{type(error).__name__}: {error}"
            return False
    return True


def run_sync_with_feedback(
    config: AppConfig,
    preset: str,
    *,
    startup: bool = False,
) -> SyncRunResult | None:
    """Compatibility wrapper: start asynchronously and return a finished result."""

    start(config, preset, startup=startup)
    current = snapshot()["result"]
    return current if isinstance(current, SyncRunResult) else None


def _render_result(st: Any, result: SyncRunResult) -> None:
    if any(step.key == "market-window" for step in result.steps):
        st.info(
            "当前处于同步保护时段；本次未发起数据请求，"
            "页面继续显示本地最近完整交易日。"
        )
    elif any(
        step.key == "recommendation" and step.status == "skipped" for step in result.steps
    ):
        st.info(
            "尚无今天已确认收盘的行情；本次只处理可回补数据，"
            "未生成新的推荐或前向留证。"
        )
    elif result.ok and not result.has_warnings:
        st.success("同步完成，推荐已按最新本地归档重新计算。")
    elif result.ok:
        st.warning("同步流程完成，但有数据源返回警告；请查看下方步骤明细。")
    else:
        st.error("同步未完整完成；本地推荐仍保留可用结果，请查看步骤明细。")
    for step in result.steps:
        icon = {
            "done": "完成",
            "warning": "警告",
            "failed": "失败",
            "skipped": "延后",
        }.get(step.status, step.status)
        st.write(f"{icon} · {step.label}：{step.summary}")
        if step.detail:
            st.caption(step.detail)


def _render_status_body(st: Any) -> None:
    state = snapshot()
    progress = state["progress"]
    result = state["result"]
    if state["running"]:
        st.markdown(
            f"<div class='section-label'>{state['kind'] or '同步'}</div>",
            unsafe_allow_html=True,
        )
        if isinstance(progress, SyncProgress):
            st.progress(progress.fraction, text=f"{progress.label} · {progress.message}")
            st.caption(f"{progress.status.upper()} / {progress.label} / {progress.message}")
        else:
            st.progress(0.0, text="同步任务已启动，等待数据源响应…")
        st.info(
            "同步正在后台执行，网页已经可以打开并继续查看上一次本地推荐；"
            "同步完成后页面会自动刷新。"
        )
        return

    if state["error"]:
        st.markdown("<div class='section-label'>同步异常</div>", unsafe_allow_html=True)
        st.error(f"后台同步线程异常：{state['error']}")
        st.info("本地推荐仍可查看；点击“再次同步全部数据”可重试。")
        return

    if not isinstance(result, SyncRunResult):
        return
    result_id = result.sync_run_id or result.finished_at or str(id(result))
    should_refresh = False
    with _STATE.lock:
        if _STATE.last_result_id != result_id:
            _STATE.last_result_id = result_id
            should_refresh = True
    if should_refresh:
        st.rerun()
        return
    st.markdown(
        f"<div class='section-label'>{state['kind'] or '同步'}结果</div>",
        unsafe_allow_html=True,
    )
    st.progress(1.0, text="同步流程结束")
    _render_result(st, result)


def render_status(st: Any) -> None:
    """Render status in a periodic fragment when the installed Streamlit supports it."""

    if hasattr(st, "fragment"):
        fragment = st.fragment(run_every="2s")(_render_status_body)
        fragment(st)
    else:  # pragma: no cover - compatibility with older Streamlit
        _render_status_body(st)
