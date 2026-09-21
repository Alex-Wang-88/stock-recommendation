"""同步编排：把远端抓取、项目内归档、推荐报告与前向验证串成一次可观察运行。

网页端只调用这里，不直接操作各个数据源。这样同步按钮和命令行仍共用同一套
数据口径，同时把「网络抓到了什么」与「推荐读取什么」分开：推荐始终只从本项目
目录里的 parquet / DuckDB / report 读取，网络请求的结果先落盘再进入筛选。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from .archive import (
    ArchiveStore,
    BackfillResult,
    backfill_dragon_tiger,
    backfill_margin_exchange,
    backfill_themes,
    snapshot_all,
)
from .bars import advance_bars
from .config import AppConfig
from .context import ScreenContext
from .forward import append_record, build_record, evaluate_and_persist
from .fundamentals import sync_baostock
from .http import ThrottledClient
from .market import MarketStore, is_current_eod_ready
from .report import render_markdown
from .screen import Spec, run_screen
from .screen.presets import load_preset

BASE_SYNC_STAGES = (
    ("market", "行情日线"),
    ("recommendation", "生成推荐"),
    ("validation", "推荐留证与验证"),
)


@dataclass(frozen=True)
class SyncProgress:
    """同步进度事件，供 Streamlit 或其它前端消费。"""

    stage: str
    label: str
    fraction: float
    message: str
    status: str = "running"


@dataclass
class SyncStep:
    key: str
    label: str
    status: str
    summary: str
    detail: str = ""


@dataclass
class SyncRunResult:
    started_at: str
    finished_at: str = ""
    as_of: str | None = None
    steps: list[SyncStep] = field(default_factory=list)
    screen: object | None = None
    report_path: Path | None = None
    csv_path: Path | None = None
    forward_record_path: Path | None = None
    forward_journal_path: Path | None = None
    evaluation_path: Path | None = None
    evaluation_summary_path: Path | None = None
    validation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(step.status == "failed" for step in self.steps)

    @property
    def has_warnings(self) -> bool:
        return any(step.status == "warning" for step in self.steps)


ProgressCallback = Callable[[SyncProgress], None]


@dataclass(frozen=True)
class SyncSource:
    """一个可插拔同步源。

    新增数据源只需在 ``SYNC_SOURCE_REGISTRY`` 注册一个 runner；网页同步和其它
    使用 ``run_sync_all`` 的入口会自动得到该源的步骤、进度和失败隔离。
    """

    key: str
    label: str
    runner: Callable[..., tuple[str, str, str]]
    enabled: Callable[[AppConfig], bool] = lambda _config: True


class _SkipSyncStage(Exception):
    """内部控制流：把可选同步阶段标记为 skipped 而不是 failed。"""


def _sync_t1_source(
    config: AppConfig,
    archive: ArchiveStore,
    client: ThrottledClient,
    as_of: str,
    trade_dates: list[str],
    emit_inside: Callable[[float, str, str], None],
    *,
    key: str,
    label: str,
    table: str,
    fallback_start: str,
    fetcher: Callable[..., BackfillResult],
) -> tuple[str, str, str]:
    del config, as_of
    start_day, pending_count, last_archived = _auto_backfill_window(
        archive, table, trade_dates, fallback_start
    )
    if start_day is None:
        last_label = last_archived or f"未达到初始化起点 {fallback_start}"
        return "done", f"已追平到本地行情最新日（最后归档 {last_label}），跳过抓取", ""

    last_label = last_archived or f"未初始化（起点 {fallback_start}）"
    end_day = trade_dates[-1]
    emit_inside(
        0.15,
        f"自动追平：{last_label} → {end_day}，约 {pending_count} 个交易日…",
        "running",
    )
    try:
        fetched = fetcher(archive, client, start_day, end_day)
    except Exception as error:  # noqa: BLE001 - 单源失败不能阻断其它源
        return "warning", f"{label}同步失败：{type(error).__name__}: {error}", ""
    status = _backfill_status(fetched)
    return status, fetched.summary(), ""


def _sync_themes(
    config: AppConfig,
    archive: ArchiveStore,
    client: ThrottledClient,
    as_of: str,
    trade_dates: list[str],
    emit_inside: Callable[[float, str, str], None],
) -> tuple[str, str, str]:
    return _sync_t1_source(
        config,
        archive,
        client,
        as_of,
        trade_dates,
        emit_inside,
        key="themes",
        label="题材归因",
        table="theme_attribution",
        fallback_start=config.backfill.themes_start,
        fetcher=backfill_themes,
    )


def _sync_dragon_tiger(
    config: AppConfig,
    archive: ArchiveStore,
    client: ThrottledClient,
    as_of: str,
    trade_dates: list[str],
    emit_inside: Callable[[float, str, str], None],
) -> tuple[str, str, str]:
    return _sync_t1_source(
        config,
        archive,
        client,
        as_of,
        trade_dates,
        emit_inside,
        key="dragon-tiger",
        label="龙虎榜",
        table="dragon_tiger",
        fallback_start=config.backfill.dragon_tiger_start,
        fetcher=backfill_dragon_tiger,
    )


def _sync_margin(
    config: AppConfig,
    archive: ArchiveStore,
    client: ThrottledClient,
    as_of: str,
    trade_dates: list[str],
    emit_inside: Callable[[float, str, str], None],
) -> tuple[str, str, str]:
    return _sync_t1_source(
        config,
        archive,
        client,
        as_of,
        trade_dates,
        emit_inside,
        key="margin",
        label="官方两融",
        table="margin_trading",
        fallback_start=config.backfill.margin_start,
        fetcher=backfill_margin_exchange,
    )


def _sync_sector(
    config: AppConfig,
    archive: ArchiveStore,
    client: ThrottledClient,
    as_of: str,
    trade_dates: list[str],
    emit_inside: Callable[[float, str, str], None],
) -> tuple[str, str, str]:
    del config, trade_dates
    emit_inside(0.0, f"尝试保存 {as_of} 板块快照…", "running")
    try:
        results = snapshot_all(archive, client, as_of)
        status = _snapshot_status(results)
        summary = "；".join(item.summary() for item in results) or "无板块快照结果"
        return status, summary, ""
    except Exception as error:  # noqa: BLE001 - 单源失败不能阻断其它源
        return "warning", f"板块快照同步失败：{type(error).__name__}: {error}", ""


def _sync_fundamentals(
    config: AppConfig,
    archive: ArchiveStore,
    client: ThrottledClient,
    as_of: str,
    trade_dates: list[str],
    emit_inside: Callable[[float, str, str], None],
) -> tuple[str, str, str]:
    del client, trade_dates
    try:
        with MarketStore(config.market.daily_parquet) as market:
            latest = market.bars(end=as_of, window=1)
        latest = latest.loc[latest["trade_date"].astype(str).str[:10] == as_of].copy()
        symbols = latest["symbol"].tolist()
        if not symbols:
            return "warning", "BaoStock：本地行情最新日没有可用股票代码", ""
        close_prices = dict(zip(latest["symbol"].astype(str), latest["close"], strict=False))

        def on_progress(index: int, count: int, message: str) -> None:
            # 财务请求完成后才构造估值，两个阶段映射到连续区间，确保网页进度条
            # 不会倒退；如果财务已覆盖，直接跳到估值归档阶段。
            if "财务" in message:
                base, span = 0.04, 0.80
            elif "估值" in message:
                base, span = 0.88, 0.10
            else:
                base, span = 0.04, 0.90
            emit_inside(base + span * index / max(count, 1), message, "running")

        synced = sync_baostock(
            archive,
            as_of=as_of,
            symbols=symbols,
            close_prices=close_prices,
            start_date=config.fundamentals.start_date,
            financial_refresh_days=config.fundamentals.financial_refresh_days,
            workers=config.fundamentals.workers,
            history_periods=config.fundamentals.history_periods,
            progress=on_progress,
        )
        status = "warning" if synced.warning else "done"
        return status, synced.summary(), "；".join(
            f"{target}: {detail}" for target, detail in synced.failures[:5]
        )
    except Exception as error:  # noqa: BLE001 - 依赖未安装/登录失败不能阻断其它源
        return "warning", f"BaoStock 同步失败：{type(error).__name__}: {error}", ""


# 所有会被「同步全部数据」调用的数据源统一登记在这里。新增数据源时只需追加
# 一个 SyncSource；阶段、进度、网页明细和失败隔离会自动纳入。
SYNC_SOURCE_REGISTRY: tuple[SyncSource, ...] = (
    SyncSource("themes", "题材归因", _sync_themes),
    SyncSource("dragon-tiger", "龙虎榜", _sync_dragon_tiger),
    SyncSource("margin", "官方两融", _sync_margin),
    SyncSource("sector", "板块快照", _sync_sector),
    SyncSource(
        "fundamentals",
        "估值与财务（BaoStock）",
        _sync_fundamentals,
        lambda c: c.fundamentals.enabled,
    ),
)


def _active_sources(config: AppConfig) -> tuple[SyncSource, ...]:
    return tuple(source for source in SYNC_SOURCE_REGISTRY if source.enabled(config))


def _stages_for(config: AppConfig) -> tuple[tuple[str, str], ...]:
    sources = tuple((source.key, source.label) for source in _active_sources(config))
    return (
        ("market", "行情日线"),
        *sources,
        ("recommendation", "生成推荐"),
        ("validation", "推荐留证与验证"),
    )


# 对外保留完整注册清单，便于状态页、测试和未来扩展者查看。
SYNC_STAGES = (
    ("market", "行情日线"),
    *((source.key, source.label) for source in SYNC_SOURCE_REGISTRY),
    ("recommendation", "生成推荐"),
    ("validation", "推荐留证与验证"),
)


def run_sync_all(
    config: AppConfig,
    *,
    preset: str = "low-position",
    workers: int = 8,
    progress: ProgressCallback | None = None,
    skip_sources: set[str] | None = None,
    skip_recommendation: bool = False,
    skip_validation: bool = False,
) -> SyncRunResult:
    """同步所有数据类型，并生成一份项目内的最新推荐报告。

    T1 数据不再依赖手动配置的固定回补窗口：每张归档表都会自动读取最后一个
    已成功写入的交易日，从下一交易日追平到本地行情最新日。首次运行某张空表时，
    才使用该数据源的 ``backfill`` 起始日作为初始化下界。任何一个远端端点失败都会
    留下 warning 并继续其它步骤；只有本地行情不存在或推荐无法计算才是 fatal failure。
    """

    started = datetime.now().isoformat(timespec="seconds")
    result = SyncRunResult(started_at=started)
    stages = _stages_for(config)
    stage_labels = dict(stages)
    sources = _active_sources(config)
    skipped_sources = skip_sources or set()
    total = len(stages)

    def emit(stage: str, fraction: float, message: str, status: str = "running") -> None:
        if progress is None:
            return
        label = stage_labels.get(stage, stage)
        progress(
            SyncProgress(
                stage=stage,
                label=label,
                fraction=min(1.0, max(0.0, fraction)),
                message=message,
                status=status,
            )
        )

    def stage_fraction(index: int, inside: float = 1.0) -> float:
        return (index + min(1.0, max(0.0, inside))) / total

    def add_step(key: str, status: str, summary: str, detail: str = "") -> None:
        result.steps.append(
            SyncStep(
                key=key,
                label=stage_labels[key],
                status=status,
                summary=summary,
                detail=detail,
            )
        )

    # 1. 行情日线 ---------------------------------------------------------
    emit("market", 0.0, "检查本地行情归档…")
    market_path = Path(config.market.daily_parquet)
    if not market_path.exists():
        message = f"行情归档不存在：{market_path}。请先把项目内的 source_daily.parquet 引导进来。"
        add_step("market", "failed", message)
        result.notes.append(message)
        emit("market", stage_fraction(0), message, "failed")
        result.finished_at = datetime.now().isoformat(timespec="seconds")
        return result

    try:
        with MarketStore(market_path) as store:
            advance = advance_bars(
                store,
                workers=workers,
                progress=lambda index, count, phase: emit(
                    "market",
                    stage_fraction(
                        0,
                        (
                            0.05 + 0.70 * index / max(count, 1)
                            if phase == "initial"
                            else 0.75 + 0.20 * index / max(count, 1)
                        ),
                    ),
                    f"行情抓取 {phase}：{index}/{count}",
                ),
            )
            raw_market_as_of = store.latest_trade_date()
            result.as_of = store.latest_complete_trade_date()
            trade_dates = store.trade_dates(end=result.as_of)
            current_eod_ready = is_current_eod_ready(raw_market_as_of)
        market_status = "done" if advance.up_to_date or advance.rows else "warning"
        market_summary = advance.summary().splitlines()[0]
        market_detail = "；".join(advance.notes)
        if result.as_of != raw_market_as_of:
            market_detail = "；".join(
                item
                for item in (
                    market_detail,
                    (
                        f"行情源已出现 {raw_market_as_of} 的盘中数据，但尚未过 15:30；"
                        f"本次推荐和收盘后数据暂按最近完整交易日 {result.as_of}，"
                        "收盘后重试即可继续追平。"
                    ),
                )
                if item
            )
        add_step("market", market_status, market_summary, market_detail)
        emit("market", stage_fraction(0), market_summary, market_status)
    except Exception as error:  # noqa: BLE001 - UI 要继续汇报其它可执行步骤
        message = f"行情同步失败：{type(error).__name__}: {error}"
        add_step("market", "failed", message)
        result.notes.append(message)
        emit("market", stage_fraction(0), message, "failed")
        result.finished_at = datetime.now().isoformat(timespec="seconds")
        return result

    # 2. 注册数据源：所有源共用同一套进度、失败隔离和本地归档生命周期。
    archive = ArchiveStore(config.archive_db)
    client = ThrottledClient(
        min_interval=config.http.min_interval_seconds,
        jitter=config.http.jitter_seconds,
        retries=config.http.retries,
        backoff=config.http.backoff_seconds,
        timeout=config.http.timeout_seconds,
    )
    try:
        for stage_index, source in enumerate(sources, start=1):
            key = source.key
            label = source.label
            if key in skipped_sources:
                summary = "已按调用方要求跳过"
                add_step(key, "skipped", summary)
                emit(key, stage_fraction(stage_index), summary, "skipped")
                continue
            if key == "sector" and not current_eod_ready:
                summary = (
                    "当前没有可安全归档的已收盘板块快照；"
                    f"本次使用 {result.as_of}，下一个交易日收盘后再同步板块快照"
                )
                add_step(key, "skipped", summary)
                emit(key, stage_fraction(stage_index), summary, "skipped")
                continue
            emit(key, stage_fraction(stage_index, 0.0), f"检查{label}同步进度…")

            def emit_inside(
                inside: float,
                message: str,
                status: str = "running",
                *,
                source_key: str = key,
                source_index: int = stage_index,
            ) -> None:
                emit(source_key, stage_fraction(source_index, inside), message, status)

            try:
                status, summary, detail = source.runner(
                    config,
                    archive,
                    client,
                    result.as_of,
                    trade_dates,
                    emit_inside,
                )
            except Exception as error:  # noqa: BLE001 - 单源失败不阻断其它源
                status = "warning"
                summary = f"{label}同步失败：{type(error).__name__}: {error}"
                detail = ""
            add_step(key, status, summary, detail)
            if status == "warning":
                result.notes.append(summary)
            emit(key, stage_fraction(stage_index), summary, status)
    finally:
        archive.close()

    # 3. 生成报告 ---------------------------------------------------------
    recommendation_index = len(sources) + 1
    validation_index = recommendation_index + 1
    if skip_recommendation:
        summary = "已按调用方要求跳过推荐生成"
        add_step("recommendation", "skipped", summary)
        emit("recommendation", stage_fraction(recommendation_index), summary, "skipped")
    else:
        emit("recommendation", stage_fraction(recommendation_index, 0.0), "用本地归档重新计算推荐…")
    try:
        if skip_recommendation:
            raise _SkipSyncStage
        payload = load_preset(preset)
        payload["as_of"] = result.as_of
        spec = Spec.from_dict(payload)
        with ScreenContext.open(config, spec.as_of) as context:
            table = context.build()
            spec = replace(spec, as_of=table.as_of)
            screened = run_screen(table, spec, config.rank.as_dict())

        reports_dir = Path(config.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{screened.as_of}.md"
        csv_path = reports_dir / f"{screened.as_of}.csv"
        report_path.write_text(render_markdown(screened), encoding="utf-8")
        screened.frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        result.screen = screened
        result.report_path = report_path
        result.csv_path = csv_path
        summary = f"as-of {screened.as_of}，输出 {screened.kept} 只推荐；报告已写入 {report_path}"
        status = "done" if screened.kept else "warning"
        add_step("recommendation", status, summary, "；".join(screened.notes))
        emit("recommendation", stage_fraction(recommendation_index), summary, status)
    except _SkipSyncStage:
        pass
    except Exception as error:  # noqa: BLE001
        message = f"推荐生成失败：{type(error).__name__}: {error}"
        add_step("recommendation", "failed", message)
        result.notes.append(message)
        emit("recommendation", stage_fraction(5), message, "failed")

    # 4. 推荐留证与前向验证 -----------------------------------------------
    if skip_validation:
        summary = "已按调用方要求跳过推荐留证与验证"
        add_step("validation", "skipped", summary)
        emit("validation", stage_fraction(validation_index), summary, "skipped")
    else:
        emit("validation", stage_fraction(validation_index, 0.0), "记录本次推荐并更新历史验证…")
    if skip_validation:
        pass
    elif result.screen is None:
        message = "推荐未生成，无法创建前向留证"
        add_step("validation", "warning", message)
        result.notes.append(message)
        emit("validation", stage_fraction(validation_index), message, "warning")
    else:
        try:
            forward_dir = Path(config.reports_dir) / "forward"
            record = build_record(
                result.screen,
                run_at=datetime.now().isoformat(timespec="seconds"),
            )
            record_path, journal_path = append_record(forward_dir, record)
            with MarketStore(market_path) as store:
                _per_run, grouped, evaluation_path, summary_path = evaluate_and_persist(
                    store,
                    forward_dir,
                )
            result.forward_record_path = record_path
            result.forward_journal_path = journal_path
            result.evaluation_path = evaluation_path
            result.evaluation_summary_path = summary_path
            result.validation_summary = grouped
            summary = (
                f"已记录 {record.kept} 只推荐；"
                f"{_validation_summary_text(grouped)}"
            )
            detail = (
                f"留证：{record_path}；逐次验证：{evaluation_path}；"
                f"持有期汇总：{summary_path}"
            )
            add_step("validation", "done", summary, detail)
            emit("validation", stage_fraction(validation_index), summary, "done")
        except Exception as error:  # noqa: BLE001 - 留证失败不抹掉当次推荐
            message = f"推荐留证/验证失败：{type(error).__name__}: {error}"
            add_step("validation", "warning", message)
            result.notes.append(message)
            emit("validation", 1.0, message, "warning")

    result.finished_at = datetime.now().isoformat(timespec="seconds")
    return result


def _backfill_status(result: BackfillResult) -> str:
    if result.failed:
        return "warning"
    if result.rejected and not result.ok:
        return "warning"
    if result.attempted and not result.ok and not result.skipped:
        return "warning"
    return "done"


def _validation_summary_text(grouped: pd.DataFrame) -> str:
    """把验证汇总压缩成同步步骤里的一行提示。"""

    if grouped.empty:
        return "暂无可测的后续交易日"
    parts: list[str] = []
    for row in grouped.itertuples(index=False):
        measured = int(row.measured or 0)
        excess = row.excess_mean
        if pd.isna(excess):
            result = "待测"
        else:
            result = f"超额 {float(excess):+.2%}"
        parts.append(f"{int(row.horizon)}日 {measured} 次/{result}")
    return "；".join(parts)


def _auto_backfill_window(
    archive: ArchiveStore,
    table: str,
    trade_dates: list[str],
    fallback_start: str,
) -> tuple[str | None, int, str | None]:
    """根据本地行情与归档进度计算下一段自动追平区间。

    已有数据时只从该表最后一个成功归档日之后继续，避免每次启动重复扫描多年
    历史；空表才从数据源允许的初始化起点开始。返回值依次是起始日、待追平的
    本地交易日数量、最后归档日。

    这里故意不把 T2 快照纳入计划：板块快照只能保存当前日，无法历史回补。
    """

    done = archive.archived_dates(table)
    last_archived = max(done) if done else None
    if last_archived is None:
        pending = [day for day in trade_dates if day >= fallback_start]
    else:
        pending = [day for day in trade_dates if day > last_archived]
    if not pending:
        return None, 0, last_archived
    return pending[0], len(pending), last_archived


def _snapshot_status(results: list[BackfillResult]) -> str:
    if any(item.failed for item in results):
        return "warning"
    if any(item.rejected for item in results):
        return "warning"
    return "done"


def coverage_snapshot(config: AppConfig) -> pd.DataFrame:
    """读取当前项目内归档覆盖，供前端展示，不触发网络请求。"""

    with ArchiveStore(config.archive_db) as store:
        return store.coverage()


__all__ = [
    "SYNC_STAGES",
    "SYNC_SOURCE_REGISTRY",
    "SyncProgress",
    "SyncRunResult",
    "SyncSource",
    "SyncStep",
    "coverage_snapshot",
    "run_sync_all",
]
