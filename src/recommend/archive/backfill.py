"""T1 历史回补。

三条铁律
--------
1. **幂等**：可随时重跑、可断点续跑。默认跳过已归档的日期（`--force` 覆盖）。
2. **区分 EMPTY 与 FAILED**：非交易日返回空是**正常**的，接口失效返回空是**故障**。
   混在一起，归档覆盖率就没法信了。
3. **逐日独立记账**：一天失败不中断整批，最后统一汇报失败清单。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from ..http import ThrottledClient
from .sources import (
    MARGIN_COLUMNS,
    DragonTigerWithFallbackSource,
    FallbackResponseError,
    MarginSource,
    OfficialMarginSource,
    ThemeSource,
    candidate_dates,
)
from .store import ArchiveStore

logger = logging.getLogger(__name__)


@dataclass
class BackfillResult:
    task: str
    attempted: int = 0
    ok: int = 0
    empty: int = 0
    rejected: int = 0
    failed: int = 0
    rows: int = 0
    skipped: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"{self.task}：尝试 {self.attempted}",
            f"成功 {self.ok}",
            f"空 {self.empty}",
            f"回退拒绝 {self.rejected}",
            f"失败 {self.failed}",
            f"跳过 {self.skipped}",
            f"写入 {self.rows} 行",
        ]
        if self.failures:
            parts.append(f"失败样例 {self.failures[:3]}")
        return " | ".join(parts)


def run_single(
    store: ArchiveStore,
    result: BackfillResult,
    task: str,
    target: str,
    fetch: object,
    write: object,
    sync_run_id: str | None = None,
) -> None:
    """执行单次抓取 + 写入，并把结果记进 `archive_runs`。

    公共入口：回补（按日）与快照（按次）都走这里，保证**失败/空/成功的记账口径一致**。
    """

    started = datetime.now(UTC)
    try:
        frame = fetch()  # type: ignore[operator]
    except FallbackResponseError as exc:
        # 上游回退 ≠ 故障，也 ≠ 真的没有强势股。它是「这一天没有数据」的**证据**，
        # 只是上游用"返回别人的快照"而不是"返回空"来表达。单独记一类 `REJECTED`，
        # 否则休市日会被当成正常交易日写进归档（实测污染过 38 天）。
        result.rejected += 1
        store.record_run(task, target, started, 0, "REJECTED", str(exc), sync_run_id)
        logger.debug("%s %s 判定为无数据（上游回退）：%s", task, target, exc)
        return
    except Exception as exc:  # noqa: BLE001 - 单日失败不能中断整批回补
        result.failed += 1
        result.failures.append((target, f"{type(exc).__name__}: {exc}"))
        store.record_run(task, target, started, 0, "FAILED", str(exc), sync_run_id)
        logger.warning("%s %s 失败：%s", task, target, exc)
        return

    if frame is None or frame.empty:
        result.empty += 1
        store.record_run(
            task,
            target,
            started,
            0,
            "EMPTY",
            "接口返回空（非交易日或数据未更新）",
            sync_run_id,
        )
        return

    try:
        written = write(frame)  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - 写入失败必须进入归档账本
        result.failed += 1
        result.failures.append((target, f"写入 {type(exc).__name__}: {exc}"))
        store.record_run(task, target, started, 0, "FAILED", str(exc), sync_run_id)
        logger.warning("%s %s 写入失败：%s", task, target, exc)
        return
    result.ok += 1
    result.rows += int(written)
    store.record_run(task, target, started, int(written), "OK", "", sync_run_id)


def backfill_themes(
    store: ArchiveStore,
    client: ThrottledClient,
    start: str,
    end: str,
    force: bool = False,
    sync_run_id: str | None = None,
) -> BackfillResult:
    """回补题材归因（同花顺，按日期路径参数，实测可回溯到 2024-09 及更早）。"""

    source = ThemeSource(client)
    result = BackfillResult(task="backfill-themes")
    days = candidate_dates(start, end)
    pending = set(
        store.retryable_dates(
            "theme_attribution", days, task="backfill-themes", force=force
        )
    )

    # 长回补必须**打印进度**：几百个交易日 × ~1.5s 是十几分钟的黑箱，
    # 没有进度输出就没法判断"在正常跑"还是"卡住了"。
    for index, trade_date in enumerate(days, start=1):
        if trade_date not in pending:
            result.skipped += 1
        else:
            result.attempted += 1
            run_single(
                store,
                result,
                "backfill-themes",
                trade_date,
                lambda day=trade_date: source.fetch(day),
                lambda frame, day=trade_date: store.write_themes(frame, day),
                sync_run_id,
            )
        if index % 20 == 0 or index == len(days):
            logger.info(
                "[%d/%d] %s 累计：成功 %d 空 %d 回退拒绝 %d 失败 %d 写入 %d 行",
                index,
                len(days),
                trade_date,
                result.ok,
                result.empty,
                result.rejected,
                result.failed,
                result.rows,
            )
    return result


def backfill_dragon_tiger(
    store: ArchiveStore,
    client: ThrottledClient,
    start: str,
    end: str,
    force: bool = False,
    sync_run_id: str | None = None,
) -> BackfillResult:
    """回补全市场龙虎榜（东财主源，新浪仅在主源失败时提供事件备用）。"""

    source = DragonTigerWithFallbackSource(client)
    result = BackfillResult(task="backfill-dragon-tiger")
    columns = (
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
    )
    days = candidate_dates(start, end)
    pending = set(
        store.retryable_dates(
            "dragon_tiger", days, task="backfill-dragon-tiger", force=force
        )
    )

    for index, trade_date in enumerate(days, start=1):
        if trade_date not in pending:
            result.skipped += 1
        else:
            result.attempted += 1
            run_single(
                store,
                result,
                "backfill-dragon-tiger",
                trade_date,
                lambda day=trade_date: source.fetch(day),
                lambda frame: store.upsert("dragon_tiger", frame, columns),
                sync_run_id,
            )
        if index % 20 == 0 or index == len(days):
            logger.info(
                "[%d/%d] %s 累计：成功 %d 空 %d 回退拒绝 %d 失败 %d 写入 %d 行",
                index,
                len(days),
                trade_date,
                result.ok,
                result.empty,
                result.rejected,
                result.failed,
                result.rows,
            )
    return result


def backfill_margin(
    store: ArchiveStore,
    client: ThrottledClient,
    symbols: list[str],
    page_size: int = 60,
    sync_run_id: str | None = None,
) -> BackfillResult:
    """回补两融明细（**按标的逐只**，所以只对关注的标的跑）。"""

    source = MarginSource(client)
    result = BackfillResult(task="backfill-margin")
    columns = (
        "trade_date",
        "symbol",
        "rzye",
        "rzmre",
        "rzche",
        "rqye",
        "rqmcl",
        "rqchl",
        "rzrqye",
    )

    for symbol in symbols:
        result.attempted += 1
        run_single(
            store,
            result,
            "backfill-margin",
            symbol,
            lambda code=symbol: source.fetch(code, page_size=page_size),
            lambda frame: store.upsert("margin_trading", frame, columns),
            sync_run_id,
        )
    return result


def backfill_margin_exchange(
    store: ArchiveStore,
    client: ThrottledClient,
    start: str,
    end: str,
    force: bool = False,
    sync_run_id: str | None = None,
) -> BackfillResult:
    """按交易日回补上交所 + 深交所官方两融明细。

    一次请求覆盖全市场，适合做完整归档；`backfill_margin` 保留为旧的按标的东财
    兼容入口，避免已有脚本突然失效。官方接口返回空时只记 EMPTY，不清除已有数据。
    """

    source = OfficialMarginSource(client)
    result = BackfillResult(task="backfill-margin-exchange")
    days = candidate_dates(start, end)
    pending = set(
        store.retryable_dates(
            "margin_trading", days, task="backfill-margin-exchange", force=force
        )
    )

    for index, trade_date in enumerate(days, start=1):
        if trade_date not in pending:
            result.skipped += 1
        else:
            result.attempted += 1
            run_single(
                store,
                result,
                "backfill-margin-exchange",
                trade_date,
                lambda day=trade_date: source.fetch(day),
                lambda frame: store.upsert("margin_trading", frame, MARGIN_COLUMNS),
                sync_run_id,
            )
        if index % 20 == 0 or index == len(days):
            logger.info(
                "[%d/%d] %s 累计：成功 %d 空 %d 回退拒绝 %d 失败 %d 写入 %d 行",
                index,
                len(days),
                trade_date,
                result.ok,
                result.empty,
                result.rejected,
                result.failed,
                result.rows,
            )
    return result


def symbols_from_themes(store: ArchiveStore, limit: int | None = None) -> list[str]:
    """从已归档的题材数据里取标的（避免手工维护股票池）。"""

    query = "select distinct symbol from theme_attribution order by symbol"
    if limit:
        query += f" limit {int(limit)}"
    return [row[0] for row in store.conn.execute(query).fetchall()]


def frame_preview(frame: pd.DataFrame, rows: int = 3) -> str:
    """给 CLI 用的一小段可读预览。"""

    if frame is None or frame.empty:
        return "（空）"
    return frame.head(rows).to_string(index=False)
