"""T2 每日快照 —— **这是整个归档层最不能省的一步**。

T2 类数据（板块排名、板块资金流、分钟级资金流、北向分钟、涨停梯队）**只有当日快照**，
没有任何 `date` 参数可以把历史补回来。⇒ 除了每天落盘，**没有别的办法**。

所以：归档任务必须先于任何模型上线。晚一天上线，就永久少一天数据。
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import pandas as pd

from ..http import ThrottledClient
from .backfill import BackfillResult, run_single
from .sources import SECTOR_COLUMNS, SectorSnapshotSource
from .store import ArchiveStore

logger = logging.getLogger(__name__)

T2_NOT_BACKFILLABLE = (
    "T2（板块排名 / 分钟资金流 / 北向 / 涨停梯队）只有当日快照，无法回补历史。"
    "唯一来源是本地每日归档 —— 晚一天上线就永久少一天数据。"
)
MIN_SECTOR_ROWS = 20


def check_connectivity(client: ThrottledClient) -> bool:
    """用**对照端点**判断通路。

    ⚠️ 一个业务端点失败，**不能**推断成"IP 被封"：实测同一秒内
    `fflow/daykline` 成功而 `kline/get` / `clist` 被掐断。
    先跑这个对照，能避免把"端点/参数问题"误诊成"被封"而无谓地拉长冷却。
    """

    alive = client.probe()
    if not alive:
        logger.warning("对照端点也失败 ⇒ 网络或 IP 层面可能确实不可用")
    return alive


def _write_sectors(store: ArchiveStore, frame: pd.DataFrame, trade_date: str) -> int:
    if frame is None or len(frame) < MIN_SECTOR_ROWS:
        raise ValueError(f"板块快照行数不足：{len(frame) if frame is not None else 0}")
    payload = frame.copy()
    payload["trade_date"] = pd.to_datetime(trade_date).date()
    payload["fetched_at"] = datetime.now(UTC).replace(tzinfo=None)
    try:
        store.conn.execute("begin transaction")
        store.conn.execute(
            "delete from sector_snapshot where trade_date = ?",
            [date.fromisoformat(trade_date)],
        )
        written = store.upsert(
            "sector_snapshot",
            payload,
            ("trade_date", *SECTOR_COLUMNS, "fetched_at"),
        )
        if written < MIN_SECTOR_ROWS:
            raise ValueError(f"板块快照去重后行数不足：{written}")
        store.conn.execute("commit")
        return written
    except Exception:
        store.conn.execute("rollback")
        raise


def snapshot_sectors(
    store: ArchiveStore,
    client: ThrottledClient,
    trade_date: str | None = None,
    sync_run_id: str | None = None,
) -> BackfillResult:
    """落盘当日行业板块排名 + 板块主力资金流。已有记录则跳过（幂等）。"""

    target = trade_date or date.today().isoformat()
    source = SectorSnapshotSource(client)
    result = BackfillResult(task="snapshot-sectors")
    result.attempted = 1

    existing = store.conn.execute(
        "select count(*) from sector_snapshot where trade_date = ?",
        [date.fromisoformat(target)],
    ).fetchone()[0]
    if existing >= MIN_SECTOR_ROWS:
        result.skipped = 1
        logger.info("snapshot-sectors %s 已存在（%d 行），跳过", target, existing)
        return result

    run_single(
        store,
        result,
        "snapshot-sectors",
        target,
        lambda: source.fetch(),
        lambda frame: _write_sectors(store, frame, target),
        sync_run_id,
    )
    return result


def snapshot_all(
    store: ArchiveStore,
    client: ThrottledClient,
    trade_date: str | None = None,
    sync_run_id: str | None = None,
) -> list[BackfillResult]:
    """当日全部 T2 快照。已有结果先跳过，缺失时再做通路对照并落盘。"""

    target = trade_date or date.today().isoformat()
    # 先判断幂等结果，再做网络探测。否则当天早些时候已经成功保存快照，
    # 晚些时候重复运行却遇到探测端点失败时，会把“本地已有数据”误报成失败。
    existing = store.conn.execute(
        "select count(*) from sector_snapshot where trade_date = ?",
        [date.fromisoformat(target)],
    ).fetchone()[0]
    if existing >= MIN_SECTOR_ROWS:
        return [snapshot_sectors(store, client, target, sync_run_id)]

    if not check_connectivity(client):
        logger.error("通路对照失败，跳过全部 T2 快照")
        # 记录失败批次，便于调度器和覆盖率页面区分真实故障。
        store.record_run(
            "snapshot-all",
            target,
            datetime.now(UTC),
            0,
            "FAILED",
            "对照端点不可用（网络或端点暂时不可达）",
            sync_run_id,
        )
        return [
            BackfillResult(
                task="snapshot-all",
                attempted=1,
                failed=1,
                failures=[("all", "对照端点不可用（网络或端点暂时不可达）")],
            )
        ]
    return [snapshot_sectors(store, client, target, sync_run_id)]


__all__ = [
    "T2_NOT_BACKFILLABLE",
    "check_connectivity",
    "snapshot_all",
    "snapshot_sectors",
]
