"""归档层：T1 历史回补 + T2 每日快照。

这是整个推荐系统的**地基**，必须先于任何因子/模型上线 ——
因为 T2 类数据只有当日快照，晚一天上线就永久少一天。
"""

from .backfill import (
    BackfillResult,
    backfill_dragon_tiger,
    backfill_margin,
    backfill_margin_exchange,
    backfill_themes,
    symbols_from_themes,
)
from .snapshot import T2_NOT_BACKFILLABLE, check_connectivity, snapshot_all, snapshot_sectors
from .store import ArchiveStore, split_tags

__all__ = [
    "T2_NOT_BACKFILLABLE",
    "ArchiveStore",
    "BackfillResult",
    "backfill_dragon_tiger",
    "backfill_margin",
    "backfill_margin_exchange",
    "backfill_themes",
    "check_connectivity",
    "snapshot_all",
    "snapshot_sectors",
    "split_tags",
    "symbols_from_themes",
]
