"""排序层 —— 透明加权，不用 ML。

`score.py` 提供 `add_scores` / `explain`，并机械地禁止位置因子进入权重。
"""

from .score import (
    FACTOR_LABELS,
    RANK_MEMBERS,
    RankWeightError,
    add_scores,
    explain,
)

__all__ = [
    "FACTOR_LABELS",
    "RANK_MEMBERS",
    "RankWeightError",
    "add_scores",
    "explain",
]
