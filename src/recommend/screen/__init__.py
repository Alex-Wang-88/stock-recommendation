"""筛选层 —— 确定性规则 → 候选表。

* `spec.py`   —— 条件的数据结构（白名单枚举，可序列化）
* `engine.py` —— 执行过滤，返回候选表 + 剔除日志

**本层不含 LLM。** 自然语言 → spec 的翻译属于阶段 2（`translate.py`），
而且它只产出 `spec`，不产出候选、不产出排名。
"""

from .engine import ScreenResult, run_screen
from .spec import ALLOWED_SORTS, FILTER_FIELDS, Spec, SpecError

__all__ = [
    "ALLOWED_SORTS",
    "FILTER_FIELDS",
    "ScreenResult",
    "Spec",
    "SpecError",
    "run_screen",
]
