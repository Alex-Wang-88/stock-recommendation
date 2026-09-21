"""预置筛选条件 —— 手写跑通的起点（方案 §七 阶段 1：「条件先用 YAML/JSON 手写跑通」）。

每个预置都是一份**普通 spec 字典**，与阶段 2 的 LLM 输出走同一条解析路径
（`Spec.from_dict`）。所以预置同时也是"LLM 应该产出什么"的样例。

⚠️ 这里的阈值是**政策选择**，不是拟合出来的最优值。
它们的作用是演示筛选器能跑，而不是宣称这组数字能赚钱。
"""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    # 「低位 + 有量」：位置只用来缩小范围，排序仍由动量/题材/资金决定。
    "low-position": {
        "note": "近一年区间下三分位 + 放量，位置仅作筛选条件",
        "close_position_250": [None, 0.35],
        "dist_to_52w_high": [None, -0.20],
        "history_days": [250, None],
        "listed_days": [365, None],
        "volume_trend_20": [1.10, None],
        "turnover_value_20": [50_000_000, None],
        "limit": 25,
    },
    # 「低位 + 均线确认」：把低位和短/中期重新站上均线结合；仍然只是观察池。
    "low-position-trend": {
        "note": "近一年低位 + 重新站上 20/60 日均线，观察趋势修复",
        "close_position_250": [None, 0.35],
        "dist_to_52w_high": [None, -0.20],
        "ma20_dev": [0, None],
        "ma60_dev": [0, None],
        "history_days": [250, None],
        "listed_days": [365, None],
        "volume_trend_20": [1.10, None],
        "turnover_value_20": [50_000_000, None],
        "limit": 25,
    },
    # 「题材启动观察」：配合 --theme 使用；没有题材命中就没有结果（这是刻意的）。
    "theme-watch": {
        "note": "命中指定题材 + 近期有量，观察启动迹象",
        "theme_days_20": [1, None],
        "volume_trend_20": [1.20, None],
        "turnover_value_20": [80_000_000, None],
        "history_days": [60, None],
        "limit": 25,
    },
    # 「次新放量」：上市不足两年、有足够交易历史、明显放量。
    "recent-ipo": {
        "note": "次新（上市 ≤ 2 年）+ 放量，注意次新波动与解禁风险",
        "listed_days": [None, 730],
        "history_days": [60, None],
        "volume_trend_20": [1.50, None],
        "turnover_value_20": [50_000_000, None],
        "limit": 25,
    },
    # 「924 以来没跟上」：用区间涨幅做筛选（不是位置分位，口径不同）。
    "behind-924": {
        "note": "924 以来跑输市场，属于补涨观察池，非买入信号",
        "ret_since_924": [None, 0.30],
        "history_days": [250, None],
        "turnover_value_20": [30_000_000, None],
        "limit": 25,
    },
}


def load_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"未知预置：{name}。可选：{sorted(PRESETS)}")
    # 深拷贝：调用方可能往里塞 themes，不要污染模块级字典。
    return {
        key: (list(value) if isinstance(value, list) else value)
        for key, value in PRESETS[name].items()
    }


__all__ = ["PRESETS", "load_preset"]
