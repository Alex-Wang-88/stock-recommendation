"""筛选条件（spec）—— **白名单枚举**，可序列化、可审计。

为什么必须是白名单
----------------
阶段 2 的 LLM 翻译层会把自然语言变成这里的结构（方案 §四.1）。LLM 会**编**字段名，
所以解析器遇到未知字段必须**报错而不是忽略** —— 报错会让幻觉立刻暴露，
忽略则会让幻觉悄悄变成一次"条件更少因而结果更多"的筛选。
这是本系统里唯一一处"宁可不干活也不能猜"的地方。

两条机械保证
----------
1. `FILTER_FIELDS` 里没有的位置因子**不能**出现在 `sort`（`ALLOWED_SORTS` 排除它们）
   ⇒ 「低位」只能缩小范围，不能决定名次。
2. `rank/score.py` 另外独立检查权重表，双重保险。

支持的写法
---------
    {"themes": ["算力"], "ret_ytd": [-0.2, null], "volume_trend_20": ">1.5"}
    {"filters": {"close_position_250": {"max": 0.3}}, "limit": 20}

区间统一表达为 `(min, max)`，任一端为 `None` 表示不设限。
标量 `1.5` 等价于 `(1.5, None)`；字符串 `">=1.5"` 等价于 `(1.5, None)`。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..factors import POSITION_FACTORS

RANGE = "range"
TAGS = "tags"
BOOL = "bool"


class SpecError(ValueError):
    """spec 不合法（未知字段 / 类型不对 / 排序字段无资格）。"""


# -- 白名单：筛选因子 ------------------------------------------------------- #

FILTER_FIELDS: dict[str, str] = {
    # 价格 / 位置（只做筛选，不做排序）
    "ret_since_924": "自 2024-09-24 起的涨幅，1.0 = +100%",
    "ret_ytd": "年内涨幅（基准为上年最后一个交易日收盘）",
    "close_position_20": "在近 20 日收盘区间中的位置，0 = 区间下沿，1 = 上沿",
    "close_position_60": "在近 60 日收盘区间中的位置",
    "close_position_250": "在近 250 日收盘区间中的位置（「低位」的主力判据）",
    "dist_to_52w_high": "距近 250 日最高收盘的回撤，-0.3 = 距高点 30%",
    "ma20_dev": "相对 20 日均线的偏离",
    "ma60_dev": "相对 60 日均线的偏离",
    "realized_vol_20": "近 20 日实现波动率（每日收益标准差）",
    "max_drawdown_60": "近 60 日最大回撤",
    "signed_amount_pressure_20": "近 20 日成交额方向压力，-1 到 1",
    # 基本面 / 估值
    "pe_ttm": "滚动市盈率 PE(TTM)，亏损或缺失时为空",
    "pb_mrq": "市净率 PB(MRQ)",
    "ps_ttm": "滚动市销率 PS(TTM)",
    "eps_ttm": "滚动每股收益 EPS(TTM)",
    "roe_avg": "平均净资产收益率 ROE",
    "gross_profit_margin": "销售毛利率",
    "net_profit_margin": "销售净利率",
    "yoy_net_income": "净利润同比",
    "yoy_eps_basic": "基本每股收益同比",
    "revenue_yoy": "营收同比（基于跨年同期财报）",
    "profit_growth_acceleration": "净利润同比增速的环比加速度",
    "eps_growth_acceleration": "EPS 同比增速的环比加速度",
    "positive_profit_streak": "连续盈利财报期数",
    "positive_growth_streak": "连续净利润同比为正的财报期数",
    "current_ratio": "流动比率",
    "quick_ratio": "速动比率",
    "cash_ratio": "现金比率",
    "liability_to_asset": "资产负债率",
    "asset_to_equity": "资产权益比",
    "ebit_to_interest": "EBIT 利息保障倍数",
    "cfo_to_or": "经营现金流/营收",
    "cfo_to_np": "经营现金流/净利润",
    "cfo_to_gr": "经营现金流/营业收入（增长口径）",
    "pe_industry_value": "相对行业中位 PE 的估值分",
    "relative_ret_20": "20 日相对等权市场收益",
    "relative_ret_60": "60 日相对等权市场收益",
    "volatility_value": "低波动方向分",
    "drawdown_value": "回撤方向分",
    "downside_volatility_value": "低下行波动方向分",
    "industry_score": "行业景气分（市场强度 + 行业盈利增长）",
    "industry_leader_score": "行业内市值龙头分位",
    # 动量 / 量价（可排序）
    "ret_20": "近 20 个交易日收益",
    "ret_60": "近 60 个交易日收益",
    "ret_rank_20": "近 20 日收益在全体标的中的分位，0–1",
    "volume_trend_20": "量比：近 5 日均量 ÷ 近 20 日均量，>1 = 放量",
    "turnover_value_20": "近 20 日日均成交额（元），流动性门槛",
    "amount_avg_20": "近 20 日日均成交额（元），与上者同源，保留别名",
    # 题材（可排序）
    "theme_heat_max": "所属题材中当日最热那个的上榜股票数",
    "theme_heat_sum": "所属题材上榜股票数之和",
    "theme_days_20": "近 20 个交易日出现在题材归因名单里的天数",
    # 资金（可排序）
    "lhb_count_20": "近 20 个交易日龙虎榜上榜次数",
    "lhb_net_buy_20": "近 20 个交易日龙虎榜净买合计（元）",
    "lhb_net_buy_rel": "龙虎榜净买 ÷ 20 日成交额",
    "main_net_inflow": "当日主力净额（元）",
    "margin_rz_chg_20": "融资余额 20 日变化率",
    # 元数据（只做筛选）
    "listed_days": "上市天数（自然日，按 as-of 计）",
    "history_days": "行情窗口内可用的交易日根数",
    "market_cap": "总市值（元），若归档提供",
}

CONTROL_FIELDS: dict[str, str] = {
    "as_of": "as-of 交易日（YYYY-MM-DD）。缺省取最近一个已收盘交易日",
    "themes": "命中**任一**题材标签（整标签精确匹配）",
    "exclude_themes": "命中任一即剔除",
    "exclude_st": "剔除名称含 ST / 退 的标的（默认 true）",
    "filters": "嵌套形式的条件字典，等价于把条件写在顶层",
    "limit": "返回条数上限",
    "sort": "排序字段（**位置因子无资格**）",
    "descending": "是否降序",
    "note": "人类可读的备注，只用于回显，不参与计算",
}

# 排序白名单：位置因子**不在此列** —— 「低位」只做筛选，不做排序加分。
ALLOWED_SORTS: tuple[str, ...] = (
    "score",
    "ret_20",
    "ret_60",
    "ret_rank_20",
    "volume_trend_20",
    "turnover_value_20",
    "amount_avg_20",
    "theme_heat_max",
    "theme_heat_sum",
    "theme_days_20",
    "lhb_count_20",
    "lhb_net_buy_20",
    "lhb_net_buy_rel",
    "main_net_inflow",
    "margin_rz_chg_20",
    "ret_since_924",
    "ret_ytd",
)

_OPERATOR = re.compile(r"^\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class Spec:
    as_of: str | None = None
    themes: tuple[str, ...] = ()
    exclude_themes: tuple[str, ...] = ()
    exclude_st: bool = True
    filters: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    limit: int = 30
    sort: str = "score"
    descending: bool = True
    note: str = ""

    # -- 构造 ------------------------------------------------------------- #

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Spec:
        if payload is None:
            raise SpecError("spec 不能为空")
        if not isinstance(payload, Mapping):
            raise SpecError(f"spec 必须是字典，收到 {type(payload).__name__}")

        known = set(FILTER_FIELDS) | set(CONTROL_FIELDS)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise SpecError(
                f"未知字段：{unknown}。"
                f"可用筛选字段：{sorted(FILTER_FIELDS)}；"
                f"控制字段：{sorted(CONTROL_FIELDS)}"
            )

        raw_filters: dict[str, Any] = {}
        nested = payload.get("filters")
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise SpecError("filters 必须是字典")
            bad = sorted(set(nested) - set(FILTER_FIELDS))
            if bad:
                raise SpecError(f"filters 里的未知字段：{bad}")
            raw_filters.update(nested)
        for key, value in payload.items():
            if key in FILTER_FIELDS:
                raw_filters[key] = value

        filters = {key: _parse_range(key, value) for key, value in raw_filters.items()}

        sort = str(payload.get("sort", "score"))
        if sort not in ALLOWED_SORTS:
            hint = ""
            if sort in POSITION_FACTORS:
                hint = (
                    " —— 位置因子只做筛选条件，不做排序加分（低位可能是价值，也可能是价值陷阱）"
                )
            raise SpecError(f"sort 不支持 {sort!r}{hint}。可选：{list(ALLOWED_SORTS)}")

        return cls(
            as_of=payload.get("as_of"),
            themes=tuple(_as_tags(payload.get("themes"), "themes")),
            exclude_themes=tuple(_as_tags(payload.get("exclude_themes"), "exclude_themes")),
            exclude_st=bool(payload.get("exclude_st", True)),
            filters=filters,
            limit=int(payload.get("limit", 30)),
            sort=sort,
            descending=bool(payload.get("descending", True)),
            note=str(payload.get("note", "")),
        )

    # -- 输出 ------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """可序列化回写（阶段 2 的 LLM 层要能把 spec 原样存证）。"""

        payload: dict[str, Any] = {
            "themes": list(self.themes),
            "exclude_themes": list(self.exclude_themes),
            "exclude_st": self.exclude_st,
            "limit": self.limit,
            "sort": self.sort,
            "descending": self.descending,
        }
        if self.as_of:
            payload["as_of"] = self.as_of
        if self.note:
            payload["note"] = self.note
        for key, (low, high) in self.filters.items():
            payload[key] = [low, high]
        return payload

    def describe(self) -> list[str]:
        """人类可读的条件清单 —— 「为什么推荐它」的第一段。"""

        lines: list[str] = []
        if self.as_of:
            lines.append(f"数据截至 {self.as_of}")
        if self.themes:
            lines.append(f"题材命中任一：{'、'.join(self.themes)}")
        if self.exclude_themes:
            lines.append(f"剔除题材：{'、'.join(self.exclude_themes)}")
        if self.exclude_st:
            lines.append("剔除 ST / 退市风险名称")
        for key, (low, high) in sorted(self.filters.items()):
            lines.append(f"{key} ∈ {_render_range(low, high)}")
        lines.append(
            f"按 {self.sort} {'降' if self.descending else '升'}序取前 {self.limit} 条"
        )
        return lines


def _render_range(low: float | None, high: float | None) -> str:
    if low is not None and high is not None:
        return f"[{_num(low)}, {_num(high)}]"
    if low is not None:
        return f"≥ {_num(low)}"
    if high is not None:
        return f"≤ {_num(high)}"
    return "（无限制）"


def _num(value: float) -> str:
    return f"{value:g}"


def _as_tags(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise SpecError(f"{field_name} 必须是字符串或字符串列表")


def _parse_range(field_name: str, value: Any) -> tuple[float | None, float | None]:
    """把标量 / 二元列表 / 比较字符串 / 字典统一成 `(min, max)`。"""

    if value is None:
        return (None, None)
    if isinstance(value, bool):
        raise SpecError(f"{field_name} 需要数值区间，收到布尔值")

    if isinstance(value, (int, float)):
        return (_check(field_name, float(value)), None)

    if isinstance(value, str):
        match = _OPERATOR.match(value)
        if not match:
            raise SpecError(
                f"{field_name} 的字符串形式必须形如 '>1.5' / '<=0.3'，收到 {value!r}"
            )
        operator, number = match.groups()
        bound = _check(field_name, float(number))
        if operator in (">", ">="):
            return (bound, None)
        return (None, bound)

    if isinstance(value, Mapping):
        bad = sorted(set(value) - {"min", "max"})
        if bad:
            raise SpecError(f"{field_name} 的区间字典只能有 min / max，多了 {bad}")
        return (
            _opt(field_name, value.get("min")),
            _opt(field_name, value.get("max")),
        )

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise SpecError(f"{field_name} 的区间必须是 [min, max] 两个元素，收到 {value!r}")
        low, high = _opt(field_name, value[0]), _opt(field_name, value[1])
        if low is not None and high is not None and low > high:
            raise SpecError(f"{field_name} 区间上下界颠倒：{value!r}")
        return (low, high)

    raise SpecError(f"{field_name} 的值类型不支持：{type(value).__name__}")


def _opt(field_name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SpecError(f"{field_name} 需要数值或 null，收到布尔值")
    if isinstance(value, (int, float)):
        return _check(field_name, float(value))
    raise SpecError(f"{field_name} 的区间端点必须是数值或 null，收到 {value!r}")


def _check(field_name: str, value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        raise SpecError(f"{field_name} 不能是 NaN / inf")
    return value
