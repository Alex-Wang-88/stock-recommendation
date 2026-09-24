"""因子层 —— 全确定性、可解释，不含 ML、不含 LLM。

分工
----
* `price.py`    长期收益（924以来 / 年内）与位置量（区间分位、距高点）；
  `ret_since_924` 可排序，位置量仍**只做筛选**
* `momentum.py` 方向量（区间收益、量比、流动性）— 可排序
* `theme.py`    题材（标签、热度、上榜天数）— 可排序（带选择偏差警告）
* `capital.py`  资金（龙虎榜、两融、主力净额）— 可排序
* `risk.py`     风险/确认（波动率、回撤、下行波动、成交额方向压力）— 可排序

`factors/__init__.py` 也提供 `FactorTable`：把上述四块拼成一张宽表，
并把每列的**类别**登记下来，供排序层判定"能否进权重"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .capital import CAPITAL_FACTORS, add_capital_factors
from .fundamental import FUNDAMENTAL_FACTORS, add_fundamental_factors
from .momentum import MOMENTUM_FACTORS, add_momentum_factors
from .price import POSITION_FACTORS, PRICE_FACTORS, add_price_factors
from .risk import RISK_FACTORS, add_risk_factors
from .theme import THEME_FACTORS, add_theme_factors, matches_themes

# 因子 → 所属类别。排序层靠这张表判断某个因子能不能进权重。
FACTOR_CATEGORY: dict[str, str] = {}
for _name in PRICE_FACTORS:
    FACTOR_CATEGORY[_name] = "position" if _name in POSITION_FACTORS else "price"
for _name in MOMENTUM_FACTORS:
    FACTOR_CATEGORY[_name] = "momentum"
for _name in THEME_FACTORS:
    FACTOR_CATEGORY[_name] = "theme"
for _name in CAPITAL_FACTORS:
    FACTOR_CATEGORY[_name] = "capital"
for _name in FUNDAMENTAL_FACTORS:
    if _name in {"symbol", "pe_ttm", "pb_mrq", "ps_ttm", "eps_ttm", "market_cap"}:
        FACTOR_CATEGORY[_name] = "fundamental"
    elif _name in {"pe_ttm_value", "pe_industry_value"}:
        FACTOR_CATEGORY[_name] = "valuation"
    elif _name in {"industry_score", "industry_leader_score"}:
        FACTOR_CATEGORY[_name] = "industry"
    elif _name in {
        "roe_avg",
        "gross_profit_margin",
        "net_profit_margin",
        "cfo_to_np",
        "cfo_to_or",
        "cfo_to_gr",
        "liability_to_asset",
        "financial_safety_value",
        "current_ratio",
        "quick_ratio",
        "cash_ratio",
        "ebit_to_interest",
    }:
        FACTOR_CATEGORY[_name] = "quality"
    elif _name in {
        "yoy_net_income",
        "yoy_eps_basic",
        "yoy_profit_net_income",
        "revenue_yoy",
        "profit_growth_acceleration",
        "eps_growth_acceleration",
        "positive_profit_streak",
        "positive_growth_streak",
    }:
        FACTOR_CATEGORY[_name] = "growth"
    else:
        FACTOR_CATEGORY[_name] = "growth_quality"  # 兼容旧版自定义因子
for _name in RISK_FACTORS:
    FACTOR_CATEGORY[_name] = "risk"

# 元数据列：不是因子，只是标识/描述。
ID_COLUMNS = (
    "symbol",
    "name",
    "industry",
    "trade_date",
    "listed_date",
    "listed_days",
    "theme_tags",
    "reason",
    "announcement_alert",
)


@dataclass
class FactorTable:
    """一张 symbol × 因子的宽表，外加来源元信息。"""

    frame: pd.DataFrame
    as_of: str
    notes: list[str] = field(default_factory=list)

    def categories(self) -> dict[str, str]:
        return {
            column: FACTOR_CATEGORY.get(column, "meta")
            for column in self.frame.columns
            if column not in ID_COLUMNS
        }


def build_factor_table(
    *,
    universe: pd.DataFrame,
    bars: pd.DataFrame,
    base_closes: pd.DataFrame,
    attribution: pd.DataFrame,
    tag_daily: pd.DataFrame,
    theme_history: pd.DataFrame | None = None,
    dragon_tiger: pd.DataFrame | None = None,
    margin: pd.DataFrame | None = None,
    valuation: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
    financial_history: pd.DataFrame | None = None,
    as_of: str,
) -> FactorTable:
    """拼装完整因子宽表。任一块数据缺失都只影响对应列（NaN），不阻断整体。"""

    notes: list[str] = []
    price = add_price_factors(bars, base_closes)
    momentum = add_momentum_factors(bars)
    risk = add_risk_factors(bars)
    if momentum["volume_trend_20"].isna().all():
        notes.append("量比全为 NaN：行情窗口不足以构成 20 日均量，请检查 sync-bars 是否跑过。")

    merged = (
        universe.merge(price, on="symbol", how="left")
        .merge(momentum, on="symbol", how="left")
        .merge(risk, on="symbol", how="left")
    )
    merged = add_theme_factors(merged, attribution, tag_daily, theme_history)
    merged = add_capital_factors(merged, attribution, dragon_tiger, margin)
    merged = add_fundamental_factors(merged, valuation, financial, financial_history)
    merged["trade_date"] = as_of
    return FactorTable(frame=merged, as_of=as_of, notes=notes)


__all__ = [
    "CAPITAL_FACTORS",
    "FUNDAMENTAL_FACTORS",
    "FACTOR_CATEGORY",
    "FactorTable",
    "ID_COLUMNS",
    "MOMENTUM_FACTORS",
    "POSITION_FACTORS",
    "PRICE_FACTORS",
    "RISK_FACTORS",
    "THEME_FACTORS",
    "add_capital_factors",
    "add_fundamental_factors",
    "add_momentum_factors",
    "add_price_factors",
    "add_risk_factors",
    "add_theme_factors",
    "build_factor_table",
    "matches_themes",
]
