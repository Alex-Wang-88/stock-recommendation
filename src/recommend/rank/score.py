"""排序层 —— **透明加权，不用 ML**（方案 §四.3）。

为什么不用 ML
------------
推荐场景的样本量只有几十~几百条「我们事后认为好」的样本，特征却有几十个。
任何拟合都会把噪声当信号记住，而**它记错了我们看不出来** —— 因为没有样本外检验的余量。
透明加权相反：权重是人写的政策参数，**每一条推荐都能逐项列出"它得了多少分、因为什么"**。

两个守卫（都在 `_validate_weights`）
---------------------------------
1. 位置因子（`POSITION_FACTORS`）**禁止**作为权重键或出现在成员列表里
   ⇒ 「低位」只能做筛选条件，不能变成排序加分。
2. 权重必须是有限非负数，且总和 > 0 ⇒ 不允许"负权重"这种看似精巧实则不可解释的写法。

缺失值处理
---------
某个类别下所有因子都缺 ⇒ 该类别不参与该行打分，权重**在剩余类别间重新归一**，
并披露 `score_coverage`（实际参与权重 ÷ 总权重）。
**不把缺失当 0** —— 那会把"没数据"当成"表现差"来惩罚，这正是量化项目踩过的坑
（占位符 `UNKNOWN` 被当行业执行上限，把仓位钉在 12%）。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..factors import POSITION_FACTORS

# 允许进排序的类别 → 成员因子（必须都是**方向性**因子）。
RANK_MEMBERS: dict[str, tuple[str, ...]] = {
    "industry": ("industry_score", "industry_leader_score"),
    "growth": (
        "yoy_net_income",
        "yoy_eps_basic",
        "revenue_yoy",
        "profit_growth_acceleration",
        "eps_growth_acceleration",
        "positive_profit_streak",
        "positive_growth_streak",
    ),
    "quality": (
        "roe_avg",
        "gross_profit_margin",
        "net_profit_margin",
        "current_ratio",
        "quick_ratio",
        "cash_ratio",
        "ebit_to_interest",
        "cfo_to_or",
        "cfo_to_np",
        "cfo_to_gr",
        "financial_safety_value",
    ),
    # 兼容旧版外部调用；默认配置已拆成 growth + quality。
    "growth_quality": (
        "yoy_net_income",
        "yoy_eps_basic",
        "roe_avg",
        "gross_profit_margin",
    ),
    "valuation": ("pe_ttm_value", "pe_industry_value"),
    # 924 以来收益替代了「近一年低位」的排序意图：低位仍只筛选，长期方向才加分。
    "momentum": (
        "ret_20",
        "ret_60",
        "ret_since_924",
        "relative_ret_20",
        "relative_ret_60",
        "volume_trend_20",
    ),
    "risk": ("volatility_value", "drawdown_value", "downside_volatility_value"),
    "theme": ("theme_heat_max", "theme_days_20"),
    # 两融是可回溯的资金确认因子；当日主力净额仍只在题材接口返回时有值。
    "capital": ("lhb_net_buy_rel", "main_net_inflow", "margin_rz_chg_20"),
    "liquidity": ("turnover_value_20",),
}

# 元数据里保留的因子 → 披露时用的中文名。
FACTOR_LABELS: dict[str, str] = {
    "industry_score": "行业景气分",
    "industry_leader_score": "行业龙头分",
    "yoy_net_income": "净利润同比",
    "yoy_eps_basic": "EPS 同比",
    "revenue_yoy": "营收同比",
    "profit_growth_acceleration": "利润增速加速度",
    "eps_growth_acceleration": "EPS 增速加速度",
    "positive_profit_streak": "连续盈利期数",
    "positive_growth_streak": "连续利润增长期数",
    "roe_avg": "ROE",
    "gross_profit_margin": "毛利率",
    "net_profit_margin": "净利率",
    "current_ratio": "流动比率",
    "quick_ratio": "速动比率",
    "cash_ratio": "现金比率",
    "ebit_to_interest": "EBIT 利息保障倍数",
    "cfo_to_or": "经营现金流/营收",
    "cfo_to_np": "经营现金流/净利润",
    "cfo_to_gr": "经营现金流/营业收入",
    "financial_safety_value": "财务安全分",
    "pe_ttm_value": "PE(TTM)相对估值",
    "pe_industry_value": "行业内PE分位",
    "ret_20": "20 日收益",
    "ret_60": "60 日收益",
    "ret_since_924": "924 以来收益",
    "relative_ret_20": "20 日相对大盘收益",
    "relative_ret_60": "60 日相对大盘收益",
    "volume_trend_20": "量比",
    "volatility_value": "低波动分",
    "drawdown_value": "回撤分",
    "downside_volatility_value": "下行波动分",
    "theme_heat_max": "题材热度",
    "theme_days_20": "题材上榜天数",
    "lhb_net_buy_rel": "龙虎榜净买强度",
    "main_net_inflow": "当日主力净额",
    "margin_rz_chg_20": "20 日融资余额变化",
    "turnover_value_20": "成交额",
}


class RankWeightError(ValueError):
    """权重表不合法（含位置因子 / 未知键 / 负值 / 全零）。"""


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    forbidden = sorted(set(weights) & set(POSITION_FACTORS))
    if forbidden:
        raise RankWeightError(
            f"位置因子禁止进入排序权重：{forbidden}。"
            "「低位」只做筛选条件 —— 低位可能是价值，也可能是价值陷阱。"
        )
    unknown = sorted(set(weights) - set(RANK_MEMBERS))
    if unknown:
        raise RankWeightError(
            f"未知权重键：{unknown}。可选类别：{sorted(RANK_MEMBERS)}"
        )
    cleaned: dict[str, float] = {}
    for key, value in weights.items():
        number = float(value)
        if not np.isfinite(number) or number < 0:
            raise RankWeightError(f"权重必须是有限非负数，{key} = {value!r}")
        cleaned[key] = number
    if sum(cleaned.values()) <= 0:
        raise RankWeightError("权重总和必须大于 0")
    return cleaned


def _pct_rank(series: pd.Series) -> pd.Series:
    """0–1 分位秩。NaN 保持 NaN（不参与排名，也不会被当成最低分）。"""

    if series.isna().all():
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return series.rank(pct=True)


def add_scores(frame: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    """加 `score` + 每个类别的子分 + `score_coverage`。

    每个成员因子先转成横截面分位（**秩**，不是 z 分数）——
    秩对极端值免疫，不会被一只翻倍股带偏整列权重。
    """

    if frame is None or frame.empty:
        out = frame.copy() if frame is not None else pd.DataFrame()
        out["score"] = pd.Series(dtype="float64")
        out["score_coverage"] = pd.Series(dtype="float64")
        return out

    cleaned = _validate_weights(weights)
    out = frame.copy()

    weighted_sum = pd.Series(0.0, index=out.index)
    weight_seen = pd.Series(0.0, index=out.index)

    for category, weight in cleaned.items():
        members = [name for name in RANK_MEMBERS[category] if name in out.columns]
        if not members:
            out[f"score_{category}"] = np.nan
            continue
        parts = pd.concat([_pct_rank(out[name]) for name in members], axis=1)
        sub = parts.mean(axis=1, skipna=True)
        out[f"score_{category}"] = sub
        available = sub.notna()
        weighted_sum = weighted_sum.add(sub.fillna(0.0) * weight, fill_value=0.0)
        weight_seen = weight_seen.add(available.astype(float) * weight, fill_value=0.0)

    total_weight = sum(cleaned.values())
    out["score"] = weighted_sum / weight_seen.replace(0.0, np.nan)
    # 实际参与权重 ÷ 总权重：< 1 说明有类别整列缺失，分数可信度下降。
    out["score_coverage"] = (weight_seen / total_weight).replace(0.0, np.nan)
    return out


def explain(row: Mapping[str, object], weights: Mapping[str, float]) -> list[str]:
    """把一个候选的分数拆成人类可读的贡献明细 —— 每条推荐必须带「为什么」。

    ⚠️ 「参与因子」必须**逐行**判断哪些成员真的有值。
    如果只把 `RANK_MEMBERS` 全列出来，读者会以为龙虎榜、主力净额都参与了打分，
    而实际上它们可能整列缺失 —— 那就等于给出了一份假的理由。
    """

    cleaned = _validate_weights(weights)
    total_weight = sum(cleaned.values())
    lines: list[str] = []
    contributing: list[str] = []
    absent: list[str] = []

    for category in sorted(cleaned, key=lambda key: -cleaned[key]):
        sub = row.get(f"score_{category}")
        if sub is None or _na(sub):
            lines.append(f"  - {category}: 数据缺失，未参与打分")
        else:
            contribution = float(sub) * cleaned[category] / total_weight
            lines.append(
                f"  - {category}: 分位 {float(sub):.3f}"
                f" × 权重 {cleaned[category]:g} ≈ {contribution:.3f}"
            )
        for name in RANK_MEMBERS[category]:
            label = FACTOR_LABELS.get(name, name)
            value = row.get(name)
            if value is None or _na(value):
                absent.append(label)
            else:
                contributing.append(label)

    if contributing:
        lines.append(f"  实际参与打分的因子：{'、'.join(contributing)}")
    if absent:
        lines.append(f"  数据缺失、未参与的因子：{'、'.join(absent)}")
    lines.append("  位置类因子（区间分位 / 距高点）**不参与排序**，只用于缩小范围。")
    return lines


def _na(value: object) -> bool:
    try:
        return bool(np.isnan(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
