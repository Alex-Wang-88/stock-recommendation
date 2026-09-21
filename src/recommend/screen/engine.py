"""筛选引擎 —— 把因子宽表按 spec 过滤，产出候选表 + 因子明细。

三个设计决定，都是为了防止「筛选器悄悄放水」
------------------------------------------
1. **NaN 一律判不通过**，并单独计数（`<字段>:missing`）。
   如果某个因子整列缺失（归档没回补），结果会是空表而**不是**"条件被忽略、结果变多"。
   这是最危险的失败模式 —— 用户以为筛了，其实没筛。
2. **每次筛选都返回完整的剔除日志**（`rejections`），逐条件显示还剩多少只。
   数字对不上时，能立刻定位是哪一步把标的吃掉了。
3. **ST 剔除在名称缺失时不生效，但会记 note** —— 宁可显式告知，不要假装筛过。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..factors import FactorTable, matches_themes
from ..operation import add_operation_plan
from ..rank.score import RankWeightError, add_scores
from .spec import Spec

# 名称里出现这些片段即视为风险标的。
# * `ST` 覆盖 `ST` 与 `*ST`（风险警示）
# * `退` 覆盖 `退市XX` / `XX退`（退市整理期）
RISK_NAME_MARKERS = ("ST", "退")


@dataclass
class ScreenResult:
    frame: pd.DataFrame
    spec: Spec
    as_of: str
    universe_size: int
    pool_size: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def kept(self) -> int:
        return len(self.frame)

    @property
    def truncated(self) -> int:
        """通过全部条件、但被 `limit` 截掉的条数。

        必须披露：否则「命中 25 只」会被读成"恰好 25 只满足条件"，
        而实际可能是"347 只满足，只给你看 25 只"。
        """

        return max(0, self.pool_size - self.kept)

    def rejection_lines(self) -> list[str]:
        return [f"  - {reason}: {count} 只" for reason, count in self.rejections.items() if count]

    def summary(self) -> str:
        lines = [
            f"as-of {self.as_of}",
            f"全市场候选 {self.universe_size} 只 → 通过条件 {self.pool_size} 只"
            f" → 取前 {self.kept} 只",
        ]
        lines.extend(self.rejection_lines())
        if self.truncated:
            lines.append(f"  - 被 limit 截断: {self.truncated} 只（未显示，不是未通过）")
        lines.extend(f"  ! {note}" for note in self.notes)
        return "\n".join(lines)


def run_screen(table: FactorTable, spec: Spec, weights: dict[str, float]) -> ScreenResult:
    """按 `spec` 过滤 `table`，再按 `spec.sort` 排序取前 `spec.limit`。"""

    frame = table.frame.copy()
    universe_size = len(frame)
    rejections: dict[str, int] = {}
    notes: list[str] = list(table.notes)

    # -- 1. 风险名称 -------------------------------------------------------- #
    if spec.exclude_st:
        if "name" in frame.columns and frame["name"].notna().any():
            names = frame["name"].fillna("").astype(str)
            # 用**字面量**匹配，不走正则 —— 名称里的特殊字符不应被当成模式。
            risky = pd.Series(False, index=frame.index)
            for marker in RISK_NAME_MARKERS:
                risky |= names.str.contains(marker, regex=False)
            frame, _ = _apply(frame, ~risky, "ST/退市名称", rejections)
        else:
            notes.append(
                "名称列缺失 ⇒ ST / 退市名称**未被剔除**。请检查 instrument_master 是否同步。"
            )

    # -- 2. 题材命中 / 排除 ------------------------------------------------ #
    if spec.themes:
        keep = frame["theme_tags"].map(lambda tags: matches_themes(tags, spec.themes))
        label = f"题材未命中({'/'.join(spec.themes)})"
        frame, _ = _apply(frame, keep, label, rejections)
    if spec.exclude_themes:
        hit = frame["theme_tags"].map(lambda tags: matches_themes(tags, spec.exclude_themes))
        label = f"题材被排除({'/'.join(spec.exclude_themes)})"
        frame, _ = _apply(frame, ~hit, label, rejections)

    # -- 3. 数值区间 -------------------------------------------------------- #
    for name, (low, high) in spec.filters.items():
        if name not in frame.columns:
            # 未知列在 spec 层已被拒；走到这里说明是「归档没提供该因子」。
            rejections[f"{name}:列不存在"] = len(frame)
            frame = frame.iloc[0:0]
            notes.append(
                f"因子列 {name} 在数据里不存在 ⇒ 该条件下无结果"
                "（这是「条件无法评估」，不是「条件被忽略」）。"
            )
            continue

        values = pd.to_numeric(frame[name], errors="coerce")
        missing = values.isna()
        bound = low if low is not None else high if high is not None else 0.0
        # 缺失行先填一个**落在区间内**的占位值，使 `inside` 对它们为 True。
        # 这样「越界」与「缺失」是两种互斥、且各自可数的原因 ——
        # 若直接在含 NaN 的列上比较，NaN 会静默变成 False 并被算进「不在区间」，
        # 于是「数据缺失」和「真的越界」混成同一个数字，分不出来。
        filled = values.fillna(bound)
        inside = pd.Series(True, index=frame.index)
        if low is not None:
            inside &= filled >= low
        if high is not None:
            inside &= filled <= high

        out_of_range = ~inside & ~missing
        kept_by_this = inside & ~missing
        if out_of_range.any():
            rejections[f"{name} 不在区间"] = int(out_of_range.sum())
        if missing.any():
            rejections[f"{name}:缺失"] = int(missing.sum())
        frame = frame.loc[kept_by_this]

    # -- 4. 打分 ----------------------------------------------------------- #
    frame = add_scores(frame, weights)

    # -- 5. 排序 ----------------------------------------------------------- #
    sort_key = spec.sort
    if sort_key == "score" or sort_key not in frame.columns:
        sort_key = "score"
    key = pd.to_numeric(frame[sort_key], errors="coerce")
    unrankable = int(key.isna().sum())
    if unrankable:
        rejections[f"排序字段 {sort_key}:缺失"] = unrankable
        frame = frame.loc[key.notna()]
        key = key.loc[key.notna()]
    frame = frame.assign(_sort_key=key).sort_values(
        "_sort_key", ascending=not spec.descending, kind="stable"
    )
    pool_size = len(frame)
    frame = frame.drop(columns=["_sort_key"]).head(int(spec.limit)).reset_index(drop=True)
    # 操作方案只对最终展示的推荐生成，且不参与过滤、排序或综合分。
    frame = add_operation_plan(frame, as_of=table.as_of)

    ordered = _preferred_columns(frame)
    return ScreenResult(
        frame=frame.loc[:, ordered],
        spec=spec,
        as_of=table.as_of,
        universe_size=universe_size,
        pool_size=pool_size,
        rejections=rejections,
        notes=notes,
        weights=dict(weights),
    )


def _apply(
    frame: pd.DataFrame, keep: pd.Series, label: str, rejections: dict[str, int]
) -> tuple[pd.DataFrame, int]:
    keep = keep.reindex(frame.index, fill_value=False).astype(bool)
    dropped = int((~keep).sum())
    if dropped:
        rejections[label] = dropped
    return frame.loc[keep], dropped


def _preferred_columns(frame: pd.DataFrame) -> list[str]:
    """把最该看的列排前面：标识 → 打分 → 位置 → 动量 → 题材 → 资金 → 其它。"""

    head = [
        "symbol",
        "name",
        "industry",
        "close",
        "score",
        "score_coverage",
        "score_industry",
        "score_growth",
        "score_quality",
        "score_valuation",
        "score_momentum",
        "score_risk",
        "score_theme",
        "score_capital",
        "score_liquidity",
        "ret_since_924",
        "ret_ytd",
        "close_position_250",
        "dist_to_52w_high",
        "close_position_20",
        "history_days",
        "listed_days",
        "ret_20",
        "ret_60",
        "relative_ret_20",
        "relative_ret_60",
        "realized_vol_20",
        "downside_vol_20",
        "max_drawdown_60",
        "signed_amount_pressure_20",
        "volume_trend_20",
        "turnover_value_20",
        "theme_tags",
        "theme_heat_max",
        "theme_days_20",
        "lhb_count_20",
        "lhb_net_buy_rel",
        "main_net_inflow",
        "margin_rz_chg_20",
        "pe_ttm",
        "pe_industry_value",
        "yoy_net_income",
        "revenue_yoy",
        "profit_growth_acceleration",
        "positive_growth_streak",
        "roe_avg",
        "gross_profit_margin",
        "cfo_to_np",
        "liability_to_asset",
        "operation_signal",
        "reference_price",
        "buy_low",
        "buy_high",
        "pullback_price",
        "no_chase_price",
        "invalidation_price",
        "valuation_anchor",
        "target_1",
        "target_2",
        "expectation_state",
        "operation_weekday",
        "operation_reason",
    ]
    return [name for name in head if name in frame.columns] + [
        name for name in frame.columns if name not in head
    ]


__all__ = ["RISK_NAME_MARKERS", "RankWeightError", "ScreenResult", "run_screen"]
