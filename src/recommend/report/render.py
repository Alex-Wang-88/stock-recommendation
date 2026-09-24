"""输出层 —— 把候选表渲染成「清单 + 因子贡献 + 风控提示 + 数据时点」。

一条推荐如果没有**数据时点**，读者就无法判断它是不是在拿陈旧数据当今天。
所以 `data_as_of` 是每次输出的固定段落，不是可选项。
"""

from __future__ import annotations

import pandas as pd

from ..rank.score import explain
from ..screen.engine import ScreenResult

# 表格里显示哪些列（控制宽度；完整明细在 CSV 里）。
DISPLAY_COLUMNS = (
    "symbol",
    "name",
    "industry",
    "close",
    "operation_signal",
    "announcement_alert",
    "buy_low",
    "buy_high",
    "invalidation_price",
    "target_1",
    "target_2",
    "expectation_state",
    "score",
    "score_coverage",
    "pe_ttm",
    "pe_industry_value",
    "eps_ttm",
    "roe_avg",
    "gross_profit_margin",
    "net_profit_margin",
    "revenue_yoy",
    "profit_growth_acceleration",
    "positive_growth_streak",
    "cfo_to_np",
    "liability_to_asset",
    "industry_score",
    "industry_leader_score",
    "ret_since_924",
    "ret_ytd",
    "close_position_250",
    "dist_to_52w_high",
    "ret_20",
    "relative_ret_20",
    "relative_ret_60",
    "realized_vol_20",
    "downside_vol_20",
    "max_drawdown_60",
    "signed_amount_pressure_20",
    "volume_trend_20",
    "turnover_value_20",
    "theme_heat_max",
    "theme_days_20",
    "theme_tags",
    "lhb_count_20",
    "lhb_net_buy_rel",
    "margin_rz_chg_20",
)

PCT_COLUMNS = (
    "ret_since_924",
    "ret_ytd",
    "ret_20",
    "ret_60",
    "ma20_dev",
    "ma60_dev",
    "dist_to_52w_high",
    "realized_vol_20",
    "max_drawdown_60",
    "signed_amount_pressure_20",
    "margin_rz_chg_20",
    "roe_avg",
    "gross_profit_margin",
    "net_profit_margin",
    "yoy_net_income",
    "yoy_eps_basic",
    "yoy_profit_net_income",
    "revenue_yoy",
    "profit_growth_acceleration",
    "eps_growth_acceleration",
    "relative_ret_20",
    "relative_ret_60",
    "liability_to_asset",
    "cfo_to_or",
    "cfo_to_np",
    "cfo_to_gr",
    "downside_vol_20",
)

# 定点小数位数。
DECIMAL_COLUMNS = {
    "close": 2,
    "buy_low": 2,
    "buy_high": 2,
    "pullback_price": 2,
    "no_chase_price": 2,
    "invalidation_price": 2,
    "valuation_anchor": 2,
    "target_1": 2,
    "target_2": 2,
    "reference_price": 2,
    "score": 3,
    "close_position_20": 2,
    "close_position_60": 2,
    "close_position_250": 2,
    "ret_rank_20": 2,
    "volume_trend_20": 2,
    "score_coverage": 2,
    "lhb_net_buy_rel": 3,
    "pe_ttm": 2,
    "eps_ttm": 2,
    "industry_score": 2,
    "industry_leader_score": 2,
    "pe_industry_value": 3,
    "current_ratio": 2,
    "quick_ratio": 2,
    "cash_ratio": 2,
    "asset_to_equity": 2,
    "ebit_to_interest": 2,
    "positive_profit_streak": 0,
    "positive_growth_streak": 0,
}

# 整数计数列。
COUNT_COLUMNS = (
    "theme_heat_max",
    "theme_heat_sum",
    "theme_tag_count",
    "theme_days_20",
    "lhb_count_20",
    "history_days",
    "listed_days",
    "positive_profit_streak",
    "positive_growth_streak",
)

# 金额列：用**带单位后缀**的亿元显示。
# 不改成"列名写亿元、值写数字"的写法 —— 那样一旦有人把表复制到别处，单位就丢了。
MONEY_COLUMNS = (
    "turnover_value_20",
    "amount_avg_20",
    "lhb_net_buy_20",
    "main_net_inflow",
)


def _money(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) / 1e8:.2f}亿"


def to_display(frame: pd.DataFrame) -> pd.DataFrame:
    """挑出展示列并格式化。

    ⚠️ 用显式的格式表，别用 `dtype == object` 判断"是不是文本列" ——
    pandas 3.x 会把字符串列变成 `StringDtype`，`== object` 为假，
    于是一个缺失标签会被渲染成字面量 `nan`，看起来像一个真实题材名。
    """

    columns = [name for name in DISPLAY_COLUMNS if name in frame.columns]
    view = frame.loc[:, columns].copy()
    for name in view.columns:
        if name in PCT_COLUMNS:
            view[name] = view[name].map(
                lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%"
            )
        elif name in MONEY_COLUMNS:
            view[name] = view[name].map(_money)
        elif name in DECIMAL_COLUMNS:
            digits = DECIMAL_COLUMNS[name]
            view[name] = view[name].map(
                lambda value, d=digits: "" if pd.isna(value) else f"{value:.{d}f}"
            )
        elif name in COUNT_COLUMNS:
            view[name] = view[name].map(lambda value: "" if pd.isna(value) else f"{int(value)}")
        else:
            # 兜底：文本列（题材标签等）。NA → 空串。
            view[name] = view[name].map(lambda value: "" if _is_na(value) else str(value))
    if "theme_tags" in view.columns:
        # 内部用 `|` 分隔标签（与归档层一致），但 `|` 是 Markdown 表格的分隔符 ——
        # 直接渲染会把一个标签拆成两列。展示层换成顿号。
        view["theme_tags"] = view["theme_tags"].map(
            lambda value: value.replace("|", "、") if value else ""
        )
    return view


def _is_na(value: object) -> bool:
    """标量缺失判断（`pd.isna` 对 list 会返回数组，这里只处理标量）。"""

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _cell(value: object) -> str:
    """Markdown 单元格转义：竖线与换行都会破坏表格结构。"""

    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _price(value: object) -> str:
    """价格字段的报告格式；缺失保持空，不伪造价格。"""

    if _is_na(value):
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _first_value(row: pd.Series, *names: str) -> object:
    """返回第一个非缺失值，避免对 pandas.NA 使用布尔判断。"""

    for name in names:
        value = row.get(name)
        if not _is_na(value):
            return value
    return None


def markdown_table(view: pd.DataFrame) -> str:
    """自己渲染 Markdown 表格，不引入 `tabulate` 依赖。"""

    if view.empty:
        return "（空）"
    header = "| " + " | ".join(_cell(column) for column in view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = [
        "| " + " | ".join(_cell(value) for value in record) + " |"
        for record in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def render_markdown(
    result: ScreenResult, *, spec_hash: str | None = None, sync_run_id: str | None = None
) -> str:
    """完整报告：条件 → 数据时点 → 剔除日志 → 候选表 → 每条的理由。"""

    lines: list[str] = []
    lines.append(f"# 筛选结果 · as-of {result.as_of}")
    lines.append("")
    if spec_hash or sync_run_id:
        lines.append("## 留证标识")
        if spec_hash:
            lines.append(f"- 规则指纹：`{spec_hash}`")
        if sync_run_id:
            lines.append(f"- 同步批次：`{sync_run_id}`")
        lines.append("")
    lines.append("## 筛选条件")
    lines.extend(f"- {line}" for line in result.spec.describe())
    lines.append("")

    lines.append("## 数据时点（每条推荐的日期口径）")
    lines.append(f"- 行情 / 因子：{result.as_of}")
    lines.append(f"- 题材 / 龙头 / 两融：{result.as_of} 及之前 20 个交易日")
    lines.append("- ⚠️ 名称（含 ST 标记）取**当前**状态；本表不用于历史回看。")
    lines.append("")
    lines.append("## 今日操作口径")
    lines.append(
        "- 以查看日的日线收盘价作为参考价，生成分批买入区间、回落触发价、"
        "逻辑失效价和两个参考目标价。"
    )
    lines.append(
        "- 操作方案只读取已有日线、均线、波动、位置、行业估值和基本面代理，"
        "不使用分钟数据，也不改变综合评分和推荐排序。"
    )
    lines.append("- 价格是研究参考，不是自动下单指令；短期兑现状态不是分析师一致预期。")
    lines.append("")

    lines.append("## 漏斗")
    lines.append(f"- 全市场有行情标的：{result.universe_size} 只")
    for line in result.rejection_lines():
        lines.append(line)
    lines.append(f"- 通过全部条件：**{result.pool_size} 只**")
    if result.truncated:
        lines.append(
            f"- 被 limit 截断：{result.truncated} 只"
            "（**未显示 ≠ 未通过**；要全量请看同目录 CSV）"
        )
    lines.append(f"- 本表显示：**{result.kept} 只**")
    for note in result.notes:
        lines.append(f"- ⚠️ {note}")
    lines.append("")

    lines.append("## 候选清单")
    if result.frame.empty:
        lines.append("（无命中。请检查上面的剔除日志 —— 很可能是某个因子整列缺失。）")
        lines.append("")
        return "\n".join(lines)
    lines.append(markdown_table(to_display(result.frame)))
    lines.append("")

    lines.append("## 逐条理由（因子贡献明细）")
    for _, row in result.frame.iterrows():
        symbol = row.get("symbol")
        name = row.get("name") or ""
        score = row.get("score")
        coverage = row.get("score_coverage")
        lines.append(f"### {symbol} {name} — score {score:.3f}（权重覆盖 {coverage:.0%}）")
        lines.extend(explain(row.to_dict(), result.weights))
        tags = row.get("theme_tags")
        if isinstance(tags, str) and tags:
            lines.append(f"  - 题材归因：{tags}")
        action = row.get("operation_signal")
        if not _is_na(action):
            lines.append(f"  - 今日操作：{action}（{row.get('operation_weekday', '')}）")
            lines.append(
                f"  - 参考价：{_price(_first_value(row, 'reference_price', 'close'))}；"
                f"分批买入区间：{_price(row.get('buy_low'))}–{_price(row.get('buy_high'))}；"
                f"回落触发：{_price(row.get('pullback_price'))}"
            )
            lines.append(
                f"  - 不追价上限：{_price(row.get('no_chase_price'))}；"
                f"逻辑失效参考：{_price(row.get('invalidation_price'))}；"
                f"目标一/目标二：{_price(row.get('target_1'))} / {_price(row.get('target_2'))}"
            )
            lines.append(f"  - 短期兑现：{row.get('expectation_state', '')}")
            reason = row.get("operation_reason")
            if not _is_na(reason):
                lines.append(f"  - 操作说明：{reason}")
        lines.append("")
    return "\n".join(lines)


__all__ = ["DISPLAY_COLUMNS", "PCT_COLUMNS", "markdown_table", "render_markdown", "to_display"]
