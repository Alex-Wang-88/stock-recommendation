"""题材类因子 —— 把「同花顺题材归因」变成可计算的方向性因子。

数据来源与边界
------------
`theme_attribution` 是**当日强势股名单**（不是全市场），因此本模块的因子是
「这只股票**今天有没有被题材解释**、被几个题材解释、这些题材今天有多热」。

⚠️ 这是**归因**数据，不是**预测**数据。它只说明"市场今天把这只股票的上涨归到哪些
题材上"，不说明"明天会涨"。因此它进排序、不进结论；阶段 3 的状态机才会把它变成
「题材在生命周期哪个阶段」的判断。

⚠️ `theme_universe_bias`：因为名单只含强势股，本模块的因子**天然带选择偏差**（被覆盖的
样本本就更强）。所以它**不能**用来做"全市场筛选"的主条件 —— 只能作为
"题材维度"的叠加信息。想按题材选股，应该用 `theme_tags` 做**命中过滤**，
再用其它因子横向比较。
"""

from __future__ import annotations

import pandas as pd

THEME_FACTORS = (
    "symbol",
    "theme_tags",
    "theme_tag_count",
    "theme_heat_max",
    "theme_heat_sum",
    "theme_days_20",
    "reason",
)


def add_theme_factors(
    universe: pd.DataFrame,
    attribution: pd.DataFrame,
    tag_daily: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """给 `universe`（至少含 `symbol`）补上题材因子。

    参数
    ----
    attribution
        `ArchiveStore.themes_on(as_of)`：`symbol / reason / change_pct / big_order_net ...`
    tag_daily
        `ArchiveStore.tag_counts_on(as_of)`：`tag / stock_count`
    history
        `ArchiveStore.theme_history(symbols, start, end)`：`symbol / trade_date`，
        用于算 `theme_days_20`（近 20 个交易日上榜几天）。缺失则该因子为 NaN。

    `theme_heat_max` = 该股所属题材中**最热**那个的当日上榜股票数；
    `theme_heat_sum` = 所属题材上榜股票数之和。两者都不做归一化 ——
    归一化系数（全市场当日上榜总数）会随行情波动，直接给原始计数更可解释。
    """

    out = universe.copy()
    if out.empty:
        for column in THEME_FACTORS:
            if column not in out.columns:
                out[column] = pd.NA
        return out

    tags_by_symbol = (
        attribution.loc[:, ["symbol", "reason"]].drop_duplicates(subset=["symbol"])
        if not attribution.empty
        else pd.DataFrame(columns=["symbol", "reason"])
    )
    out = out.merge(tags_by_symbol, on="symbol", how="left")

    heat = tag_daily.set_index("tag")["stock_count"].to_dict() if not tag_daily.empty else {}
    out["theme_tags"] = out["reason"].map(_split)

    if not attribution.empty:
        # 名单非空 ⇒ "不在名单里" 是**真实信号**（今天不是强势股）⇒ 热度填 0。
        out["theme_tag_count"] = out["theme_tags"].map(len)
        out["theme_heat_max"] = out["theme_tags"].map(
            lambda tags: max((heat.get(tag, 0) for tag in tags), default=0) if tags else 0
        )
        out["theme_heat_sum"] = out["theme_tags"].map(
            lambda tags: sum(heat.get(tag, 0) for tag in tags) if tags else 0
        )
    else:
        # 名单为空 ⇒ 我们**不知道**，不能填 0。填 0 会让「归档没回补」看起来像
        # 「今天所有股票都没题材热度」，进而让依赖题材的条件静默失效。
        out["theme_tag_count"] = pd.NA
        out["theme_heat_max"] = pd.NA
        out["theme_heat_sum"] = pd.NA

    out["theme_tags"] = out["theme_tags"].map(lambda tags: "|".join(tags) if tags else None)

    if history is not None and not history.empty:
        counts = (
            history.groupby("symbol")["trade_date"].nunique().rename("theme_days_20").reset_index()
        )
        out = out.merge(counts, on="symbol", how="left")
        # 历史已按 universe 查询 ⇒ 查不到就是真的 0 天，不是缺失。
        out["theme_days_20"] = out["theme_days_20"].fillna(0).astype("int64")
    else:
        out["theme_days_20"] = pd.NA

    for column in THEME_FACTORS:
        if column not in out.columns:
            out[column] = pd.NA
    # 因子列排前面，universe 自带的列（close/amount/...）跟在后面。
    ordered = list(THEME_FACTORS) + [c for c in out.columns if c not in THEME_FACTORS]
    return out.loc[:, ordered]


def _split(reason: object) -> list[str]:
    """`reason` 拆标签。复用归档层的分隔符定义，避免两处口径漂移。"""

    from ..archive.store import split_tags

    if reason is None or (isinstance(reason, float) and pd.isna(reason)):
        return []
    return split_tags(str(reason))


def matches_themes(tags: object, wanted: tuple[str, ...]) -> bool:
    """`theme_tags` 是否命中 `wanted` 中**任一**标签。

    用**整个标签**精确匹配，而不是子串 —— 「算力」不应命中「算力租赁」这种
    看起来相关但语义更窄的标签，反之亦然。宁可漏，不可错配。
    """

    if not wanted:
        return True
    if tags is None or (isinstance(tags, float) and pd.isna(tags)):
        return False
    present = {part for part in str(tags).split("|") if part}
    return any(tag in present for tag in wanted)
