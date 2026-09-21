"""资金类因子 —— 龙虎榜、两融、当日主力净额。

口径说明
-------
* `lhb_count_20` / `lhb_net_buy_20` —— 近 20 个交易日**龙虎榜**上榜次数与净买合计。
  上榜本身就是"异动"的官方标签，且**买卖席位数据可回溯**（T1）。
* `main_net_inflow` —— 当日主力净额。来自题材归因响应里带的字段，
  **只对当日上榜的强势股有值** ⇒ 缺失即 NaN，不要填 0（填 0 等于宣称"今天没有资金流入"）。
* `margin_rz_chg_20` —— 融资余额 20 日变化率。两融数据需要按标的回补，
  归档可能为空 ⇒ 整列 NaN，并且**不把 NaN 当 0 参与排序**。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAPITAL_FACTORS = (
    "symbol",
    "lhb_count_20",
    "lhb_net_buy_20",
    "lhb_net_buy_rel",
    "main_net_inflow",
    "margin_rz_chg_20",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_capital_factors(
    universe: pd.DataFrame,
    attribution: pd.DataFrame | None = None,
    dragon_tiger: pd.DataFrame | None = None,
    margin: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """给 `universe`（至少含 `symbol`）补上资金因子。

    `dragon_tiger` / `margin` 应已由调用方**按窗口过滤**（只含近 20 个交易日）。
    """

    out = universe.copy()
    if out.empty:
        for column in CAPITAL_FACTORS:
            if column not in out.columns:
                out[column] = pd.NA
        return out

    # 当日主力净额（只对当日上榜股有值）
    if attribution is not None and not attribution.empty and "big_order_net" in attribution.columns:
        main = (
            attribution.loc[:, ["symbol", "big_order_net"]]
            .drop_duplicates(subset=["symbol"])
            .rename(columns={"big_order_net": "main_net_inflow"})
        )
        out = out.merge(main, on="symbol", how="left")
    else:
        out["main_net_inflow"] = pd.NA

    # 龙虎榜
    lhb_available = dragon_tiger is not None and not dragon_tiger.empty
    if lhb_available:
        grouped = dragon_tiger.groupby("symbol")
        lhb_count = grouped["trade_date"].nunique()
        # `sum()` 默认把全 NaN 聚合成 0。新浪备用源没有席位金额，
        # 所以这里必须要求至少一个真实金额，否则会把「未知」伪装成「净买 0」。
        lhb_net_buy = grouped["net_buy"].sum(min_count=1)
        lhb = pd.DataFrame(
            {"lhb_count_20": lhb_count, "lhb_net_buy_20": lhb_net_buy}
        )
        out = out.merge(lhb.reset_index(), on="symbol", how="left")
        # 归档可用 ⇒ 不在名单里就是真的没上榜，填 0 是对的。
        out["lhb_count_20"] = out["lhb_count_20"].fillna(0).astype("int64")
    else:
        # 归档为空（没回补）⇒ 我们**不知道**，不能填 0 —— 那等于宣称"没上榜"。
        out["lhb_count_20"] = pd.NA
        out["lhb_net_buy_20"] = pd.NA

    if "turnover_value_20" in out.columns:
        out["lhb_net_buy_rel"] = _safe_ratio(out["lhb_net_buy_20"], out["turnover_value_20"])
    else:
        out["lhb_net_buy_rel"] = pd.NA

    # 两融
    if margin is not None and not margin.empty and "rzye" in margin.columns:
        margin = margin.sort_values(["symbol", "trade_date"])
        grouped = margin.groupby("symbol")["rzye"]
        first = grouped.transform("first")
        last = grouped.transform("last")
        change = _safe_ratio(last, first) - 1.0
        latest = (
            pd.DataFrame({"symbol": margin["symbol"], "margin_rz_chg_20": change})
            .groupby("symbol", sort=False)
            .tail(1)
        )
        out = out.merge(latest, on="symbol", how="left")
    else:
        out["margin_rz_chg_20"] = pd.NA

    for column in CAPITAL_FACTORS:
        if column not in out.columns:
            out[column] = pd.NA
    ordered = list(CAPITAL_FACTORS) + [c for c in out.columns if c not in CAPITAL_FACTORS]
    return out.loc[:, ordered]
