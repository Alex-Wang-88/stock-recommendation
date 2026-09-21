"""价格类因子 —— **全确定性、可解释、无主观筛选**。

关于「低位」的一条硬规则
----------------------
`close_position_*` / `dist_to_52w_high` 这类**位置因子只做筛选条件，不做排序加分**
（方案 §五.3）。理由：低位既可能是价值（错杀），也可能是价值陷阱（基本面恶化）。
它适合把 5,000 只缩到 200 只，**不适合**决定买谁 —— 后者需要动量/资金/题材这类
**方向性**证据。这条规则由 `rank/score.py` 机械保证（位置因子进权重会直接 raise）。

关于缺失值的一条原则
------------------
滚动窗口**要求满窗**（`min_periods = window`）。上市不足 250 个交易日的股票
`close_position_250` 就是 NaN —— **不要去缩短窗口把它算出来**，那是拿 30 天的区间
冒充一年的位置。调用方应通过 `history_days` 显式要求历史长度；
筛选时 NaN 一律判不通过，并由 `screen/engine.py` 披露"因缺失被剔除"的数量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 位置类因子：**只允许做筛选条件，禁止进排序权重**。
POSITION_FACTORS = (
    "close_position_20",
    "close_position_60",
    "close_position_250",
    "dist_to_52w_high",
)

PRICE_FACTORS = (
    "symbol",
    "trade_date",
    "close",
    "amount",
    "history_days",
    "base_924",
    "base_ytd",
    "ret_since_924",
    "ret_ytd",
    "close_position_20",
    "close_position_60",
    "close_position_250",
    "dist_to_52w_high",
    "ma20_dev",
    "ma60_dev",
    "amount_avg_20",
)


def _rolled(frame: pd.DataFrame, column: str, window: int, how: str) -> pd.Series:
    """按 symbol 分组做满窗滚动。返回与 `frame` 同索引的 Series。"""

    grouped = frame.groupby("symbol", sort=False)[column]
    rolled = getattr(grouped.rolling(window, min_periods=window), how)()
    return rolled.reset_index(level=0, drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """分母为 0 / NaN 时给 NaN，而不是 inf。"""

    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_price_factors(
    bars: pd.DataFrame,
    base_closes: pd.DataFrame | None = None,
    *,
    windows: tuple[int, ...] = (20, 60, 250),
) -> pd.DataFrame:
    """把日线压成「每个 symbol 一行」的价格因子表（as-of = bars 的最后一个交易日）。

    参数
    ----
    bars
        `MarketStore.bars()` 的输出，按 symbol / trade_date 升序。
    base_closes
        `MarketStore.closes_on({"base_924": ..., "base_ytd": ...})` 的输出。
        缺失时 `ret_since_924` / `ret_ytd` 为 NaN（不猜）。
    """

    if bars is None or bars.empty:
        return pd.DataFrame(columns=list(PRICE_FACTORS))

    frame = bars.sort_values(["symbol", "trade_date"], kind="stable").copy()
    windows = tuple(sorted(set(windows)))

    frame["history_days"] = frame.groupby("symbol", sort=False).cumcount() + 1
    for window in windows:
        frame[f"_high_{window}"] = _rolled(frame, "close", window, "max")
        frame[f"_low_{window}"] = _rolled(frame, "close", window, "min")
    frame["_ma20"] = _rolled(frame, "close", 20, "mean")
    frame["_ma60"] = _rolled(frame, "close", 60, "mean")
    frame["_amount_20"] = _rolled(frame, "amount", 20, "mean")

    last = frame.groupby("symbol", sort=False).tail(1).set_index("symbol")

    out = pd.DataFrame(index=last.index)
    out["trade_date"] = last["trade_date"]
    out["close"] = last["close"]
    out["amount"] = last["amount"]
    out["history_days"] = last["history_days"]
    out["amount_avg_20"] = last["_amount_20"]

    for window in windows:
        span = last[f"_high_{window}"] - last[f"_low_{window}"]
        out[f"close_position_{window}"] = _safe_ratio(last["close"] - last[f"_low_{window}"], span)

    if 250 in windows:
        out["dist_to_52w_high"] = _safe_ratio(last["close"], last["_high_250"]) - 1.0

    out["ma20_dev"] = _safe_ratio(last["close"], last["_ma20"]) - 1.0
    out["ma60_dev"] = _safe_ratio(last["close"], last["_ma60"]) - 1.0

    # 基准收盘 → 区间涨幅。分子分母都在同一复权口径（ADJUST_PREV），可比。
    base_closes = base_closes if base_closes is not None else pd.DataFrame(columns=["symbol"])
    out = out.reset_index().merge(base_closes, on="symbol", how="left")
    for label, name in (("base_924", "ret_since_924"), ("base_ytd", "ret_ytd")):
        if label in out.columns:
            out[name] = _safe_ratio(out["close"], out[label]) - 1.0
        else:
            out[label] = pd.NA
            out[name] = pd.NA

    missing = [column for column in PRICE_FACTORS if column not in out.columns]
    for column in missing:
        out[column] = pd.NA
    return out.loc[:, list(PRICE_FACTORS)].reset_index(drop=True)
