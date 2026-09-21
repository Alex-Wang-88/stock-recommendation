"""动量 / 量价类因子 —— **方向性**因子，可以进排序。

与价格因子（`price.py`）的分工
---------------------------
`price.py` 回答「它现在在什么位置」（水平量），本模块回答「它正在往哪走」（方向量）。
排序只用方向量 —— 位置量只做筛选条件。

`ret_rank_20` 为什么用横截面分位而不是减去指数
------------------------------------------
本项目同时保留全市场等权相对收益，回答「股票是否跑赢市场」；它只用本地日线，
不增加指数数据依赖。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_FACTORS = (
    "symbol",
    "ret_20",
    "ret_60",
    "relative_ret_20",
    "relative_ret_60",
    "ret_rank_20",
    "volume_trend_20",
    "turnover_value_20",
)


def _rolled(frame: pd.DataFrame, column: str, window: int, how: str) -> pd.Series:
    grouped = frame.groupby("symbol", sort=False)[column]
    rolled = getattr(grouped.rolling(window, min_periods=window), how)()
    return rolled.reset_index(level=0, drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_momentum_factors(
    bars: pd.DataFrame, *, entry_window: int = 5, base_window: int = 20
) -> pd.DataFrame:
    """把日线压成「每个 symbol 一行」的动量因子表（as-of = bars 的最后一个交易日）。

    * `ret_20` / `ret_60` —— 区间收益（用收盘价，同复权口径）。
    * `ret_rank_20` —— `ret_20` 在全体标的里的**分位**（0–1），横截面可比。
    * `volume_trend_20` —— 量比：近 `entry_window` 日均量 ÷ 近 `base_window` 日均量。
      >1 表示近期放量。**只用成交量，不用成交额** —— 成交额受价格影响，会和收益因子共线。
    * `turnover_value_20` —— 近 20 日日均成交额，用于流动性门槛与排序。
    """

    if bars is None or bars.empty:
        return pd.DataFrame(columns=list(MOMENTUM_FACTORS))

    frame = bars.sort_values(["symbol", "trade_date"], kind="stable").copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    # 5 日均量 / 20 日均量：这两个窗口都要求满窗，否则次新股的量比会跳来跳去。
    frame["_vol_short"] = _rolled(frame, "volume", entry_window, "mean")
    frame["_vol_long"] = _rolled(frame, "volume", base_window, "mean")
    frame["_amount_20"] = _rolled(frame, "amount", base_window, "mean")

    # 区间收益：取 n 根之前的收盘。上市不足 n 根 ⇒ shift 给 NaN ⇒ 收益为 NaN（不猜）。
    for window in (20, 60):
        shifted = frame.groupby("symbol", sort=False)["close"].shift(window)
        frame[f"_ret_{window}"] = _safe_ratio(frame["close"], shifted) - 1.0

    # 用全市场等权日收益构造内部基准。只使用当天及之前的本地行情，停牌/缺失
    # 标的不会被填成收益 0，而是在当日均值中自然缺席。
    frame["_daily_ret"] = frame.groupby("symbol", sort=False)["close"].pct_change()
    market_daily = frame.groupby("trade_date", sort=True)["_daily_ret"].mean()
    market_curve = (1.0 + market_daily.fillna(0.0)).cumprod()
    for window in (20, 60):
        market_period = market_curve / market_curve.shift(window) - 1.0
        frame[f"_market_ret_{window}"] = frame["trade_date"].map(market_period)

    last = frame.groupby("symbol", sort=False).tail(1).set_index("symbol")
    out = pd.DataFrame(index=last.index)
    out["ret_20"] = last["_ret_20"]
    out["ret_60"] = last["_ret_60"]
    out["relative_ret_20"] = last["_ret_20"] - last["_market_ret_20"]
    out["relative_ret_60"] = last["_ret_60"] - last["_market_ret_60"]
    out["volume_trend_20"] = _safe_ratio(last["_vol_short"], last["_vol_long"])
    out["turnover_value_20"] = last["_amount_20"]

    # 横截面分位：0 = 最弱，1 = 最强。NaN 保持 NaN（不参与排名，也不被填成中位数）。
    out["ret_rank_20"] = out["ret_20"].rank(pct=True)

    # `out` 的索引是 symbol（名字也叫 symbol），reset_index 之后才会变成列。
    # 所以「补缺失列」必须放在 reset_index **之后**，否则会插出一个重复的 symbol 列。
    out = out.reset_index()
    for column in MOMENTUM_FACTORS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, list(MOMENTUM_FACTORS)]
