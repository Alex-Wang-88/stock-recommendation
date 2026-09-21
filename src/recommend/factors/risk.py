"""本地风险/确认因子。

这些因子只使用已有日线，不增加外部数据依赖，也不自动进入默认评分权重：

* ``realized_vol_20``：近 20 个交易日收益的实现波动率；
* ``max_drawdown_60``：近 60 个交易日从历史峰值到随后低点的最大回撤；
* ``downside_vol_20``：只统计下跌日的 20 日波动率；
* ``signed_amount_pressure_20``：按收盘涨跌给成交额加符号后的 20 日净压力。

原始风险值保留给页面展示，同时生成方向统一的 ``*_value``：波动率越低、回撤越小，
方向分越高。这样排序层可以透明地加入风险权重，而不会把低风险误当成低分。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_FACTORS = (
    "symbol",
    "realized_vol_20",
    "max_drawdown_60",
    "downside_vol_20",
    "signed_amount_pressure_20",
    "volatility_value",
    "drawdown_value",
    "downside_volatility_value",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _last_realized_vol(close: pd.Series, window: int) -> float | None:
    if len(close) < window + 1:
        return None
    previous = close.shift(1)
    returns = _safe_ratio(close, previous) - 1.0
    values = returns.tail(window)
    if values.isna().any():
        return None
    return float(values.std(ddof=0))


def _last_max_drawdown(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    values = close.tail(window)
    if values.isna().any():
        return None
    peaks = np.maximum.accumulate(values.to_numpy(dtype=float))
    drawdowns = values.to_numpy(dtype=float) / peaks - 1.0
    return float(np.min(drawdowns))


def _last_signed_pressure(group: pd.DataFrame, window: int) -> float | None:
    if len(group) < window + 1:
        return None
    close = group["close"].reset_index(drop=True)
    amount = pd.to_numeric(group["amount"], errors="coerce").reset_index(drop=True)
    close_window = close.tail(window + 1)
    amount_window = amount.tail(window)
    if close_window.isna().any() or amount_window.isna().any():
        return None
    direction = np.sign(close_window.diff().to_numpy(dtype=float)[1:])
    signed_amount = amount_window.to_numpy(dtype=float) * direction
    denominator = float(amount_window.sum())
    if denominator == 0:
        return None
    return float(signed_amount.sum() / denominator)


def _last_downside_vol(close: pd.Series, window: int) -> float | None:
    if len(close) < window + 1:
        return None
    returns = _safe_ratio(close, close.shift(1)).sub(1.0).tail(window)
    if returns.isna().any():
        return None
    negative = returns[returns < 0]
    # 没有下跌日是有效的低下行风险，不应因为缺少负收益而变成 NaN。
    return float(negative.std(ddof=0)) if len(negative) > 1 else 0.0


def add_risk_factors(
    bars: pd.DataFrame,
    *,
    volatility_window: int = 20,
    drawdown_window: int = 60,
    pressure_window: int = 20,
) -> pd.DataFrame:
    """把日线压成每个 symbol 一行的补充风险/确认因子表。"""

    if bars is None or bars.empty:
        return pd.DataFrame(columns=list(RISK_FACTORS))

    frame = bars.sort_values(["symbol", "trade_date"], kind="stable").copy()
    rows: list[dict[str, object]] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        close = pd.to_numeric(group["close"], errors="coerce").reset_index(drop=True)
        rows.append(
            {
                "symbol": symbol,
                "realized_vol_20": _last_realized_vol(close, volatility_window),
                "max_drawdown_60": _last_max_drawdown(close, drawdown_window),
                "downside_vol_20": _last_downside_vol(close, volatility_window),
                "signed_amount_pressure_20": _last_signed_pressure(group, pressure_window),
            }
        )
    out = pd.DataFrame(rows, columns=list(RISK_FACTORS))
    out["volatility_value"] = -pd.to_numeric(out["realized_vol_20"], errors="coerce")
    out["drawdown_value"] = pd.to_numeric(out["max_drawdown_60"], errors="coerce")
    out["downside_volatility_value"] = -pd.to_numeric(
        out["downside_vol_20"], errors="coerce"
    )
    return out.loc[:, list(RISK_FACTORS)]


__all__ = ["RISK_FACTORS", "add_risk_factors"]
