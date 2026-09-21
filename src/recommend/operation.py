"""基于日线推荐结果生成一次性的操作参考卡。

本模块故意不参与综合评分，也不维护持仓状态。它只把已经完成排序的推荐，
按照查看日的日线、均线、波动、位置和估值信息，转换成一张可复核的操作方案：

* 今天查看的价格作为参考价；
* 给出可接受的分批买入区间和回落触发价；
* 给出原推荐逻辑失效的价格；
* 给出基于行业估值锚 / 历史高点的两个参考目标价；
* 用已有价格、量能、题材和基本面代理判断短期预期是否已经较充分兑现。

没有分钟数据时，不声称能判断早盘或 14:30 的具体买点。所有价格都是推荐快照的
参考值，不是自动下单指令，也不会改变原有排名。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date

import pandas as pd

OPERATION_COLUMNS = (
    "operation_weekday",
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
    "operation_reason",
)

WEEKDAY_LABELS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _number(row: Mapping[str, object], name: str) -> float | None:
    """读取有限数值；缺失保持 None，绝不把未知值当成 0。"""

    value = row.get(name)
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _moving_average(close: float, deviation: float | None) -> float | None:
    """由 ``close / MA - 1`` 还原均线，避免重复读取整段日线。"""

    if deviation is None or 1.0 + deviation <= 0:
        return None
    return close / (1.0 + deviation)


def _weekday(as_of: str | None) -> str:
    if not as_of:
        return "—"
    try:
        return WEEKDAY_LABELS[date.fromisoformat(str(as_of)[:10]).weekday()]
    except (TypeError, ValueError, IndexError):
        return "—"


def _expectation_state(row: Mapping[str, object]) -> tuple[str, str]:
    """用可回看代理给出预期兑现状态，不冒充分析师一致预期。"""

    ret_20 = _number(row, "ret_20")
    relative_20 = _number(row, "relative_ret_20")
    position_20 = _number(row, "close_position_20")
    distance_high = _number(row, "dist_to_52w_high")
    volume_trend = _number(row, "volume_trend_20")
    theme_days = _number(row, "theme_days_20")
    net_income = _number(row, "yoy_net_income")
    revenue = _number(row, "revenue_yoy")
    eps_value = _number(row, "pe_industry_value")

    # 只把多个独立的强势信号同时出现视为"短期兑现较充分"，避免单日涨幅
    # 就被误读成预期已经兑现。
    realized_signals = sum(
        value
        for value in (
            ret_20 is not None and ret_20 >= 0.12,
            position_20 is not None and position_20 >= 0.85,
            distance_high is not None and distance_high >= -0.05,
            volume_trend is not None and volume_trend >= 1.50,
            theme_days is not None and theme_days >= 4,
        )
    )
    if realized_signals >= 2:
        return (
            "短期兑现较充分",
            "20日涨幅、区间位置、距高点、量能或题材活跃度中至少两项偏高，当前更适合等回落。",
        )

    if (
        ret_20 is not None
        and relative_20 is not None
        and ret_20 > 0
        and relative_20 > 0
        and (volume_trend is None or volume_trend >= 1.0)
    ):
        return "正在兑现", "短期收益和相对大盘表现为正，价格正在验证推荐逻辑。"

    if (
        (net_income is not None and net_income > 0 or revenue is not None and revenue > 0)
        and ret_20 is not None
        and ret_20 <= 0
        and (eps_value is None or eps_value >= 0)
    ):
        return "基本面尚未充分反映", "增长代理为正但近20日价格尚未明显兑现，适合分批而非追价。"

    if (ret_20 is not None and ret_20 <= -0.08) or (
        relative_20 is not None and relative_20 <= -0.08
    ):
        return "兑现偏弱", "近20日收益或相对大盘收益偏弱，先确认基本面和行业逻辑没有恶化。"

    return "暂无明确代理", "当前已有数据不足以判断短期预期兑现程度，不把缺失值当成利好或利空。"


def _operation_row(row: Mapping[str, object], as_of: str | None) -> dict[str, object]:
    close = _number(row, "close")
    weekday = _weekday(as_of)
    state, state_reason = _expectation_state(row)
    empty = {name: pd.NA for name in OPERATION_COLUMNS}
    empty["operation_weekday"] = weekday
    empty["expectation_state"] = state
    empty["operation_reason"] = state_reason
    if close is None or close <= 0:
        empty["operation_signal"] = "暂缓"
        empty["operation_reason"] = "收盘价缺失，无法生成日线参考价格。"
        return empty

    volatility = _number(row, "realized_vol_20")
    # 日收益标准差不是止损幅度，只用来生成一个有上下限的参考缓冲。
    buffer = _clamp((volatility if volatility is not None else 0.02) * 1.8, 0.02, 0.08)

    ma20 = _moving_average(close, _number(row, "ma20_dev"))
    ma60 = _moving_average(close, _number(row, "ma60_dev"))
    supports = [close * (1.0 - buffer)]
    for moving_average in (ma20, ma60):
        if moving_average is not None and 0 < moving_average <= close:
            supports.append(moving_average)
    support = max(supports)

    # 当前价附近给出分批区间；回落触发价落在区间下沿，跌破失效价则需要重新审视推荐逻辑。
    buy_low = support
    pullback_price = support
    buy_high = close
    invalidation = support * (1.0 - max(buffer, 0.025))
    no_chase = close * (1.0 + min(max(buffer * 0.5, 0.01), 0.03))

    ret_20 = _number(row, "ret_20")
    position_20 = _number(row, "close_position_20")
    ma20_dev = _number(row, "ma20_dev")
    overextended = (
        (ma20_dev is not None and ma20_dev >= 0.06)
        or (
            position_20 is not None
            and position_20 >= 0.85
            and ret_20 is not None
            and ret_20 >= 0.08
        )
    )
    weak_trend = (
        ret_20 is not None
        and ret_20 <= -0.12
        and (_number(row, "ma60_dev") is not None and _number(row, "ma60_dev") <= -0.06)
    )
    coverage = _number(row, "score_coverage")

    if coverage is not None and coverage < 0.60:
        signal = "暂缓"
        signal_reason = "评分权重覆盖不足，先补齐关键数据。"
    elif weak_trend or state == "兑现偏弱":
        signal = "暂缓，先确认"
        signal_reason = "短期走势偏弱，先确认行业、基本面和价格是否重新转强。"
    elif overextended or state == "短期兑现较充分":
        signal = "等回落再买"
        signal_reason = "短期价格或题材热度已有较多兑现，不在当前价上方追价。"
        buy_high = min(close, support * (1.0 + max(buffer * 0.75, 0.01)))
    else:
        signal = "今天可分批"
        signal_reason = "当前没有明显追高或逻辑失效信号，以参考价附近分批观察。"

    # 目标一优先采用较近的行业估值锚或历史高点，目标二采用较远者。
    valuation_anchor = None
    eps_ttm = _number(row, "eps_ttm")
    pe_ttm = _number(row, "pe_ttm")
    peer_value = _number(row, "pe_industry_value")
    if (
        eps_ttm is not None
        and eps_ttm > 0
        and pe_ttm is not None
        and pe_ttm > 0
        and peer_value is not None
    ):
        peer_pe = pe_ttm * math.exp(_clamp(peer_value, -3.0, 3.0))
        candidate = eps_ttm * peer_pe * 0.90
        # 行业 PE 极端值可能把目标价推到数倍现价；超过两倍时不把它
        # 包装成精确目标，避免把异常估值直接展示成可执行价格。
        if math.isfinite(candidate) and close < candidate <= close * 2.0:
            valuation_anchor = candidate

    historical_high = None
    distance_high = _number(row, "dist_to_52w_high")
    if distance_high is not None and distance_high < 0 and 1.0 + distance_high > 0:
        candidate = close / (1.0 + distance_high)
        if math.isfinite(candidate) and close < candidate <= close * 3.0:
            historical_high = candidate

    targets = [
        value
        for value in (valuation_anchor, historical_high)
        if value is not None and value >= close * 1.02
    ]
    fallback_1 = close * (1.0 + max(0.04, buffer * 2.0))
    fallback_2 = close * (1.0 + max(0.08, buffer * 4.0))
    # 第一目标保持在较近的日线波动区间，较远的估值锚 / 历史高点只作为第二目标。
    target_1 = min([fallback_1, *targets])
    target_2 = min(max([fallback_2, *targets]), close * 2.0)
    if target_2 <= target_1:
        target_2 = target_1 * 1.05

    empty.update(
        {
            "operation_weekday": weekday,
            "operation_signal": signal,
            "reference_price": close,
            "buy_low": buy_low,
            "buy_high": max(buy_low, buy_high),
            "pullback_price": pullback_price,
            "no_chase_price": no_chase,
            "invalidation_price": invalidation,
            "valuation_anchor": valuation_anchor,
            "target_1": target_1,
            "target_2": target_2,
            "expectation_state": state,
            "operation_reason": f"{signal_reason} {state_reason}",
        }
    )
    return empty


def add_operation_plan(frame: pd.DataFrame, *, as_of: str | None = None) -> pd.DataFrame:
    """给已筛选出的推荐结果补充日线操作参考，不修改任何评分字段。"""

    out = frame.copy() if frame is not None else pd.DataFrame()
    if out.empty:
        for column in OPERATION_COLUMNS:
            out[column] = pd.Series(dtype="object")
        return out

    plans = pd.DataFrame(
        [_operation_row(row.to_dict(), as_of) for _, row in out.iterrows()],
        index=out.index,
    )
    for column in OPERATION_COLUMNS:
        out[column] = plans[column]
    return out


__all__ = ["OPERATION_COLUMNS", "add_operation_plan"]
