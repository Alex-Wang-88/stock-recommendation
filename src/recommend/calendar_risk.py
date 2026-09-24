"""Market-wide pre-holiday cash-planning notices.

The windows below are the final two *exchange trading sessions* before the
published mainland China exchange closures. They are informational only: the
cash-withdrawal mechanism does not establish that prices will fall, so this
module never changes filters, ranks, or prices.

Sources: official SSE/SZSE trading-closure notices for 2024, 2025, and 2026:
- https://www.sse.com.cn/disclosure/announcement/general/c/c_20231226_5733939.shtml
- https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/announcement/general/c/c_20241223_10767108.shtml
- https://www.szse.cn/disclosure/notice/t20241223_611283.html
- https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml
- https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml
- https://www.szse.cn/www/disclosure/notice/general/t20260917_622911.html

Update this table after the exchanges publish a new year's schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PreHolidayWindow:
    holiday: str
    sessions: tuple[str, str]

    def position(self, as_of: str) -> str:
        if as_of == self.sessions[0]:
            return "倒数第二个交易日"
        return "最后一个交易日"

    def report_note(self, as_of: str) -> str:
        return (
            f"节前资金安排提示：当前是{self.holiday}前{self.position(as_of)}。"
            "A股卖出资金通常当日可继续交易、下一交易日才可转出，假期前可能有现金安排压力；"
            "这并不等于市场必跌。本提示尚未作为下跌信号回测验证，只作市场级提醒，"
            "不改变筛选条件、个股排序或操作价格。"
        )

    def operation_note(self, as_of: str) -> str:
        return (
            f"节前提醒：{self.holiday}前{self.position(as_of)}；"
            "若假期需要银行可取资金，请留意卖出款通常要到下一交易日才能转出。"
            "这不是必跌信号，也不改变个股分数。"
        )


# Each pair is (the session two trading days before closure, the final
# trading session before closure). Dates come from official SSE/SZSE closures;
# storing sessions explicitly handles makeup workdays and shifted weekends.
PRE_HOLIDAY_WINDOWS: tuple[PreHolidayWindow, ...] = (
    PreHolidayWindow("2024年中秋节", ("2024-09-12", "2024-09-13")),
    PreHolidayWindow("2024年国庆节", ("2024-09-27", "2024-09-30")),
    PreHolidayWindow("2025年元旦", ("2024-12-30", "2024-12-31")),
    PreHolidayWindow("2025年春节", ("2025-01-24", "2025-01-27")),
    PreHolidayWindow("2025年清明节", ("2025-04-02", "2025-04-03")),
    PreHolidayWindow("2025年劳动节", ("2025-04-29", "2025-04-30")),
    PreHolidayWindow("2025年端午节", ("2025-05-29", "2025-05-30")),
    PreHolidayWindow("2025年中秋节、国庆节", ("2025-09-29", "2025-09-30")),
    PreHolidayWindow("2026年元旦", ("2025-12-30", "2025-12-31")),
    PreHolidayWindow("2026年春节", ("2026-02-12", "2026-02-13")),
    PreHolidayWindow("2026年清明节", ("2026-04-02", "2026-04-03")),
    PreHolidayWindow("2026年劳动节", ("2026-04-29", "2026-04-30")),
    PreHolidayWindow("2026年端午节", ("2026-06-17", "2026-06-18")),
    PreHolidayWindow("2026年中秋节", ("2026-09-23", "2026-09-24")),
    PreHolidayWindow("2026年国庆节", ("2026-09-29", "2026-09-30")),
)

CALENDAR_COVERAGE_START = "2024-09-02"
CALENDAR_COVERAGE_END = "2026-12-31"


def preholiday_window(as_of: str | None) -> PreHolidayWindow | None:
    """Return the published holiday window containing ``as_of``, if any."""

    day = _iso_day(as_of)
    if day is None:
        return None
    return next(
        (window for window in PRE_HOLIDAY_WINDOWS if day in window.sessions), None
    )


def calendar_coverage_note(as_of: str | None) -> str | None:
    """Warn when an as-of date is outside the currently maintained schedule."""

    day = _iso_day(as_of)
    if day is None:
        return None
    if CALENDAR_COVERAGE_START <= day <= CALENDAR_COVERAGE_END:
        return None
    return (
        f"节假日日历当前覆盖 {CALENDAR_COVERAGE_START} 至 {CALENDAR_COVERAGE_END}；"
        f"{day} 未判定节前窗口，不代表没有节假日风险。"
        "请在交易所发布新年度休市安排后更新日历。"
    )


def _iso_day(as_of: str | None) -> str | None:
    if not as_of:
        return None
    try:
        return date.fromisoformat(str(as_of)[:10]).isoformat()
    except (TypeError, ValueError):
        return None


__all__ = [
    "CALENDAR_COVERAGE_END",
    "CALENDAR_COVERAGE_START",
    "PRE_HOLIDAY_WINDOWS",
    "PreHolidayWindow",
    "calendar_coverage_note",
    "preholiday_window",
]
