"""因子层测试 —— 重点在「缺失值不能被算成 0」和「不满窗不给位置」。"""

from __future__ import annotations

import pandas as pd
import pytest

from recommend.factors import build_factor_table
from recommend.factors.momentum import add_momentum_factors
from recommend.factors.price import add_price_factors
from recommend.factors.theme import add_theme_factors, matches_themes


def test_position_requires_full_window(bars_factory):
    """历史不足 250 根 ⇒ `close_position_250` 必须是 NaN，不能拿 30 天冒充一年。"""

    short = bars_factory(n_days=30, symbols=("000001",))
    out = add_price_factors(short)
    assert out["close_position_20"].notna().all()
    assert out["close_position_250"].isna().all()
    assert out["dist_to_52w_high"].isna().all()
    assert out["history_days"].iloc[0] == 30


def test_position_and_returns_are_computed_correctly(bars_factory):
    # 收盘从 10 涨到 10 + 299*0.01 = 12.99，单调上行 ⇒ 位置应为 1。
    bars = bars_factory(n_days=300, symbols=("000001",))
    bases = pd.DataFrame({"symbol": ["000001"], "base_924": [10.0], "base_ytd": [11.0]})
    out = add_price_factors(bars, bases)
    row = out.iloc[0]
    assert row["close_position_250"] == pytest.approx(1.0)
    assert row["close_position_20"] == pytest.approx(1.0)
    assert row["dist_to_52w_high"] == pytest.approx(0.0)
    assert row["ret_since_924"] == pytest.approx(12.99 / 10.0 - 1.0)
    assert row["ret_ytd"] == pytest.approx(12.99 / 11.0 - 1.0)


def test_missing_base_gives_nan_not_zero(bars_factory):
    """算不出基准涨幅时必须是 NaN —— 填 0 会被读成「924 以来没涨」。"""

    bars = bars_factory(n_days=300, symbols=("000001",))
    out = add_price_factors(bars, pd.DataFrame(columns=["symbol"]))
    assert out["ret_since_924"].isna().all()
    assert out["ret_ytd"].isna().all()


def test_flat_price_range_gives_nan_position_not_inf(bars_factory):
    """一字板 / 长期停牌（区间上下沿相等）时分母为 0 ⇒ 位置应为 NaN，不能是 inf。

    注意 `dist_to_52w_high` 不在此列：价格贴着区间上沿时它的正确值是 **0**（距高点 0%），
    这与「算不出来」完全不同，不该被填成 NaN。
    """

    bars = bars_factory(n_days=300, symbols=("000001",), close_for=lambda s, d: 10.0)
    out = add_price_factors(bars)
    row = out.iloc[0]
    for factor in ("close_position_20", "close_position_60", "close_position_250"):
        assert pd.isna(row[factor]), factor
    assert row["dist_to_52w_high"] == pytest.approx(0.0)


def test_volume_trend_is_short_over_long(bars_factory):
    bars = bars_factory(n_days=60, symbols=("000001",))
    out = add_momentum_factors(bars, entry_window=5, base_window=20)
    # 成交量恒定 ⇒ 量比应为 1。
    assert out["volume_trend_20"].iloc[0] == pytest.approx(1.0)


def test_ret_requires_enough_history(bars_factory):
    short = bars_factory(n_days=30, symbols=("000001",))
    out = add_momentum_factors(short)
    assert out["ret_20"].notna().all()
    assert out["ret_60"].isna().all()


def test_relative_returns_compare_each_stock_with_local_equal_weight_market(bars_factory):
    def close_for(symbol_index: int, day_index: int) -> float:
        slope = 0.01 if symbol_index == 0 else 0.03
        return 10.0 + slope * day_index

    out = add_momentum_factors(
        bars_factory(n_days=100, symbols=("000001", "000002"), close_for=close_for)
    ).set_index("symbol")

    assert out.loc["000002", "relative_ret_20"] > out.loc["000001", "relative_ret_20"]
    assert out.loc["000002", "relative_ret_60"] > out.loc["000001", "relative_ret_60"]


def test_theme_heat_is_nan_when_archive_day_is_empty():
    """归档当天为空 ⇒ 热度必须是 NaN。

    若填 0，则「归档没回补」会伪装成「今天所有股票都没题材热度」——
    依赖 `theme_heat_max >= 1` 的条件会静默失效，而结果看起来完全正常。
    """

    universe = pd.DataFrame({"symbol": ["000001", "000002"]})
    out = add_theme_factors(universe, pd.DataFrame(), pd.DataFrame())
    assert out["theme_heat_max"].isna().all()
    assert out["theme_tag_count"].isna().all()


def test_theme_absent_from_list_is_zero_when_archive_exists():
    attribution = pd.DataFrame({"symbol": ["000001"], "reason": ["算力+CPO"]})
    tag_daily = pd.DataFrame({"tag": ["算力", "CPO"], "stock_count": [7, 3]})
    universe = pd.DataFrame({"symbol": ["000001", "000002"]})
    out = add_theme_factors(universe, attribution, tag_daily).set_index("symbol")
    assert out.loc["000001", "theme_heat_max"] == 7
    assert out.loc["000001", "theme_heat_sum"] == 10
    assert out.loc["000001", "theme_tags"] == "算力|CPO"
    # 名单非空 ⇒ 不在名单里是真的「今天没有题材热度」。
    assert out.loc["000002", "theme_heat_max"] == 0


def test_theme_days_is_zero_when_history_available():
    attribution = pd.DataFrame({"symbol": ["000001"], "reason": ["算力"]})
    history = pd.DataFrame(
        {"symbol": ["000001", "000001"], "trade_date": pd.to_datetime(["2026-09-15", "2026-09-16"])}
    )
    universe = pd.DataFrame({"symbol": ["000001", "000002"]})
    out = add_theme_factors(universe, attribution, pd.DataFrame(), history).set_index("symbol")
    assert out.loc["000001", "theme_days_20"] == 2
    assert out.loc["000002", "theme_days_20"] == 0


def test_matches_themes_is_exact_not_substring():
    """「算力」不应命中「算力租赁」—— 整标签精确匹配，宁可漏不可错配。"""

    assert matches_themes("算力|CPO", ("算力",))
    assert not matches_themes("算力租赁", ("算力",))
    assert matches_themes("算力租赁", ("算力租赁",))
    assert not matches_themes(None, ("算力",))
    assert matches_themes(None, ())


def test_build_factor_table_joins_everything(bars_factory):
    bars = bars_factory(n_days=300, symbols=("000001", "000002"))
    universe = pd.DataFrame({"symbol": ["000001", "000002"]})
    table = build_factor_table(
        universe=universe,
        bars=bars,
        base_closes=pd.DataFrame({"symbol": ["000001", "000002"], "base_924": [10.0, 10.0],
                                  "base_ytd": [10.0, 10.0]}),
        attribution=pd.DataFrame({"symbol": ["000001"], "reason": ["算力"]}),
        tag_daily=pd.DataFrame({"tag": ["算力"], "stock_count": [5]}),
        as_of="2026-01-01",
    )
    assert len(table.frame) == 2
    assert (table.frame["trade_date"] == "2026-01-01").all()
    # 类别登记表必须能认出位置因子，排序层靠它拦人。
    assert table.categories()["close_position_250"] == "position"
    assert table.categories()["ret_20"] == "momentum"
