"""筛选引擎测试 —— 重点是漏斗账目必须对得上、缺失必须与越界分开计。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommend.factors import FactorTable
from recommend.screen.engine import run_screen
from recommend.screen.spec import Spec

WEIGHTS = {"momentum": 0.35, "theme": 0.30, "capital": 0.20, "liquidity": 0.15}


def _table(**overrides) -> FactorTable:
    base = {
        "symbol": ["000001", "000002", "000003", "600001"],
        "name": ["平安银行", "万科A", "*ST海航", "邯郸钢铁"],
        "close": [10.0, 20.0, 3.0, 5.0],
        "amount": [1e8, 1e8, 1e8, 1e8],
        "history_days": [300, 300, 300, 300],
        "listed_days": [4000, 4000, 4000, 4000],
        "close_position_250": [0.20, 0.80, 0.10, np.nan],
        "dist_to_52w_high": [-0.4, -0.1, -0.5, np.nan],
        "close_position_20": [0.2, 0.8, 0.1, 0.5],
        "ret_since_924": [0.1, 0.2, 0.3, 0.4],
        "ret_ytd": [0.0, 0.1, -0.1, 0.05],
        "ret_20": [0.30, 0.10, 0.05, -0.05],
        "ret_60": [0.4, 0.2, 0.1, -0.1],
        "ret_rank_20": [1.0, 0.75, 0.5, 0.25],
        "volume_trend_20": [2.0, 1.5, 1.2, 0.8],
        "turnover_value_20": [1e9, 8e8, 5e8, 1e8],
        "amount_avg_20": [1e9, 8e8, 5e8, 1e8],
        "theme_tags": ["算力|CPO", None, "算力租赁", None],
        "theme_tag_count": [2, 0, 1, 0],
        "theme_heat_max": [10, 0, 3, 0],
        "theme_heat_sum": [15, 0, 3, 0],
        "theme_days_20": [8, 0, 1, 0],
        "lhb_count_20": [1, 0, 0, 0],
        "lhb_net_buy_20": [1e7, np.nan, np.nan, np.nan],
        "lhb_net_buy_rel": [0.02, np.nan, np.nan, np.nan],
        "main_net_inflow": [1e8, np.nan, np.nan, np.nan],
        "margin_rz_chg_20": [np.nan] * 4,
        "ma20_dev": [0.05, 0.02, -0.01, 0.0],
        "ma60_dev": [0.1, 0.05, -0.05, 0.0],
    }
    base.update(overrides)
    return FactorTable(frame=pd.DataFrame(base), as_of="2026-09-16")


def test_st_is_excluded():
    result = run_screen(_table(), Spec.from_dict({}), WEIGHTS)
    assert "000003" not in set(result.frame["symbol"])
    assert result.rejections["ST/退市名称"] == 1


def test_missing_values_are_counted_separately_from_out_of_range():
    """缺失必须单独计数。

    如果 NaN 被并进「不在区间」，就分不清「筛掉了」和「没数据没法筛」——
    归档断档时使用者会以为条件生效了。
    """

    result = run_screen(_table(), Spec.from_dict({"close_position_250": [None, 0.5]}), WEIGHTS)
    assert result.rejections["close_position_250:缺失"] == 1
    # 0.80 越界；0.20 / 0.10 通过；600001 缺失。
    assert result.rejections["close_position_250 不在区间"] == 1


def test_funnel_accounting_adds_up():
    """universe − Σ剔除 − 截断 = 保留。账对不上就说明有静默丢行。"""

    result = run_screen(
        _table(),
        Spec.from_dict({"close_position_250": [None, 0.9], "volume_trend_20": [1.0, None]}),
        WEIGHTS,
    )
    removed = sum(result.rejections.values())
    assert result.universe_size - removed - result.truncated == result.kept


def test_truncation_is_disclosed_not_silent():
    result = run_screen(_table(), Spec.from_dict({"limit": 2}), WEIGHTS)
    assert result.kept == 2
    assert result.pool_size == 3  # 4 只减去 1 只 ST
    assert result.truncated == 1


def test_missing_factor_column_empties_result_instead_of_ignoring_condition():
    """因子列不存在 ⇒ 无结果 + 明确标注，**不是**忽略该条件。"""

    table = _table()
    table.frame = table.frame.drop(columns=["theme_heat_max"])
    result = run_screen(table, Spec.from_dict({"theme_heat_max": [1, None]}), WEIGHTS)
    assert result.kept == 0
    assert any("列不存在" in reason for reason in result.rejections)
    assert any("无法评估" in note for note in result.notes)


def test_theme_match_is_exact():
    """「算力」命中 000001，但不命中 000003（算力租赁）。"""

    result = run_screen(_table(), Spec.from_dict({"themes": ["算力"]}), WEIGHTS)
    assert set(result.frame["symbol"]) == {"000001"}


def test_exclude_themes():
    result = run_screen(_table(), Spec.from_dict({"exclude_themes": ["算力"]}), WEIGHTS)
    assert "000001" not in set(result.frame["symbol"])


def test_st_filter_can_be_turned_off():
    result = run_screen(_table(), Spec.from_dict({"exclude_st": False}), WEIGHTS)
    assert "000003" in set(result.frame["symbol"])


def test_missing_names_are_disclosed():
    table = _table()
    table.frame = table.frame.drop(columns=["name"])
    result = run_screen(table, Spec.from_dict({}), WEIGHTS)
    assert any("未被剔除" in note for note in result.notes)


def test_sort_by_position_factor_is_impossible_via_spec():
    from recommend.screen.spec import SpecError

    with pytest.raises(SpecError):
        Spec.from_dict({"sort": "close_position_250"})


def test_sort_by_plain_factor_works():
    result = run_screen(_table(), Spec.from_dict({"sort": "ret_20", "limit": 10}), WEIGHTS)
    assert result.frame["ret_20"].is_monotonic_decreasing


def test_limit_one_and_descending_false():
    spec = Spec.from_dict({"sort": "ret_20", "descending": False, "limit": 1})
    result = run_screen(_table(), spec, WEIGHTS)
    assert result.kept == 1
    assert result.frame["symbol"].iloc[0] == "600001"


def test_empty_table_is_handled():
    table = FactorTable(frame=pd.DataFrame(columns=["symbol", "name"]), as_of="2026-09-16")
    result = run_screen(table, Spec.from_dict({}), WEIGHTS)
    assert result.kept == 0
