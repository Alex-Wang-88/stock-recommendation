"""输出层测试 —— 重点是「缺失不能被渲染成看似真实的值」。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommend.factors import FactorTable
from recommend.report.render import markdown_table, render_markdown, to_display
from recommend.screen.engine import run_screen
from recommend.screen.spec import Spec

WEIGHTS = {"momentum": 0.35, "theme": 0.30, "capital": 0.20, "liquidity": 0.15}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "name": ["平安银行", "万科A"],
            "close": [10.0, 20.5],
            "score": [0.87654, 0.5],
            "score_coverage": [0.8, 1.0],
            "ret_since_924": [0.3781, -0.05],
            "ret_ytd": [-0.391, 0.0],
            "close_position_250": [0.24809160305343508, np.nan],
            "dist_to_52w_high": [-0.478, -0.1],
            "ret_20": [0.361, 0.1],
            "volume_trend_20": [2.4651, 1.0],
            "turnover_value_20": [5.33e8, 1.2e9],
            "theme_heat_max": [1, np.nan],
            "theme_days_20": [4, np.nan],
            "theme_tags": ["北斗导航|商业航天|农机", np.nan],
            "lhb_count_20": [np.nan, np.nan],
            "lhb_net_buy_rel": [np.nan, np.nan],
            "main_net_inflow": [np.nan, np.nan],
            "theme_tag_count": [3, 0],
            "theme_heat_sum": [5, 0],
        }
    )


def test_missing_theme_tags_do_not_render_as_the_literal_nan():
    """缺失标签必须显示成空。

    渲染成 `nan` 会被读成一个真实的题材名 —— 比空白危险得多。
    （这个 bug 真出现过：pandas 3.x 的字符串列是 StringDtype，
     用 `dtype == object` 判断文本列会漏掉它们。）
    """

    view = to_display(_frame())
    assert view.loc[1, "theme_tags"] == ""
    assert "nan" not in view.loc[1, "theme_tags"].lower()
    assert view.loc[1, "theme_heat_max"] == ""
    assert view.loc[1, "lhb_count_20"] == ""


def test_theme_tags_pipe_becomes_ideographic_comma():
    """`|` 是 Markdown 表格分隔符，必须在单元格里换掉，否则一个标签被拆成两列。"""

    view = to_display(_frame())
    assert view.loc[0, "theme_tags"] == "北斗导航、商业航天、农机"
    assert view.loc[0, "theme_tags"].count("|") == 0


def test_floats_are_rounded_for_display():
    view = to_display(_frame())
    assert view.loc[0, "close_position_250"] == "0.25"
    assert view.loc[0, "score"] == "0.877"
    assert view.loc[0, "ret_since_924"] == "37.8%"
    assert view.loc[0, "turnover_value_20"] == "5.33亿"


def test_markdown_table_column_count_is_stable():
    """每行列数必须与表头一致 —— 这是表格能否渲染的硬条件。"""

    view = to_display(_frame())
    text = markdown_table(view)
    lines = [line for line in text.splitlines() if line.startswith("|")]
    widths = {line.count("|") for line in lines}
    assert len(widths) == 1, widths


def _result(**payload):
    table = FactorTable(frame=_frame(), as_of="2026-09-16")
    return run_screen(table, Spec.from_dict(payload), WEIGHTS)


def test_render_states_the_data_as_of():
    """没有数据时点，读者无法判断这是不是拿陈旧数据当今天。"""

    text = render_markdown(_result())
    assert "## 数据时点" in text
    assert "2026-09-16" in text


def test_render_states_that_position_factors_are_excluded_from_ranking():
    text = render_markdown(_result())
    assert "不参与排序" in text


def test_render_discloses_truncation():
    text = render_markdown(_result(limit=1))
    assert "未显示 ≠ 未通过" in text


def test_render_lists_only_factors_that_actually_contributed():
    """理由里不能把缺失的因子说成参与了打分。"""

    text = render_markdown(_result())
    assert "数据缺失、未参与的因子" in text
    assert "龙虎榜净买强度" in text  # 应出现在「未参与」那一行里


def test_render_handles_empty_result():
    text = render_markdown(_result(close_position_250=[2.0, 3.0]))
    assert "无命中" in text
