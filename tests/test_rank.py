"""排序层测试 —— 重点是「位置因子进不了权重」这条机械保证。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommend.rank.score import RankWeightError, add_scores, explain


def _frame(**overrides):
    base = {
        "symbol": ["a", "b", "c", "d"],
        "ret_20": [0.30, 0.10, 0.05, -0.05],
        "volume_trend_20": [2.0, 1.5, 1.2, 0.8],
        "theme_heat_max": [10, 5, 1, 0],
        "theme_days_20": [8, 4, 2, 0],
        "lhb_net_buy_rel": [0.02, 0.01, np.nan, np.nan],
        "main_net_inflow": [1e8, 5e7, np.nan, np.nan],
        "turnover_value_20": [1e9, 8e8, 5e8, 1e8],
    }
    base.update(overrides)
    return pd.DataFrame(base)


WEIGHTS = {"momentum": 0.35, "theme": 0.30, "capital": 0.20, "liquidity": 0.15}


def test_position_factor_is_forbidden_in_weights():
    """「低位只做筛选、不做排序加分」—— 写进权重就直接报错。"""

    with pytest.raises(RankWeightError, match="位置因子禁止进入排序权重"):
        add_scores(_frame(), {**WEIGHTS, "close_position_250": 0.5})


def test_unknown_weight_key_is_rejected():
    with pytest.raises(RankWeightError, match="未知权重键"):
        add_scores(_frame(), {"magic": 1.0})


def test_negative_weight_is_rejected():
    with pytest.raises(RankWeightError, match="非负"):
        add_scores(_frame(), {"momentum": -1.0})


def test_all_zero_weights_are_rejected():
    with pytest.raises(RankWeightError, match="必须大于 0"):
        add_scores(_frame(), {"momentum": 0.0})


def test_scores_are_bounded_and_ordered():
    scored = add_scores(_frame(), WEIGHTS)
    assert scored["score"].between(0, 1).all()
    # a 在动量、题材、流动性上都最好 ⇒ 分数应最高。
    assert scored["score"].idxmax() == 0


def test_since_924_return_is_a_momentum_member():
    frame = _frame(
        ret_20=[0.0, 0.0, 0.0, 0.0],
        volume_trend_20=[1.0, 1.0, 1.0, 1.0],
        ret_since_924=[0.1, 0.2, 0.3, 0.4],
    )
    scored = add_scores(frame, {"momentum": 1.0})
    assert scored["score_momentum"].idxmax() == 3


def test_missing_category_renormalizes_and_discloses_coverage():
    """资本类整列缺失时，权重在剩余类别间重新归一，并披露覆盖率。

    关键：**不能**把缺失当 0 —— 那会把「没数据」当成「资金面差」来惩罚。
    """

    frame = _frame(lhb_net_buy_rel=[np.nan] * 4, main_net_inflow=[np.nan] * 4)
    scored = add_scores(frame, WEIGHTS)
    assert scored["score"].notna().all()
    assert scored["score_coverage"].iloc[0] == pytest.approx(0.80)
    assert scored["score_capital"].isna().all()
    # 若把缺失当 0，a 的分数会被资本项压低；重归一后 a 仍应领先 b。
    assert scored["score"].iloc[0] > scored["score"].iloc[1]


def test_coverage_is_one_when_everything_present():
    scored = add_scores(_frame(), WEIGHTS)
    assert scored["score_coverage"].iloc[0] == pytest.approx(1.0)


def test_score_is_nan_when_all_categories_missing():
    empty = pd.DataFrame({"symbol": ["a"], "ret_20": [np.nan], "volume_trend_20": [np.nan],
                          "theme_heat_max": [np.nan], "theme_days_20": [np.nan],
                          "lhb_net_buy_rel": [np.nan], "main_net_inflow": [np.nan],
                          "turnover_value_20": [np.nan]})
    scored = add_scores(empty, WEIGHTS)
    assert scored["score"].isna().all()


def test_empty_frame_is_handled():
    out = add_scores(pd.DataFrame(), WEIGHTS)
    assert "score" in out.columns


def test_explain_lists_positional_factors_as_excluded():
    scored = add_scores(_frame(), WEIGHTS)
    lines = explain(scored.iloc[0].to_dict(), WEIGHTS)
    text = "\n".join(lines)
    assert "momentum" in text
    assert "不参与排序" in text
