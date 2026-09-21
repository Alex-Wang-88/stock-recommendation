"""日线操作参考卡测试：不改变排名，只生成可解释的快照字段。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommend.operation import add_operation_plan


def _row(**overrides) -> pd.DataFrame:
    values = {
        "symbol": "000001",
        "close": 10.0,
        "score": 0.8,
        "score_coverage": 1.0,
        "ret_20": 0.02,
        "relative_ret_20": 0.01,
        "close_position_20": 0.45,
        "dist_to_52w_high": -0.25,
        "volume_trend_20": 1.0,
        "theme_days_20": 1,
        "realized_vol_20": 0.02,
        "ma20_dev": -0.01,
        "ma60_dev": -0.04,
        "eps_ttm": 0.8,
        "pe_ttm": 12.0,
        "pe_industry_value": 0.1,
        "yoy_net_income": 0.12,
        "revenue_yoy": 0.10,
    }
    values.update(overrides)
    return pd.DataFrame([values])


def test_daily_plan_uses_view_date_and_has_reference_prices():
    out = add_operation_plan(_row(), as_of="2026-09-21")
    row = out.iloc[0]

    assert row["operation_weekday"] == "周一"
    assert row["operation_signal"] == "今天可分批"
    assert row["reference_price"] == 10.0
    assert row["buy_low"] <= row["buy_high"]
    assert row["invalidation_price"] < row["buy_low"]
    assert row["target_1"] > row["reference_price"]
    assert row["target_2"] >= row["target_1"]
    assert row["expectation_state"] == "正在兑现"


def test_overextended_daily_plan_waits_for_pullback():
    out = add_operation_plan(
        _row(
            ret_20=0.20,
            relative_ret_20=0.12,
            close_position_20=0.95,
            dist_to_52w_high=-0.02,
            volume_trend_20=1.7,
            theme_days_20=5,
            ma20_dev=0.10,
        ),
        as_of="2026-09-21",
    )
    row = out.iloc[0]

    assert row["operation_signal"] == "等回落再买"
    assert row["expectation_state"] == "短期兑现较充分"
    assert row["buy_high"] < row["reference_price"]


def test_missing_close_does_not_fabricate_prices():
    out = add_operation_plan(_row(close=np.nan), as_of="2026-09-21")
    row = out.iloc[0]

    assert row["operation_signal"] == "暂缓"
    assert pd.isna(row["reference_price"])
    assert pd.isna(row["target_1"])


def test_operation_plan_does_not_change_existing_columns():
    frame = _row()
    out = add_operation_plan(frame, as_of="2026-09-21")

    for column in frame.columns:
        pd.testing.assert_series_equal(out[column], frame[column], check_names=False)
