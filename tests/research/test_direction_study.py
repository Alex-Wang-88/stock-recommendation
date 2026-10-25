"""Check signal-time causality, reversal direction and common-universe isolation."""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from direction_scoring import direction_ranking
from prepare_direction_data import lag_features


def score_frame():
    return pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "ret_5": [0.1, -0.1],
            "ret_10": [0.05, -0.05],
            "ret_20": [0.1, -0.1],
            "ret_60": [0.2, -0.2],
            "ret_60_skip5": [0.2, -0.2],
            "volume_trend_20": [1.2, 1.2],
            "turnover_value_20": [100_000_000.0, 100_000_000.0],
            "volatility_value": [-0.01, -0.01],
            "drawdown_value": [-0.05, -0.05],
            "downside_volatility_value": [-0.01, -0.01],
        }
    )


def test_past_losses_rank_first_for_reversal_and_past_gains_for_trend():
    frame = score_frame()
    assert direction_ranking(frame, "reverse5", True).symbol.iloc[0] == "000002"
    assert direction_ranking(frame, "trend5", True).symbol.iloc[0] == "000001"


def test_state_switch_responds_to_the_supplied_known_market_state():
    frame = score_frame()
    assert direction_ranking(frame, "regime_switch", True).symbol.iloc[0] == "000001"
    assert direction_ranking(frame, "regime_switch", False).symbol.iloc[0] == "000002"


def test_financial_columns_and_row_order_cannot_change_direction_scores():
    frame = score_frame()
    polluted = frame.iloc[::-1].copy()
    polluted["industry_score"] = [-1e99, 1e99]
    polluted["pe_ttm"] = [-1e99, 1e99]
    left = direction_ranking(frame, "reverse_risk", True).reset_index(drop=True)
    right = direction_ranking(polluted, "reverse_risk", True).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def test_missing_common_factor_is_rejected_even_when_not_used_by_this_score():
    frame = score_frame()
    frame.loc[0, "ret_20"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        direction_ranking(frame, "reverse5", True)


def test_future_quotes_do_not_change_short_return_or_skip_week_signal():
    dates = pd.date_range("2023-01-01", periods=90)
    values = np.arange(1.0, 91.0)
    bars = pd.DataFrame({"symbol": ["000001"] * 90, "trade_date": dates, "close": values})
    con = duckdb.connect(config={"threads": 1})
    con.register("sample_prices", bars)
    original = lag_features(con, "sample_prices", [str(dates[70].date())])
    future_changed = bars.copy()
    future_changed.loc[future_changed.trade_date > dates[70], "close"] *= 1000
    con.register("sample_prices", future_changed)
    altered = lag_features(con, "sample_prices", [str(dates[70].date())])
    pd.testing.assert_frame_equal(original, altered, check_exact=True)
    assert original.ret_5.iloc[0] == pytest.approx(71 / 66 - 1)
    assert original.ret_10.iloc[0] == pytest.approx(71 / 61 - 1)
    assert original.ret_60_skip5.iloc[0] == pytest.approx(66 / 11 - 1)
    con.close()
