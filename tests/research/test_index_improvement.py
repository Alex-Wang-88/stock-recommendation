"""Check new-data feature causality, historical board rules and unchanged account arithmetic."""

# ruff: noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from research_resources import configure_background_job

configure_background_job()

import numpy as np
import pandas as pd
import pytest
from index_account_replay import replay_rules as corrected_replay
from index_improvement_data import historical_growth_flags, price_arrays
from index_validation_features import p4_features
from rule_account_replay import replay_rules as old_replay


def example_bars():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2017-01-01", periods=350).strftime("%Y-%m-%d").tolist()
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    raw = pd.DataFrame(
        {
            "close": close,
            "volume": rng.integers(1000, 2000, len(days)),
            "amount": rng.uniform(1e8, 2e8, len(days)),
            "isST": 0,
            "tradestatus": 1,
        },
        index=pd.Index(days, name="day"),
    )
    return raw, days


def test_p4_port_does_not_read_future_quotes():
    raw, days = example_bars()
    expected = p4_features(raw.iloc[:300], "2010-01-01", days[:300])
    changed = raw.copy()
    changed.loc[days[300] :, ["close", "volume", "amount"]] *= 1000
    pd.testing.assert_frame_equal(expected, p4_features(changed, "2010-01-01", days).iloc[:300])


def test_p4_risk_math_matches_independent_trailing_calculation():
    raw, days = example_bars()
    result = p4_features(raw, "2010-01-01", days).iloc[-1]
    ret = raw.close.pct_change().tail(20)
    assert np.isclose(result.volatility_value, -ret.std(ddof=0))
    assert np.isclose(result.downside_volatility_value, -ret[ret < 0].std(ddof=0))
    c = raw.close.tail(60).to_numpy()
    assert np.isclose(result.drawdown_value, np.min(c / np.maximum.accumulate(c) - 1))
    assert result.history_days == 250


def test_chinext_limit_changes_on_actual_effective_date():
    symbols = ["300001", "301001", "688001", "600000", "302132"]
    assert historical_growth_flags(symbols, "2019-01-02", "2019-02-20").tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert not historical_growth_flags(
        ["302132"], "2026-09-28", "2026-10-15", legacy_codes=True
    )[0]
    assert historical_growth_flags(symbols, "2020-09-01", "2020-10-20").tolist() == [
        True,
        True,
        True,
        False,
        True,
    ]
    with pytest.raises(ValueError):
        historical_growth_flags(symbols, "2020-08-01", "2020-09-01")


def test_missing_nullable_volume_is_non_executable_numeric_nan():
    prices = pd.DataFrame(
        {
            "day": ["prev", "a"],
            "symbol": ["X", "X"],
            **{field: [10.0, 10.0] for field in ["open", "high", "low", "close", "pre_close"]},
            "volume": pd.array([100, pd.NA], dtype="Int64"),
        }
    )
    status = prices[["day", "symbol"]].assign(tradestatus=1, isST=0)
    arrays, symbols, days = price_arrays(prices, status)
    assert arrays["volume"].dtype == np.dtype(float)
    assert np.isnan(arrays["volume"][days["a"], symbols["X"]])
    result = corrected_replay(
        {"days": ["a"], "signal_dates": ["prev"]},
        {"prev": {"pool": ["X"], "ranking": ["X"], "exposure": 1.0}},
        arrays,
        symbols,
        days,
        10,
        {"rebalance_days": 5, "positions": 1, "retain_rank": 1, "weighting": "equal"},
    )
    assert result[1] == [] and result[0][0] == 1.0


def test_growth_st_exit_uses_growth_board_limit():
    arrays = {k: np.array([[10.0], [10.0], [9.0]]) for k in ["open", "high", "low", "close"]}
    arrays.update(
        prior_close=np.full((3, 1), 10.0),
        volume=np.ones((3, 1)),
        tradable=np.ones((3, 1), dtype=bool),
        is_st=np.array([[False], [False], [True]]),
        growth_board=np.array([True]),
    )
    # A one-price -10% day is sellable on a 20%-limit growth board, including ST stocks.
    window = {"days": ["a", "b"], "signal_dates": ["prev"]}
    context = {"prev": {"pool": ["X"], "ranking": ["X"], "exposure": 1.0}}
    candidate = {"rebalance_days": 5, "positions": 1, "retain_rank": 1, "weighting": "equal"}
    corrected = corrected_replay(
        window, context, arrays, {"X": 0}, {"prev": 0, "a": 1, "b": 2}, 10, candidate
    )
    legacy = old_replay(
        window, context, arrays, {"X": 0}, {"prev": 0, "a": 1, "b": 2}, 10, candidate
    )
    assert corrected[3] == 0 and legacy[3] == 1
    arrays["is_st"][:] = False
    left = corrected_replay(
        window, context, arrays, {"X": 0}, {"prev": 0, "a": 1, "b": 2}, 10, candidate
    )
    right = old_replay(
        window, context, arrays, {"X": 0}, {"prev": 0, "a": 1, "b": 2}, 10, candidate
    )
    np.testing.assert_allclose(left[0], right[0], atol=0, rtol=0)


@pytest.mark.parametrize("last_day,residual", [("2026-07-03", 1), ("2026-07-06", 0)])
def test_main_board_st_limit_reform_effective_day(last_day, residual):
    arrays = {k: np.array([[10.0], [10.0], [9.3]]) for k in ["open", "high", "low", "close"]}
    arrays.update(
        prior_close=np.full((3, 1), 10.0),
        volume=np.ones((3, 1)),
        tradable=np.ones((3, 1), dtype=bool),
        is_st=np.array([[False], [False], [True]]),
        growth_board=np.array([False]),
    )
    window = {"days": ["2026-07-02", last_day], "signal_dates": ["2026-07-01"]}
    result = corrected_replay(
        window,
        {"2026-07-01": {"pool": ["X"], "ranking": ["X"], "exposure": 1.0}},
        arrays,
        {"X": 0},
        {"2026-07-01": 0, "2026-07-02": 1, last_day: 2},
        10,
        {"rebalance_days": 5, "positions": 1, "retain_rank": 1, "weighting": "equal"},
    )
    assert result[3] == residual
