"""Protect the public-formula port and chronological learning boundaries."""

# ruff: noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from research_resources import configure_background_job

configure_background_job()

import numpy as np
import pandas as pd
from qlib_alpha_features import (
    alpha157,
    fit_transformer,
    purged_fold_masks,
    regression,
    transform,
)


def bars():
    rng = np.random.default_rng(42)
    close = 30 * np.exp(np.cumsum(rng.normal(0, 0.02, 130)))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.97,
            "close": close,
            "volume": rng.integers(1000, 3000, len(close)),
        }
    )


def test_features_do_not_read_future_bars():
    frame = bars()
    expected = alpha157(frame.iloc[:90])
    changed = frame.copy()
    changed.loc[90:, ["open", "high", "low", "close", "volume"]] *= 100
    pd.testing.assert_frame_equal(expected, alpha157(changed).iloc[:90])


def test_feature_count_and_no_financial_dependencies():
    frame = bars()
    original = alpha157(frame)
    frame["industry"] = "FUTURE_INDUSTRY"
    frame["roe"] = 1e20
    frame["pe"] = -1e20
    pd.testing.assert_frame_equal(original, alpha157(frame))
    assert original.shape[1] == 157
    assert "VWAP0" not in original
    assert np.isfinite(original.iloc[65:].to_numpy()).all()


def test_sample_std_rank_and_extreme_position_match_formulas():
    frame = bars()
    frame.loc[75:79, "close"] = [1, 2, 2, 4, 3]
    frame.loc[75:79, "high"] = [10, 40, 30, 20, 15]
    result = alpha157(frame)
    assert np.isclose(result.loc[79, "STD5"], np.std([1, 2, 2, 4, 3], ddof=1) / 3)
    assert result.loc[79, "RANK5"] == 0.8
    assert result.loc[79, "IMAX5"] == 2 / 5
    assert np.isclose(result.loc[79, "ROC5"], frame.loc[74, "close"] / 3)


def test_regression_against_independent_polynomial_fit():
    series = pd.Series([5.0, 8, 7, 12, 9, 11])
    slope, r2, residual = regression(series, 5)
    x, y = np.arange(5), series.iloc[-5:].to_numpy()
    beta, intercept = np.polyfit(x, y, 1)
    prediction = beta * x + intercept
    assert np.isclose(slope[-1], beta)
    assert np.isclose(residual[-1], y[-1] - prediction[-1])
    assert np.isclose(r2[-1], 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def test_labels_crossing_segment_boundaries_are_excluded():
    days = pd.Series(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    ends = pd.Series(["2024-01-09", "2024-02-01", "2024-02-08", "2024-03-01"])
    fold = {
        "train_start": "2024-01-01",
        "train_end": "2024-01-31",
        "valid_start": "2024-02-01",
        "valid_end": "2024-02-28",
        "test_first_signal": "2024-03-01",
    }
    train, valid = purged_fold_masks(days, ends, fold)
    assert train.tolist() == [True, False, False, False]
    assert valid.tolist() == [False, False, True, False]


def test_preprocessing_is_fit_on_training_only_and_handles_missing():
    train = np.array([[1.0, np.nan], [2, 0], [3, 0]])
    medians, scale = fit_transformer(train)
    before = transform(train, medians, scale)
    transform(np.array([[1e12, -1e12]]), medians, scale)
    np.testing.assert_array_equal(before, transform(train, medians, scale))
    assert np.isfinite(before).all()
    assert np.max(np.abs(before)) <= 3
