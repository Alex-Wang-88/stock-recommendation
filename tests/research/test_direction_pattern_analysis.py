"""Ensure apparent patterns fail when one year or one large winner drives them."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from analyze_direction_study import paired_metrics


def experiment(effects):
    rows = []
    for i, effect in enumerate(effects):
        year = str(2024 + i // 5)
        for cost in [0, 10, 20]:
            for candidate, value in [("base", 0.0), ("alternative", effect)]:
                rows.append(
                    {
                        "window": f"w{i:02}",
                        "cost_bps": cost,
                        "candidate": candidate,
                        "return": value,
                        "start": year + "-01-01",
                        "year": year,
                        "realized_regime": "sideways",
                        "signal_regime": "above_ma60",
                    }
                )
    return paired_metrics(
        pd.DataFrame(rows),
        {"id": "test", "name": "test", "base": "base", "alternative": "alternative"},
        2 / 3,
    )[0]


def test_two_thirds_of_windows_is_not_enough_when_one_year_reverses_the_effect():
    result = experiment([0.01] * 10 + [-0.001] * 5)
    assert result["mean_effect"] > 0 and result["positive_windows"] == 10
    assert result["status"] == "mixed"


def test_one_large_win_cannot_make_an_outlier_dependent_change_consistent():
    result = experiment(
        [0.2, 0.001, 0.001, -0.05, -0.05] + [0.001, 0.001, 0.001, 0.001, -0.002] * 2
    )
    assert result["mean_effect"] > 0 and result["minimum_year_mean_effect"] > 0
    assert result["positive_windows"] == 11
    assert result["effect_without_best_window"] < 0
    assert result["status"] == "mixed"


def test_repeated_cross_year_improvement_and_deterioration_are_reported_symmetrically():
    effects = [0.01, 0.01, 0.01, 0.01, -0.001] * 3
    assert experiment(effects)["status"] == "consistent_positive"
    assert experiment([-value for value in effects])["status"] == "consistent_negative"
