"""Lock a finite structural study and untouched historical stress dates before replay."""

from __future__ import annotations

import json
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/index-improvement-20261016"
PREVIOUS = ROOT / "logs/qlib-comparison-20261015"


def main():
    from prepare_market_weight_batch import sha256

    OUT.mkdir(exist_ok=True)
    previous = json.loads((PREVIOUS / "plan_locked.json").read_text(encoding="utf-8"))
    families = [
        ("F0", "legacy", 1.1, 5, 25, 40),
        ("F1", "remove_volume_gate", 0.0, 5, 25, 40),
        ("F2", "hold_10_days", 0.0, 10, 25, 40),
        ("F3", "spread_50", 0.0, 10, 50, 75),
        ("F4", "hold_20_days", 0.0, 20, 50, 75),
        ("F5", "spread_100", 0.0, 10, 100, 150),
    ]
    candidates = []
    for family, name, volume, holding, count, retain in families:
        for scorer in previous["scorers"]:
            candidates.append(
                {
                    "id": family + "_" + scorer,
                    "family": family,
                    "name": name,
                    "scorer": scorer,
                    "volume_min": volume,
                    "rebalance_days": holding,
                    "positions": count,
                    "retain_rank": retain,
                    "weighting": "equal",
                }
            )
    config = {
        "version": 1,
        "date": "2026-10-16",
        "objective": "net outperformance of the pool's own index",
        "scope": "18 finite candidate rules, 15 viewed development windows, two pools",
        "pools": previous["pools"],
        "cost_bps": [0, 10, 20],
        "common": previous["common"],
        "windows": previous["windows"],
        "candidates": candidates,
        "exposure": 1.0,
        "new_model_fits_in_development": 0,
        "pairs": [["F1", "F0"], ["F2", "F1"], ["F3", "F2"], ["F4", "F3"], ["F5", "F3"]],
        "development_gates": {
            "mean_own_index_excess_min": 0.005,
            "median_own_index_excess_min": 0.0,
            "wins_min": 9,
            "mean_20bps_own_index_excess_min": 0.0,
            "aligned_total_return_index_excess_min": 0.0,
            "drop_best_mean_excess_min": 0.0,
            "all_entry_year_mean_excess_min": 0.0,
            "worst_relative_drawdown_floor": -0.02,
        },
        "selection": (
            "one finalist per pool; all-gate passers first; otherwise provisional only; "
            "sort by smallest maximum shortfall across entry-year, drop-best and 20bps excess, "
            "then mean own-index excess, then candidate id; never reselect after external results"
        ),
        "external_windows": [
            {"label": "external_2019_jan", "start_on_or_after": "2019-01-02"},
            {"label": "external_2019_jul", "start_on_or_after": "2019-07-01"},
            {"label": "external_2020_feb", "start_on_or_after": "2020-02-03"},
            {"label": "external_2020_sep", "start_on_or_after": "2020-09-01"},
            {"label": "external_2021_mar", "start_on_or_after": "2021-03-01"},
            {"label": "external_2022_apr", "start_on_or_after": "2022-04-01"},
        ],
        "external_window_trading_days": 30,
        "external_data_start": "2017-07-01",
        "external_data_end": "2022-06-30",
        "external_gates": {
            "mean_own_index_excess_min": 0.005,
            "wins_min": 4,
            "mean_20bps_own_index_excess_min": 0.0,
            "aligned_total_return_index_excess_min": 0.0,
            "drop_best_mean_excess_min": 0.0,
        },
        "external_interpretation": (
            "previously unviewed backward-time historical stress validation, "
            "not prospective validation; "
            "fitted candidates must train only before each external window"
        ),
        "recent_forward_check": {
            "start_on_or_after": "2026-09-25",
            "end": "2026-10-15",
            "full_30_day_acceptance": False,
        },
        "source_plan_sha256": sha256(PREVIOUS / "plan_locked.json"),
        "frozen_market_sha256": previous["market_sha256"],
        "production_config_sha256": sha256(ROOT / "config/default.toml"),
        "financial_industry_inputs": [],
        "post_external_parameter_changes": False,
    }
    destination = ROOT / "config/research/index_improvement_20261016.json"
    locked = OUT / "plan_locked.json"
    for path in [destination, locked]:
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == config
        else:
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    assert sha256(destination) == sha256(locked)
    print("Locked", len(candidates), "candidates and six untouched historical windows", flush=True)


if __name__ == "__main__":
    main()
