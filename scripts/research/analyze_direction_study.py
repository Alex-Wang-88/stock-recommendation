"""Describe paired changes and cross-period consistency without treating viewed data as new OOS."""

from __future__ import annotations

import json
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-direction-study-20261015"
PREVIOUS = ROOT / "logs/market-rule-optimization-20261014"


def add_robustness(frame):
    import pandas as pd
    from select_rule_finalist import aggregate

    result = aggregate(frame)
    extras = []
    for (candidate, cost), group in frame[frame.account == "strategy"].groupby(
        ["candidate", "cost_bps"]
    ):
        alpha = group.excess_000985.sort_values()
        years = group.groupby(group.start.str[:4]).excess_000985.mean()
        extras.append(
            {
                "candidate": candidate,
                "cost_bps": cost,
                "mean_excess_without_best_window": alpha.iloc[:-1].mean(),
                "mean_excess_without_worst_window": alpha.iloc[1:].mean(),
                "minimum_year_mean_excess": years.min(),
                "positive_year_count": int((years > 0).sum()),
                "calendar_years": len(years),
                "index_win_fraction": (alpha > 1e-10).mean(),
            }
        )
    return result.merge(pd.DataFrame(extras), on=["candidate", "cost_bps"], validate="one_to_one")


def paired_metrics(frame, spec, fraction):
    base = frame[frame.candidate == spec["base"]].set_index(["window", "cost_bps"])
    alt = frame[frame.candidate == spec["alternative"]].set_index(["window", "cost_bps"])
    assert set(base.index) == set(alt.index)
    effect = (alt["return"] - base["return"]).rename("effect").reset_index()
    effect = effect.merge(
        frame[["window", "start", "year", "realized_regime", "signal_regime"]].drop_duplicates(),
        on="window",
        validate="many_to_one",
    )
    primary = effect[effect.cost_bps == 10]
    stress = effect[effect.cost_bps == 20]
    changes = primary.effect
    by_year = primary.groupby("year").effect.mean()
    drop_best = changes.sort_values().iloc[:-1].mean()
    drop_worst = changes.sort_values().iloc[1:].mean()
    positive = bool(
        changes.mean() > 1e-12
        and changes.median() > 1e-12
        and (changes > 1e-12).mean() + 1e-12 >= fraction
        and by_year.ge(-1e-12).all()
        and stress.effect.mean() >= -1e-12
        and drop_best > 1e-12
    )
    negative = bool(
        changes.mean() < -1e-12
        and changes.median() < -1e-12
        and (changes < -1e-12).mean() + 1e-12 >= fraction
        and by_year.le(1e-12).all()
        and stress.effect.mean() <= 1e-12
        and drop_worst < -1e-12
    )
    row = {
        **spec,
        "windows": len(primary),
        "mean_effect": changes.mean(),
        "median_effect": changes.median(),
        "positive_windows": int((changes > 1e-12).sum()),
        "negative_windows": int((changes < -1e-12).sum()),
        "20bps_mean_effect": stress.effect.mean(),
        "effect_without_best_window": drop_best,
        "effect_without_worst_window": drop_worst,
        "minimum_year_mean_effect": by_year.min(),
        "maximum_year_mean_effect": by_year.max(),
        "year_means": json.dumps(by_year.to_dict()),
        "status": "consistent_positive"
        if positive
        else ("consistent_negative" if negative else "mixed"),
    }
    effect["pair_id"], effect["pair_name"] = spec["id"], spec["name"]
    return row, effect


def main():
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import sha256

    qa = json.loads((OUT / "research/qa.json").read_text(encoding="utf-8"))
    assert qa["technical_status"] == "PASS" and qa["baseline_max_return_difference"] < 1e-10
    for receipt in qa["input_receipts"]:
        assert sha256(Path(receipt["path"])) == receipt["sha256"], receipt["path"]
    config = json.loads((OUT / "candidate_suite_locked.json").read_text(encoding="utf-8"))
    locked = json.loads((OUT / "windows_locked.json").read_text(encoding="utf-8"))
    targets = json.loads((OUT / "targets_locked.json").read_text(encoding="utf-8"))[
        "first_round_development_targets"
    ]
    full = pd.read_csv(OUT / "research/summary.csv")
    assert len(full[full.account == "strategy"]) == 1080
    levels = pd.read_parquet(PREVIOUS / "index_data/levels.parquet")
    index = levels[levels.code == "000985"].set_index("day").sort_index()
    ma60 = index.close.rolling(60).mean()
    metadata = []
    for window in locked["windows"]:
        signal = window["signal_dates"][0]
        value = float(index.loc[window["end"], "close"] / index.loc[window["start"], "open"] - 1)
        metadata.append(
            {
                "window": window["label"],
                "year": window["start"][:4],
                "realized_regime": "up"
                if value > 0.05
                else ("down" if value < -0.05 else "sideways"),
                "signal_regime": "above_ma60"
                if index.loc[signal, "close"] > ma60.loc[signal]
                else "below_ma60",
                "index_return": value,
                "original14": window["prior_role"] == "development_previously_viewed",
            }
        )
    metadata = pd.DataFrame(metadata)
    metadata.to_csv(OUT / "window_metadata.csv", index=False, encoding="utf-8-sig")
    full = full.merge(metadata, on="window", validate="many_to_one")
    all_aggregates, pairs, pair_windows = [], [], []
    for cohort, frame in [("original_14", full[full.original14]), ("all_15_viewed", full)]:
        result = add_robustness(frame)
        result["cohort"] = cohort
        all_aggregates.append(result)
        for spec in config["paired_hypotheses"]:
            row, effect = paired_metrics(
                frame[frame.account == "strategy"],
                spec,
                config["pattern_consistency_rule"]["same_direction_fraction_min"],
            )
            row["cohort"], effect["cohort"] = cohort, cohort
            pairs.append(row)
            pair_windows.append(effect)
    agg = pd.concat(all_aggregates, ignore_index=True)
    agg.to_csv(OUT / "aggregate.csv", index=False, encoding="utf-8-sig")
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(OUT / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    paired = pd.concat(pair_windows, ignore_index=True)
    paired.to_csv(OUT / "paired_window_effects.csv", index=False, encoding="utf-8-sig")
    stratified = []
    for column in ["year", "realized_regime", "signal_regime"]:
        for value, group in full.groupby(column):
            part = add_robustness(group)
            part["grouping"], part["group"] = column, value
            stratified.append(part)
    pd.concat(stratified, ignore_index=True).to_csv(
        OUT / "stratified_results.csv", index=False, encoding="utf-8-sig"
    )
    paired_groups = []
    selection = paired[(paired.cohort == "all_15_viewed") & (paired.cost_bps == 10)]
    for column in ["year", "realized_regime", "signal_regime"]:
        for (pair_id, value), group in selection.groupby(["pair_id", column]):
            paired_groups.append(
                {
                    "pair_id": pair_id,
                    "grouping": column,
                    "group": value,
                    "windows": len(group),
                    "mean_effect": group.effect.mean(),
                    "median_effect": group.effect.median(),
                    "positive_windows": int(group.effect.gt(1e-12).sum()),
                }
            )
    pd.DataFrame(paired_groups).to_csv(
        OUT / "paired_stratified_results.csv", index=False, encoding="utf-8-sig"
    )
    original = agg[(agg.cohort == "original_14") & (agg.cost_bps == 10)]
    stress = agg[(agg.cohort == "original_14") & (agg.cost_bps == 20)].set_index("candidate")
    gates = []
    for _, row in original.iterrows():
        checks = {
            "index_excess": row.mean_excess_000985 >= targets["mean_excess_index_min"],
            "same_pool_excess": row.mean_pool_excess >= targets["mean_pool_excess_same_period_min"],
            "winning_windows": row.wins_000985 >= targets["positive_excess_windows_min"],
            "worst_loss": row.worst_return >= targets["worst_strategy_return_min"],
            "drawdown_gate": row.worst_drawdown >= targets["worst_strategy_drawdown_min"],
            "20bps_stress": stress.loc[row.candidate, "mean_excess_000985"]
            >= targets["20bps_mean_excess_index_min"],
            "total_return_aligned": row.mean_excess_H00985
            >= targets["total_return_close_close_mean_excess_index_min"],
        }
        gates.append(
            {
                "candidate": row.candidate,
                **checks,
                "passed_gates": sum(checks.values()),
                "all_development_targets_passed": all(checks.values()),
            }
        )
    gate_frame = pd.DataFrame(gates)
    gate_frame.to_csv(OUT / "original14_target_gates.csv", index=False, encoding="utf-8-sig")
    primary = agg[(agg.cohort == "all_15_viewed") & (agg.cost_bps == 10)].sort_values(
        "mean_excess_000985", ascending=False
    )
    primary.to_csv(OUT / "primary_all15.csv", index=False, encoding="utf-8-sig")
    consistent = pair_frame[(pair_frame.cohort == "all_15_viewed") & (pair_frame.status != "mixed")]
    assert np.isfinite(primary[["mean_return", "mean_excess_000985", "worst_drawdown"]]).all().all()
    insights = {
        "technical_status": "PASS",
        "source_strategy_accounts": 1080,
        "all_windows_previously_viewed": True,
        "new_rule_variants": 21,
        "cumulative_rule_variants_in_two_rounds": 33,
        "original14_target_passers": gate_frame.loc[
            gate_frame.all_development_targets_passed, "candidate"
        ].tolist(),
        "top_mean_excess_candidates": primary.head(5).candidate.tolist(),
        "consistent_pairs": consistent[
            ["id", "name", "mean_effect", "positive_windows", "negative_windows", "status"]
        ].to_dict(orient="records"),
        "regime_counts": metadata.realized_regime.value_counts().to_dict(),
        "signal_regime_counts": metadata.signal_regime.value_counts().to_dict(),
        "formal_out_of_sample_status": "NOT_TESTED_NO_NEW_UNVIEWED_WINDOWS",
        "research_summary_sha256": sha256(OUT / "research/summary.csv"),
    }
    (OUT / "insights.json").write_text(
        json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        primary[
            [
                "candidate",
                "name",
                "mean_return",
                "mean_excess_000985",
                "median_excess_000985",
                "wins_000985",
                "worst_drawdown",
                "minimum_year_mean_excess",
                "mean_excess_without_best_window",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(json.dumps(insights, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
