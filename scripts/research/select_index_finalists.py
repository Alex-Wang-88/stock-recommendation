"""Disclose every development candidate and seal one finalist per pool before validation."""

from __future__ import annotations

import json

from design_index_improvement import OUT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def summarize(summary):
    import pandas as pd

    rows = []
    strategy = summary.loc[summary.account == "strategy"]
    for (pool, candidate, cost), group in strategy.groupby(["pool", "candidate", "cost_bps"]):
        own, tr = ("000300", "H00300") if pool == "csi300" else ("000905", "H00905")
        alpha = group["excess_" + own]
        years = group.groupby(group.start.str[:4])["excess_" + own].mean()
        rows.append(
            {
                "pool": pool,
                "candidate": candidate,
                "scorer": group.scorer.iloc[0],
                "family": group.family.iloc[0],
                "cost_bps": cost,
                "windows": len(group),
                "mean_return": group["return"].mean(),
                "mean_excess": alpha.mean(),
                "median_excess": alpha.median(),
                "wins": int(alpha.gt(0).sum()),
                "tr_excess": group["excess_" + tr].mean(),
                "drop_best_excess": group.drop(alpha.idxmax())["excess_" + own].mean(),
                "min_year_excess": years.min(),
                "worst_drawdown": group.max_drawdown.min(),
                "worst_relative_drawdown": group.relative_drawdown.min(),
                "mean_pool_excess": group.excess_pool.mean(),
                "mean_fees": group.fees.mean(),
                **{"year_" + str(y): v for y, v in years.items()},
                **{k: group[k].mean() for k in group.columns if k.startswith("excess_")},
            }
        )
    return pd.DataFrame(rows)


def main():
    import pandas as pd
    from prepare_market_weight_batch import sha256

    plan = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    summary_path = OUT / "development/summary.csv"
    summary = pd.read_csv(summary_path)
    aggregate = summarize(summary)
    aggregate.to_csv(OUT / "development/aggregate.csv", index=False)
    main = aggregate.loc[aggregate.cost_bps == 10].copy()
    higher = aggregate.loc[aggregate.cost_bps == 20, ["pool", "candidate", "mean_excess"]].rename(
        columns={"mean_excess": "excess_20bps"}
    )
    main = main.merge(higher, on=["pool", "candidate"], validate="one_to_one")
    gates = plan["development_gates"]
    main["gate_mean"] = main.mean_excess.ge(gates["mean_own_index_excess_min"])
    main["gate_median"] = main.median_excess.ge(gates["median_own_index_excess_min"])
    main["gate_wins"] = main.wins.ge(gates["wins_min"])
    main["gate_cost"] = main.excess_20bps.ge(gates["mean_20bps_own_index_excess_min"])
    main["gate_tr"] = main.tr_excess.ge(gates["aligned_total_return_index_excess_min"])
    main["gate_outlier"] = main.drop_best_excess.ge(gates["drop_best_mean_excess_min"])
    main["gate_years"] = main.min_year_excess.ge(gates["all_entry_year_mean_excess_min"])
    main["gate_drawdown"] = main.worst_relative_drawdown.ge(gates["worst_relative_drawdown_floor"])
    gate_columns = [x for x in main if x.startswith("gate_")]
    main["all_gates"] = main[gate_columns].all(axis=1)
    main["gates_passed"] = main[gate_columns].sum(axis=1)
    main["robust_floor"] = main[["min_year_excess", "drop_best_excess", "excess_20bps"]].min(axis=1)
    main.to_csv(OUT / "development/target_gates.csv", index=False)
    finalists = []
    for pool in plan["pools"]:
        ordered = main.loc[main.pool == pool].sort_values(
            ["all_gates", "robust_floor", "mean_excess", "candidate"],
            ascending=[False, False, False, True],
        )
        chosen = ordered.iloc[0]
        candidate = next(c for c in plan["candidates"] if c["id"] == chosen.candidate)
        finalists.append(
            {
                **candidate,
                "pool": pool,
                "development_all_gates_passed": bool(chosen.all_gates),
                "development_mean_excess": float(chosen.mean_excess),
                "development_robust_floor": float(chosen.robust_floor),
                "release_eligible": False,
            }
        )
    receipt = {
        "plan_sha256": sha256(OUT / "plan_locked.json"),
        "development_summary_sha256": sha256(summary_path),
        "selection_before_external_return_evaluation": True,
        "selection_method": plan["selection"],
        "finalists": finalists,
        "no_reselection_after_validation": True,
    }
    path = OUT / "finalists_locked.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
    else:
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    pairs = []
    strategy = summary.loc[summary.account == "strategy"]
    for left, right in plan["pairs"]:
        for pool in plan["pools"]:
            for scorer in ["P4", "Lasso", "LightGBM"]:
                a = strategy.loc[
                    (strategy.pool == pool)
                    & (strategy.family == left)
                    & (strategy.scorer == scorer)
                ]
                b = strategy.loc[
                    (strategy.pool == pool)
                    & (strategy.family == right)
                    & (strategy.scorer == scorer)
                ]
                merged = a.merge(
                    b, on=["window", "cost_bps"], suffixes=("_new", "_old"), validate="one_to_one"
                )
                merged["difference"] = merged.return_new - merged.return_old
                for cost, part in merged.groupby("cost_bps"):
                    pairs.append(
                        {
                            "pool": pool,
                            "scorer": scorer,
                            "new_family": left,
                            "old_family": right,
                            "cost_bps": cost,
                            "mean_difference": part.difference.mean(),
                            "median_difference": part.difference.median(),
                            "wins": int(part.difference.gt(0).sum()),
                            "fee_difference": (part.fees_new - part.fees_old).mean(),
                        }
                    )
    pd.DataFrame(pairs).to_csv(OUT / "development/paired_structural_changes.csv", index=False)
    print(
        main[
            [
                "pool",
                "candidate",
                "mean_excess",
                "wins",
                "excess_20bps",
                "drop_best_excess",
                "min_year_excess",
                "gates_passed",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("SEALED FINALISTS", json.dumps(finalists, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
