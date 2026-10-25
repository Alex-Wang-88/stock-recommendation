"""Evaluate every predeclared development gate and lock one finalist before holdout."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from research_resources import configure_background_job

configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-rule-optimization-20261014"


def aggregate(frame):
    import pandas as pd

    rows = []
    for (candidate, cost), group in frame[frame.account == "strategy"].groupby(
        ["candidate", "cost_bps"], sort=False
    ):
        row = {
            "candidate": candidate,
            "name": group.name.iloc[0],
            "cost_bps": cost,
            "windows": len(group),
            "mean_return": group["return"].mean(),
            "median_return": group["return"].median(),
            "mean_pool_excess": group.excess_pool.mean(),
            "mean_full_pool_excess": group.excess_full_pool.mean(),
            "worst_return": group["return"].min(),
            "worst_drawdown": group.max_drawdown.min(),
            "mean_trade_notional": group.trade_notional.mean(),
            "mean_fees": group.fees.mean(),
            "mean_exposure": group.mean_target_exposure.mean(),
            "max_residual": group.residual.max(),
        }
        for code in [
            "000300",
            "000905",
            "000852",
            "000985",
            "H00300",
            "H00905",
            "H00852",
            "H00985",
        ]:
            row["mean_excess_" + code] = group["excess_" + code].mean()
            row["median_excess_" + code] = group["excess_" + code].median()
            row["wins_" + code] = int(group["excess_" + code].gt(1e-10).sum())
        if "excess_weighted_pool" in group:
            row["mean_weighted_pool_excess"] = group.excess_weighted_pool.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    import pandas as pd
    from prepare_market_weight_batch import sha256

    summary_path = OUT / "development/summary.csv"
    frame = pd.read_csv(summary_path)
    qa = json.loads((OUT / "development/qa.json").read_text(encoding="utf-8"))
    assert qa["technical_status"] == "PASS" and qa["baseline_max_return_difference"] < 1e-10
    suite_path = OUT / "candidate_suite_locked.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    target = json.loads((OUT / "targets_locked.json").read_text(encoding="utf-8"))[
        "first_round_development_targets"
    ]
    agg = aggregate(frame)
    agg.to_csv(OUT / "development/aggregate.csv", index=False, encoding="utf-8-sig")
    primary = agg[agg.cost_bps == 10].copy()
    stress = agg[agg.cost_bps == 20].set_index("candidate")
    gates = []
    for _, row in primary.iterrows():
        checks = {
            "index_excess": row.mean_excess_000985 >= target["mean_excess_index_min"],
            "same_pool_excess": row.mean_pool_excess >= target["mean_pool_excess_same_period_min"],
            "winning_windows": row.wins_000985 >= target["positive_excess_windows_min"],
            "worst_loss": row.worst_return >= target["worst_strategy_return_min"],
            "drawdown_gate": row.worst_drawdown >= target["worst_strategy_drawdown_min"],
            "20bps_stress": stress.loc[row.candidate, "mean_excess_000985"]
            >= target["20bps_mean_excess_index_min"],
            "total_return_aligned": row.mean_excess_H00985
            >= target["total_return_close_close_mean_excess_index_min"],
        }
        gates.append(
            {
                "candidate": row.candidate,
                **checks,
                "all_development_gates_passed": all(checks.values()),
                "passed_gate_count": sum(checks.values()),
            }
        )
    gates = pd.DataFrame(gates)
    gates.to_csv(OUT / "development/target_gates.csv", index=False, encoding="utf-8-sig")
    primary = primary.merge(gates, on="candidate", validate="one_to_one")
    eligible = primary[primary.all_development_gates_passed]
    selection_reason = "all_development_targets_passed"
    if eligible.empty:
        eligible = primary[primary.worst_loss & primary.drawdown_gate]
        selection_reason = "targets_not_met; best_index_excess_among_risk_gate_passers"
    if eligible.empty:
        eligible = primary.sort_values(
            ["worst_drawdown", "mean_excess_000985"], ascending=[False, False]
        )
        selection_reason = "targets_not_met; no_risk_gate_passers; least_worst_drawdown"
    else:
        eligible = eligible.sort_values(
            ["mean_excess_000985", "mean_pool_excess", "mean_trade_notional"],
            ascending=[False, False, True],
        )
    winner = str(eligible.iloc[0].candidate)
    destination = OUT / "finalist_locked.json"
    payload = {
        "candidate_id": winner,
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "candidate_suite_sha256": sha256(suite_path),
        "development_summary_sha256": sha256(summary_path),
        "selection_reason": selection_reason,
        "candidate": next(c for c in suite["candidates"] if c["id"] == winner),
        "all_development_targets_passed": bool(eligible.iloc[0].all_development_gates_passed),
        "holdout_returns_read_during_selection": False,
        "candidates_screened": 12,
    }
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        assert (
            previous["candidate_id"] == winner
            and previous["development_summary_sha256"] == payload["development_summary_sha256"]
        )
    else:
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    primary.to_csv(OUT / "development/primary_targets.csv", index=False, encoding="utf-8-sig")
    frame["year"] = frame.start.str[:4]
    by_year = []
    for year, group in frame.groupby("year"):
        part = aggregate(group)
        part["year"] = year
        by_year.append(part)
    pd.concat(by_year, ignore_index=True).to_csv(
        OUT / "development/year_stability.csv", index=False, encoding="utf-8-sig"
    )
    diagnostic = pd.read_csv(OUT / "development/factor_diagnostics.csv")
    diag_aggregate = diagnostic.groupby(["volume_min", "horizon", "factor"]).agg(
        signal_dates=("rank_ic", "size"),
        mean_rank_ic=("rank_ic", "mean"),
        median_rank_ic=("rank_ic", "median"),
        mean_top_minus_bottom=("top_minus_bottom", "mean"),
        positive_rank_ic_fraction=("rank_ic", lambda x: (x > 0).mean()),
    )
    diag_aggregate.to_csv(OUT / "development/factor_diagnostic_aggregate.csv", encoding="utf-8-sig")
    print(
        primary[
            [
                "candidate",
                "name",
                "mean_return",
                "mean_excess_000985",
                "mean_pool_excess",
                "worst_return",
                "worst_drawdown",
                "wins_000985",
                "mean_trade_notional",
                "all_development_gates_passed",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
