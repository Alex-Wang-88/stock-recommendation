"""Freeze the existing development windows and all remaining disjoint 30-day blocks."""

from __future__ import annotations

import json
from pathlib import Path

from research_resources import configure_background_job

configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-rule-optimization-20261014"


def main():
    import duckdb
    from prepare_market_weight_batch import sha256

    OUT.mkdir(exist_ok=True)
    destination = OUT / "windows_locked.json"
    if destination.exists():
        raise FileExistsError("Research windows already locked")
    market = ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet"
    source_plans = [
        ROOT / "logs/market-only-weight-design-20261014/batch_plan.json",
        ROOT / "logs/market-only-extended-validation-20261014/plan/batch_plan.json",
    ]
    development = []
    for plan in source_plans:
        development.extend(
            {**w, "role": "development_previously_viewed"}
            for w in json.loads(plan.read_text(encoding="utf-8"))["windows"]
        )
    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    days = (
        con.execute(
            "select cast(trade_date as date) as day from read_parquet(?) "
            "where trade_date >= date '2023-01-01' group by 1 "
            "having count(*)>=4000 order by 1",
            [market.as_posix()],
        )
        .df()
        .day.astype(str)
        .tolist()
    )
    occupied = {d for w in development for d in w["days"]}
    holdouts = []
    # Use the same minimum calendar bound as the previous broad-universe feature study.
    for i, day in enumerate(days):
        if day < "2024-02-01" or day in occupied:
            continue
        held = days[i : i + 30]
        if len(held) != 30 or set(held) & occupied:
            continue
        holdouts.append(
            {
                "label": "untouched_" + day.replace("-", ""),
                "start": day,
                "end": held[-1],
                "days": held,
                "signal_dates": [days[i + j - 1] for j in range(0, 30, 5)],
                "entry_dates": [days[i + j] for j in range(0, 30, 5)],
                "exit_dates": [days[i + j + 4] for j in range(0, 30, 5)],
                "role": "untouched_historical_time_holdout",
            }
        )
        occupied.update(held)
    con.close()
    config = ROOT / "config/research/market_rule_candidates.json"
    targets = ROOT / "config/research/market_only_optimization_targets.json"
    payload = {
        "market_sha256": sha256(market),
        "candidate_sha256": sha256(config),
        "target_sha256": sha256(targets),
        "source_plans": [str(p) for p in source_plans],
        "holdout_selection": (
            "earliest disjoint complete 30-day blocks after 2024-02-01; no returns inspected"
        ),
        "holdouts_locked_before_new_candidate_returns": True,
        "holdout_limit": (
            "time holdout is interspersed with training history, "
            "not forward chronological out-of-sample; "
            "existing data may not provide the six required windows"
        ),
        "windows": development + holdouts,
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "candidate_suite_locked.json").write_bytes(config.read_bytes())
    (OUT / "targets_locked.json").write_bytes(targets.read_bytes())
    print(
        json.dumps(
            {
                "development": len(development),
                "untouched_blocks": [
                    {k: w[k] for k in ["label", "start", "end"]} for w in holdouts
                ],
                "six_window_validation_possible": len(holdouts) >= 6,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
