"""Evaluate sealed finalists and pre-existing controls once on untouched data."""

from __future__ import annotations

import json

from design_index_improvement import OUT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import argparse

    import duckdb
    import pandas as pd
    from prepare_market_weight_batch import sha256
    from run_index_suite import run_accounts

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["external", "recent", "both"], default="both")
    args = parser.parse_args()
    plan = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    scope = json.loads((OUT / "validation_windows_locked.json").read_text(encoding="utf-8"))
    chosen = json.loads((OUT / "finalists_locked.json").read_text(encoding="utf-8"))
    assert chosen["plan_sha256"] == sha256(OUT / "plan_locked.json")
    members = pd.read_parquet(OUT / "validation_all_membership.parquet")
    levels = pd.read_parquet(OUT / "validation_indices/levels.parquet")
    candidates = []
    for candidate in chosen["finalists"]:
        candidates.append(candidate)
        for scorer in sorted({"P4", candidate["scorer"]}):
            control = next(c for c in plan["candidates"] if c["id"] == "F0_" + scorer)
            if control["id"] != candidate["id"]:
                candidates.append({**control, "pool": candidate["pool"]})
    for cohort, windows in [
        ("external", scope["external_windows"]),
        ("recent", [scope["recent_window"]]),
    ]:
        if args.cohort != "both" and args.cohort != cohort:
            continue
        config = {**plan, "windows": windows}
        predictions = pd.read_parquet(OUT / (cohort + "_predictions.parquet"))
        factors = pd.read_parquet(OUT / (cohort + "_p4.parquet"))
        factors = factors.merge(
            members[["day", "symbol", "pool"]],
            on=["day", "symbol"],
            validate="one_to_one",
        )
        wanted = sorted({d for w in windows for d in w["days"] + w["signal_dates"]})
        con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
        con.register("wanted_days", pd.DataFrame({"day": wanted}))
        quotes = con.execute(
            "select b.* from read_parquet(?) b join wanted_days w using(day)",
            [(OUT / "validation_quotes" / cohort / "*.parquet").as_posix()],
        ).df()
        con.close()
        status = quotes[["day", "symbol", "isST", "tradestatus"]].copy()
        run_accounts(
            config,
            factors,
            predictions,
            quotes,
            status,
            levels.loc[levels.cohort == cohort].drop(columns="cohort"),
            candidates,
            OUT / cohort,
            cohort,
        )


if __name__ == "__main__":
    main()
