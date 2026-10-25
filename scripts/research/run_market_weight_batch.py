"""Execute the frozen market-only research plan without financial or industry inputs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from research_resources import configure_background_job

# One sequential process; keep Windows foreground apps responsive.
RESOURCE_CONTEXT = configure_background_job()


def job_key(job: dict) -> tuple:
    return (
        job["window"],
        job.get("pool", "market"),
        job["profile"],
        job["one_way_cost_bps"],
        job["execution"],
    )


def main() -> None:
    import duckdb
    import numpy as np
    import pandas as pd
    from account_replay import drawdown, replay, self_checks
    from prepare_market_weight_batch import (
        READ_COLUMNS,
        ROOT,
        pool_frame,
        score_profile,
        sha256,
        validate_suite,
        verify_isolation,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "logs/market-only-weight-design-20261014/batch_plan.json",
    )
    parser.add_argument(
        "--market",
        type=Path,
        default=ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=ROOT
        / "logs/historical-weight-validation-20261014/historical_status_valuation.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / datetime.now().strftime("market-only-weight-results-%Y%m%d"),
    )
    args = parser.parse_args()
    started = time.monotonic()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    suite_path = args.plan.parent / "suite_locked.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    validate_suite(suite)
    suite_source = Path(plan["suite_file"])
    assert sha256(suite_source) == plan["suite_sha256"], "Source suite changed"
    assert suite == json.loads(suite_source.read_text(encoding="utf-8"))
    suite_snapshot_hash = sha256(suite_path)
    factor_path = Path(plan["factor_file"])
    assert sha256(factor_path) == plan["factor_sha256"], "Factor snapshot changed"
    assert plan["fresh_validation_included"] == any(
        window["role"] == "new_time_validation" for window in plan["windows"]
    )
    settings = plan["settings"]
    assert settings == suite["batch_protocol"]
    windows = plan["windows"]
    total_accounts = len(plan["account_jobs"]) + len(plan["market_reference_jobs_plan"])
    profiles = {profile["id"]: profile for profile in suite["profiles"]}
    input_paths = [
        args.plan,
        suite_path,
        suite_source,
        factor_path,
        args.market,
        args.status,
        Path(__file__),
        Path(__file__).with_name("account_replay.py"),
        Path(__file__).with_name("prepare_market_weight_batch.py"),
        Path(__file__).with_name("research_resources.py"),
    ]
    receipt = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in input_paths
    ]
    production_path = ROOT / "config/default.toml"
    production_hash = sha256(production_path)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.csv").exists():
        raise FileExistsError("Completed results already exist; select a new --output directory")
    (args.output / "input_receipts.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "suite_locked.json").write_bytes(suite_path.read_bytes())
    (args.output / "plan_locked.json").write_bytes(args.plan.read_bytes())
    checks = self_checks()
    checks.update(RESOURCE_CONTEXT)

    factors = pd.read_parquet(factor_path, columns=list(READ_COLUMNS))
    factors["symbol"] = factors.symbol.astype(str).str.zfill(6)
    factors["day"] = factors.day.astype(str).str[:10]
    days = sorted({day for window in windows for day in window["days"] + window["signal_dates"]})
    # Source file includes PE, but it is not loaded: only historical execution status is read.
    status_columns = ["date", "symbol", "isST", "tradestatus"]
    history = pd.read_parquet(args.status, columns=status_columns)
    history["day"] = history.date.astype(str).str[:10]
    history = history.loc[history.day.isin(days)].copy()
    history["symbol"] = history.symbol.astype(str).str.zfill(6)
    assert not history.duplicated(["day", "symbol"]).any()
    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    con.register("wanted_days", pd.DataFrame({"day": days}))
    prices = con.execute(
        "select b.symbol,cast(b.trade_date as date) as day,b.open,b.high,b.low,b.close,"
        "b.pre_close,b.volume from read_parquet(?) b join wanted_days w "
        "on cast(b.trade_date as date)=cast(w.day as date)",
        [args.market.as_posix()],
    ).df()
    con.close()
    prices["day"] = prices.day.astype(str)
    prices["symbol"] = prices.symbol.astype(str).str.zfill(6)
    assert not prices.duplicated(["day", "symbol"]).any()
    grid = prices.pivot(index="day", columns="symbol", values="close")
    symbols = grid.columns.tolist()
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    day_index = {day: i for i, day in enumerate(grid.index)}
    arrays = {
        name: prices.pivot(index="day", columns="symbol", values=name)
        .reindex(index=grid.index, columns=symbols)
        .to_numpy()
        for name in ("open", "close", "high", "low", "volume")
    }
    arrays["prior_close"] = (
        prices.pivot(index="day", columns="symbol", values="pre_close")
        .reindex(index=grid.index, columns=symbols)
        .to_numpy()
    )
    for name, column, fill in [("tradable", "tradestatus", 0), ("is_st", "isST", 1)]:
        arrays[name] = (
            history.pivot(index="day", columns="symbol", values=column)
            .reindex(index=grid.index, columns=symbols)
            .fillna(fill)
            .to_numpy()
            == 1
        )
    arrays["growth_board"] = np.array(
        [symbol.startswith(("300", "301", "688", "689")) for symbol in symbols]
    )
    source_status = {day: group.set_index("symbol") for day, group in history.groupby("day")}

    def execute(signal, entry, ranking, limit):
        row, previous = day_index[entry], day_index[signal]
        chosen = []
        rejected = {"missing_or_suspended": 0, "limit_up_open": 0}
        for symbol in ranking:
            if symbol not in symbol_index:
                rejected["missing_or_suspended"] += 1
                continue
            stock = symbol_index[symbol]
            opening = arrays["open"][row, stock]
            valid = np.isfinite(opening) and opening > 0 and arrays["volume"][row, stock] > 0
            if not valid or not arrays["tradable"][row, stock] or arrays["is_st"][row, stock]:
                rejected["missing_or_suspended"] += 1
                continue
            reference = arrays["close"][previous, stock]
            threshold = 0.195 if arrays["growth_board"][stock] else 0.095
            if np.isfinite(reference) and opening / reference - 1 >= threshold:
                rejected["limit_up_open"] += 1
                continue
            chosen.append(symbol)
            if limit is not None and len(chosen) >= limit:
                break
        return chosen, rejected

    schedules, coverage, picks, cycles = {}, [], [], []
    declared_coverage = {(row["signal"], row["pool"]): row for row in plan["coverage"]}
    for window in windows:
        market_book = []
        for signal, entry in zip(window["signal_dates"], window["entry_dates"], strict=True):
            source = [
                symbol
                for symbol in source_status[signal].index
                if symbol in symbol_index
                and np.isfinite(arrays["close"][day_index[signal], symbol_index[symbol]])
            ]
            market_book.append(execute(signal, entry, source, None)[0])
        schedules[(window["label"], "market", "market_equal_gross")] = market_book
        for pool in settings["pool_groups"]:
            books = {profile: [] for profile in profiles}
            books["same_pool_equal_same_cost"] = []
            for signal, entry in zip(window["signal_dates"], window["entry_dates"], strict=True):
                frame, incomplete = pool_frame(
                    factors.loc[factors.day == signal],
                    pool["id"],
                    settings["shared_pool_rules"],
                )
                declared = declared_coverage[(signal, pool["id"])]
                assert len(frame) == declared["common_complete_candidates"]
                assert incomplete == declared["incomplete_candidates_excluded_for_all_profiles"]
                if not coverage:
                    checks.update(verify_isolation(suite, frame.head(100)))
                coverage.append(
                    {
                        "signal": signal,
                        "pool": pool["id"],
                        "complete_candidates": len(frame),
                        "incomplete_excluded": incomplete,
                    }
                )
                books["same_pool_equal_same_cost"].append(
                    execute(signal, entry, frame.symbol.tolist(), None)[0],
                )
                for profile_id, profile in profiles.items():
                    ranking = score_profile(frame, profile)
                    selected, excluded = execute(
                        signal,
                        entry,
                        ranking.symbol.tolist(),
                        settings["position_limit"],
                    )
                    assert np.allclose(ranking.score_coverage, 1.0)
                    books[profile_id].append(selected)
                    cycles.append(
                        {
                            "window": window["label"],
                            "pool": pool["id"],
                            "profile": profile_id,
                            "signal": signal,
                            "entry": entry,
                            "pool_size": len(frame),
                            "selected": len(selected),
                            **excluded,
                        }
                    )
                    lookup = ranking.set_index("symbol")
                    for rank, symbol in enumerate(selected, 1):
                        picks.append(
                            {
                                "window": window["label"],
                                "pool": pool["id"],
                                "profile": profile_id,
                                "signal": signal,
                                "entry": entry,
                                "rank": rank,
                                "symbol": symbol,
                                "score": lookup.loc[symbol, "score"],
                            }
                        )
            for profile, book in books.items():
                schedules[(window["label"], pool["id"], profile)] = book
    (args.output / "target_schedules.json").write_text(
        json.dumps(
            [
                {"window": key[0], "pool": key[1], "profile": key[2], "targets": value}
                for key, value in schedules.items()
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, rows in [("holdings", picks), ("cycles", cycles), ("factor_coverage", coverage)]:
        pd.DataFrame(rows).to_csv(args.output / (name + ".csv"), index=False, encoding="utf-8-sig")
    print("Targets frozen; no financial/industry/PE data loaded.", flush=True)

    results, summary, daily, events = {}, [], [], []
    manual_differences = []
    benchmark_name = "same_pool_equal_same_cost"
    with (args.output / "trades_10bps.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window",
                "pool",
                "profile",
                "execution",
                "day",
                "symbol",
                "side",
                "notional",
                "fee",
                "price",
            ],
        )
        writer.writeheader()
        for window in windows:
            label = window["label"]
            jobs = [job for job in plan["market_reference_jobs_plan"] if job["window"] == label]
            pool_jobs = [job for job in plan["account_jobs"] if job["window"] == label]
            pool_jobs.sort(key=lambda job: (job["profile"] != benchmark_name, job_key(job)))
            jobs += pool_jobs
            for job in jobs:
                key = job_key(job)
                assert key not in results
                pool = job.get("pool", "market")
                profile_id = job["profile"]
                book = schedules[(label, pool, profile_id)]
                limit = settings["position_limit"] if profile_id in profiles else None
                path, ledger, unfilled, residual = replay(
                    window,
                    book,
                    arrays,
                    symbol_index,
                    day_index,
                    job["one_way_cost_bps"],
                    job["execution"],
                    limit,
                )
                row = {
                    **job,
                    "pool": pool,
                    "start": window["start"],
                    "end": window["end"],
                    "return": float(path[-1] - 1),
                    "max_drawdown": drawdown(path),
                    "total_traded_notional": sum(trade["notional"] for trade in ledger),
                    "total_fees": sum(trade["fee"] for trade in ledger),
                    "trade_orders": len(ledger),
                    "unfilled_exit_events": sum(
                        event["action"] == "unfilled_exit_retained" for event in unfilled
                    ),
                    "retained_rebalance_events": sum(
                        event["action"] == "rebalance_retained" for event in unfilled
                    ),
                    "residual_positions": residual,
                }
                results[key] = (path, row)
                if profile_id != "market_equal_gross":
                    benchmark_key = (
                        label,
                        pool,
                        benchmark_name,
                        job["one_way_cost_bps"],
                        job["execution"],
                    )
                    reference_key = (label, "market", "market_equal_gross", 0, job["execution"])
                    benchmark = results[benchmark_key][0]
                    reference = results[reference_key][0]
                    row["pool_return_same_cost"] = float(benchmark[-1] - 1)
                    row["market_return_gross"] = float(reference[-1] - 1)
                    row["excess_pool_same_cost"] = float(path[-1] - benchmark[-1])
                    row["excess_market_gross"] = float(path[-1] - reference[-1])
                summary.append(row)
                context = {
                    "window": label,
                    "pool": pool,
                    "profile": profile_id,
                    "execution": job["execution"],
                }
                daily.extend(
                    {
                        **context,
                        "one_way_cost_bps": job["one_way_cost_bps"],
                        "day": day,
                        "nav": float(value),
                    }
                    for day, value in zip(window["days"], path, strict=True)
                )
                events.extend(
                    {
                        **context,
                        "one_way_cost_bps": job["one_way_cost_bps"],
                        **event,
                        "symbol": symbols[event["stock"]],
                    }
                    for event in unfilled
                )
                if profile_id in profiles and job["one_way_cost_bps"] == 10:
                    for trade in ledger:
                        writer.writerow(
                            {
                                **context,
                                **{
                                    field: trade[field]
                                    for field in ("day", "side", "notional", "fee", "price")
                                },
                                "symbol": symbols[trade["stock"]],
                            }
                        )
                if (
                    profile_id == "P0_price_weight_control"
                    and job["execution"] == "cycle_close_open"
                    and not unfilled
                ):
                    value = 1.0
                    fee = job["one_way_cost_bps"] / 10000
                    for entry, exit_day, names in zip(
                        window["entry_dates"],
                        window["exit_dates"],
                        book,
                        strict=True,
                    ):
                        stocks = [symbol_index[name] for name in names]
                        relative = (
                            arrays["close"][day_index[exit_day], stocks]
                            / arrays["open"][day_index[entry], stocks]
                        )
                        assert np.isfinite(relative).all()
                        value *= float(np.mean(relative)) * (1 - fee) ** 2
                    difference = abs(value - path[-1])
                    assert difference < 1e-10
                    manual_differences.append(difference)
            print(
                f"{label}: {len(jobs)} accounts completed ({len(results)}/{total_accounts})",
                flush=True,
            )
    expected = {job_key(job) for job in plan["account_jobs"] + plan["market_reference_jobs_plan"]}
    assert set(results) == expected
    assert len(results) == total_accounts
    for name, rows in [("summary", summary), ("daily_nav", daily), ("account_events", events)]:
        pd.DataFrame(rows).to_csv(args.output / (name + ".csv"), index=False, encoding="utf-8-sig")
    assert sha256(production_path) == production_hash
    assert sha256(suite_path) == suite_snapshot_hash
    assert sha256(suite_source) == plan["suite_sha256"]
    assert sha256(factor_path) == plan["factor_sha256"]
    checks.update(
        {
            "technical_status": "PASS",
            "full_batch_returns_executed": True,
            "completed_account_jobs": len(results),
            "all_declared_jobs_completed": True,
            "factor_columns_read": list(READ_COLUMNS),
            "status_columns_read": status_columns,
            "financial_tables_read": [],
            "industry_reference_files_read": [],
            "independent_cycle_formula_max_difference": max(manual_differences),
            "production_config_unchanged": True,
            "fresh_validation_included": plan["fresh_validation_included"],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    )
    (args.output / "qa.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "accounts": len(results),
                "elapsed_seconds": checks["elapsed_seconds"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
