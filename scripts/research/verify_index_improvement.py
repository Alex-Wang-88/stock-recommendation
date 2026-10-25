"""Audit sealed study outputs, independently rebuild trade ledgers, and snapshot code."""

from __future__ import annotations

import json
import shutil

from design_index_improvement import OUT, ROOT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
KEYS = ["cohort", "pool", "window", "candidate", "cost_bps", "account"]


def quote_closes(cohort, days):
    import duckdb
    import pandas as pd

    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    con.register("audit_days", pd.DataFrame({"day": sorted(days)}))
    if cohort == "development":
        source = ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet"
        quotes = con.execute(
            "select b.symbol, cast(b.trade_date as date)::varchar as day, b.close "
            "from read_parquet(?) b join audit_days a "
            "on cast(b.trade_date as date)=cast(a.day as date)",
            [source.as_posix()],
        ).df()
    else:
        quotes = con.execute(
            "select b.symbol,b.day,b.close from read_parquet(?) b join audit_days using(day)",
            [(OUT / "validation_quotes" / cohort / "*.parquet").as_posix()],
        ).df()
    con.close()
    quotes["symbol"] = quotes.symbol.astype(str).str.zfill(6)
    return quotes.pivot(index="day", columns="symbol", values="close")


def audit_quotes(cohort, expected_symbols, allowed_missing):
    import duckdb

    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    con.from_parquet((OUT / "validation_quotes" / cohort / "*.parquet").as_posix()).create_view(
        "quotes"
    )
    result = con.execute(
        "select count(*) as rows, count(distinct symbol) as symbols, "
        "min(day) as first_day,max(day) as last_day, "
        "count(*)-count(distinct (day,symbol)) as duplicates, "
        "count(*) filter(where tradestatus=1 and volume>0 and "
        "(not isfinite(open) or open<=0 or not isfinite(close) or close<=0 "
        "or high<greatest(open,close) or low>least(open,close))) as invalid_active_ohlc "
        "from quotes"
    ).df().iloc[0].to_dict()
    gaps = con.execute(
        "select symbol,day,close,pre_close,previous_close, "
        "abs(pre_close/previous_close-1) as prior_close_gap from "
        "(select *,lag(close) over(partition by symbol order by day) previous_close "
        "from quotes) where previous_close>0 and pre_close>0 "
        "and abs(pre_close/previous_close-1)>0.001"
    ).df()
    con.close()
    assert result["duplicates"] == 0 and result["invalid_active_ohlc"] == 0, result
    assert result["symbols"] + allowed_missing == expected_symbols
    gaps.to_csv(OUT / cohort / "quote_prior_close_gaps.csv", index=False)
    return {**result, "prior_close_gaps_over_10bps": len(gaps)}


def ledger_nav_error(trades, nav, closes, main_st_days):
    """Rebuild cash and quantities without calling either replay implementation."""
    import numpy as np

    worst, count, st_observations = 0.0, 0, 0
    paths = nav.loc[(nav.account == "strategy") & (nav.cost_bps == 10)]
    trade_groups = dict(tuple(trades.groupby(KEYS, sort=False)))
    for key, path in paths.groupby(KEYS, sort=False):
        ledger = trade_groups.get(key)
        by_day = {} if ledger is None else dict(tuple(ledger.groupby("day", sort=False)))
        cash, quantities, marks = 1.0, {}, {}
        for row in path.sort_values("day").itertuples():
            opening_symbols = set(quantities)
            for trade in () if row.day not in by_day else by_day[row.day].itertuples():
                quantity = trade.notional / trade.price
                if trade.side == "buy":
                    cash -= trade.notional + trade.fee
                    quantities[trade.symbol] = quantities.get(trade.symbol, 0.0) + quantity
                    marks[trade.symbol] = trade.price
                else:
                    assert trade.side == "sell"
                    cash += trade.notional - trade.fee
                    quantities[trade.symbol] -= quantity
                    assert quantities[trade.symbol] >= -1e-10
                    if quantities[trade.symbol] < 1e-12:
                        del quantities[trade.symbol]
            close = closes.loc[row.day]
            for symbol in quantities:
                price = close.get(symbol, float("nan"))
                if np.isfinite(price) and price > 0:
                    marks[symbol] = price
            value = cash + sum(quantity * marks[s] for s, quantity in quantities.items())
            st_observations += sum(
                (row.day, s) in main_st_days for s in opening_symbols | set(quantities)
            )
            worst = max(worst, abs(value - row.nav))
            assert cash >= -1e-9
            count += 1
    assert worst < 1e-9, worst
    return worst, count, st_observations


def verify_cohort(cohort, windows, expected_strategy, expected_reference):
    import pandas as pd
    from account_replay import drawdown

    folder = OUT / cohort
    summary = pd.read_csv(folder / "summary.csv")
    nav = pd.read_csv(folder / "daily_nav.csv")
    trades = pd.read_csv(folder / "trades_10bps.csv", dtype={"symbol": str})
    qa = json.loads((folder / "qa.json").read_text(encoding="utf-8"))
    assert not summary.duplicated(KEYS).any()
    assert not nav.duplicated([*KEYS, "day"]).any()
    assert len(summary) == expected_strategy * 2
    assert qa["strategy_accounts"] == expected_strategy
    assert qa["unique_reference_accounts"] == expected_reference
    expected_days = {w["label"]: w["days"] for w in windows}
    for key, part in nav.groupby(KEYS):
        assert part.day.tolist() == expected_days[key[2]], key
    first = nav.groupby(KEYS).head(1)[[*KEYS, "nav"]].rename(columns={"nav": "first_nav"})
    last = nav.groupby(KEYS).tail(1)[[*KEYS, "nav"]]
    joined = summary.merge(last, on=KEYS, validate="one_to_one").merge(
        first, on=KEYS, validate="one_to_one"
    )
    terminal_error = float((joined["return"] - (joined.nav - 1)).abs().max())
    aligned_error = float(
        (joined.return_close_close - (joined.nav / joined.first_nav - 1)).abs().max()
    )
    assert max(terminal_error, aligned_error) < 1e-12
    measured_dd = (
        nav.groupby(KEYS)
        .nav.apply(lambda x: drawdown(x.to_numpy()))
        .reset_index(name="measured_dd")
    )
    dd_joined = summary.merge(measured_dd, on=KEYS, validate="one_to_one")
    dd_error = float((dd_joined.max_drawdown - dd_joined.measured_dd).abs().max())
    assert dd_error < 1e-12
    ledger = trades.groupby(KEYS).agg(ledger_fees=("fee", "sum"), ledger_orders=("fee", "size"))
    costed = (
        summary.loc[(summary.cost_bps == 10) & (summary.account == "strategy")]
        .merge(ledger.reset_index(), on=KEYS, how="left", validate="one_to_one")
        .fillna({"ledger_fees": 0.0, "ledger_orders": 0})
    )
    fee_error = float((costed.fees - costed.ledger_fees).abs().max())
    formula_error = float((trades.fee - trades.notional * 0.001).abs().max())
    assert max(fee_error, formula_error) < 1e-12
    assert costed.orders.eq(costed.ledger_orders).all()
    costs = summary.pivot(
        index=[k for k in KEYS if k != "cost_bps"], columns="cost_bps", values="return"
    )
    assert (costs[0] + 1e-10 >= costs[10]).all()
    assert (costs[10] + 1e-10 >= costs[20]).all()
    closes = quote_closes(cohort, nav.day.unique())
    if cohort == "development":
        st = pd.read_parquet(
            ROOT / "logs/market-direction-study-20261015/execution_status_research.parquet",
            filters=[("day", ">=", "2026-07-06"), ("isST", "==", 1)],
        )
        st["symbol"] = st.symbol.astype(str).str.zfill(6)
        st = st.loc[~st.symbol.str.startswith(("30", "688", "689"))]
        main_st_days = set(st[["day", "symbol"]].itertuples(index=False, name=None))
    else:
        main_st_days = set()
    rebuild_error, observations, st_days = ledger_nav_error(trades, nav, closes, main_st_days)
    return {
        "summary_rows": len(summary),
        "strategy_accounts": expected_strategy,
        "unique_reference_accounts": expected_reference,
        "terminal_nav_error": terminal_error,
        "aligned_return_error": aligned_error,
        "drawdown_error": dd_error,
        "fee_ledger_error": fee_error,
        "fee_formula_error": formula_error,
        "independent_cash_quantity_nav_error": rebuild_error,
        "independent_nav_observations": observations,
        "legacy_10bps_strategy_main_st_held_days_after_20260706": (
            st_days if cohort == "development" else None
        ),
        "cost_monotonic": True,
    }


def main():
    import pandas as pd
    from prepare_market_weight_batch import sha256

    plan_path = OUT / "plan_locked.json"
    seal_path = OUT / "finalists_locked.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    scope = json.loads((OUT / "validation_windows_locked.json").read_text(encoding="utf-8"))
    request = json.loads((OUT / "validation_download_request.json").read_text(encoding="utf-8"))
    fit = json.loads((OUT / "validation_fit_qa.json").read_text(encoding="utf-8"))
    download = json.loads((OUT / "validation_download_receipt.json").read_text(encoding="utf-8"))
    assert sha256(plan_path) == seal["plan_sha256"] == request["plan_sha256"]
    assert sha256(plan_path) == sha256(ROOT / "config/research/index_improvement_20261016.json")
    assert sha256(seal_path) == request["finalists_sha256"] == fit["finalists_sha256"]
    assert seal["no_reselection_after_validation"]
    assert sha256(OUT / "development/summary.csv") == seal["development_summary_sha256"]
    assert not fit["parameter_search_after_finalist_lock"] and fit["models_fitted"] == 7
    for fold in fit["folds"]:
        assert fold["train_label_end_max"] < fold["valid_start"]
        assert fold["valid_label_end_max"] < fold["test_first_signal"]
    assert not any(x["status"] == "failed" for x in download["outcomes"])
    assert download["request_sha256"] == sha256(OUT / "validation_download_request.json")
    members = pd.read_parquet(OUT / "validation_all_membership.parquet")
    assert not members.duplicated(["day", "pool", "symbol"]).any()
    assert members.updateDate.le(members.day).all()
    viewed = {d for w in plan["windows"] for d in w["days"]}
    unseen = {d for w in scope["external_windows"] for d in w["days"]}
    recent = set(scope["recent_window"]["days"])
    assert not (viewed & unseen or viewed & recent or unseen & recent)
    assert all(len(w["days"]) == 30 for w in scope["external_windows"])
    assert len(recent) == 9
    production = sha256(ROOT / "config/default.toml")
    assert production == "a39f2116505a00b4aff9ba9b6130d8fda5ec1be364548c81a8e30e3479bd1d52"
    cohorts = {}
    for cohort, windows, strategies, references in [
        ("development", plan["windows"], 1620, 360),
        ("external", scope["external_windows"], 90, 72),
        ("recent", [scope["recent_window"]], 15, 12),
    ]:
        cohorts[cohort] = verify_cohort(cohort, windows, strategies, references)
        if cohort != "development":
            absent = sum(
                o["cohort"] == cohort and o["status"] == "no_quotes" for o in download["outcomes"]
            )
            cohorts[cohort]["quote_audit"] = audit_quotes(
                cohort, len(request["jobs"][cohort]["symbols"]), absent
            )
        print(cohort, "independent account audit PASS", flush=True)
    snapshots = OUT / "code_snapshot"
    snapshots.mkdir(exist_ok=True)
    source_names = [
        "design_index_improvement.py",
        "index_improvement_data.py",
        "run_index_suite.py",
        "select_index_finalists.py",
        "prepare_index_validation.py",
        "fetch_index_validation.py",
        "fetch_index_validation_indices.py",
        "index_validation_features.py",
        "fit_index_validation.py",
        "run_index_validation.py",
        "index_account_replay.py",
        "report_index_improvement.py",
        "verify_index_improvement.py",
        "research_resources.py",
        "rule_account_replay.py",
        "account_replay.py",
        "prepare_market_weight_batch.py",
        "qlib_alpha_features.py",
    ]
    paths = [ROOT / "scripts/research" / name for name in source_names]
    paths.append(ROOT / "tests/research/test_index_improvement.py")
    provenance = []
    for path in paths:
        target = snapshots / path.name
        if not target.exists():
            shutil.copyfile(path, target)
        assert sha256(path) == sha256(target), path
        provenance.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    result = {
        "technical_status": "PASS",
        "cohorts": cohorts,
        "sealed_plan_and_finalists_unchanged": True,
        "all_folds_purged": True,
        "validation_windows_disjoint_from_viewed_windows": True,
        "production_configuration_unchanged": True,
        "production_configuration_sha256": production,
        "financial_industry_inputs": [],
        "resource": RESOURCE_CONTEXT,
        "code_provenance": provenance,
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "code_provenance"}, indent=2))


if __name__ == "__main__":
    main()
