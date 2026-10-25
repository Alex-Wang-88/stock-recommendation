"""Replay a finite frozen rule suite, and test only the development-selected finalist."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-rule-optimization-20261014"


def main():
    import duckdb
    import numpy as np
    import pandas as pd
    from account_replay import drawdown
    from prepare_market_weight_batch import FACTORS, READ_COLUMNS, pool_frame, score_profile, sha256
    from rule_account_replay import replay_rules, self_checks

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["development", "holdout"], default="development")
    args = parser.parse_args()
    started = time.monotonic()
    result_dir = OUT / args.cohort
    result_dir.mkdir(exist_ok=True)
    if (result_dir / "summary.csv").exists():
        raise FileExistsError("Completed results already exist")
    locked_path = OUT / "windows_locked.json"
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    config_path = OUT / "candidate_suite_locked.json"
    target_path = OUT / "targets_locked.json"
    assert sha256(config_path) == locked["candidate_sha256"]
    assert sha256(target_path) == locked["target_sha256"]
    assert (
        sha256(ROOT / "config/research/market_rule_candidates.json") == locked["candidate_sha256"]
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = config["candidates"]
    assert len(candidates) == 12 and len({c["id"] for c in candidates}) == 12
    if args.cohort == "holdout":
        finalist = json.loads((OUT / "finalist_locked.json").read_text(encoding="utf-8"))
        assert finalist["candidate_suite_sha256"] == locked["candidate_sha256"]
        assert finalist["development_summary_sha256"] == sha256(OUT / "development/summary.csv")
        candidates = [c for c in candidates if c["id"] in {"O00", finalist["candidate_id"]}]
    windows = [
        w
        for w in locked["windows"]
        if (w["role"] == "development_previously_viewed") == (args.cohort == "development")
    ]
    suite = json.loads(
        (ROOT / "config/research/market_only_weights.json").read_text(encoding="utf-8")
    )
    profile = next(p for p in suite["profiles"] if p["id"] == "P4_equal_categories")
    factor_path = OUT / (args.cohort + "_factors.parquet")
    status_path = OUT / ("execution_status_" + args.cohort + ".parquet")
    market = ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet"
    assert sha256(market) == locked["market_sha256"]
    factors = pd.read_parquet(factor_path, columns=list(READ_COLUMNS))
    factors["symbol"] = factors.symbol.astype(str).str.zfill(6)
    status = pd.read_parquet(status_path, columns=["day", "symbol", "isST", "tradestatus"])
    status["symbol"] = status.symbol.astype(str).str.zfill(6)
    days = sorted({d for w in windows for d in w["days"] + w["signal_dates"]})
    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    con.register("wanted", pd.DataFrame({"day": days}))
    prices = con.execute(
        "select b.symbol,cast(b.trade_date as date) as day,"
        "b.open,b.high,b.low,b.close,b.pre_close,b.volume "
        "from read_parquet(?) b join wanted w on cast(b.trade_date as date)=cast(w.day as date)",
        [market.as_posix()],
    ).df()
    con.close()
    prices["day"] = prices.day.astype(str)
    prices["symbol"] = prices.symbol.astype(str).str.zfill(6)
    assert not prices.duplicated(["day", "symbol"]).any()
    grid = prices.pivot(index="day", columns="symbol", values="close")
    symbols, trading_days = grid.columns.tolist(), grid.index.tolist()
    symbol_index, day_index = (
        {s: i for i, s in enumerate(symbols)},
        {d: i for i, d in enumerate(trading_days)},
    )
    arrays = {
        name: prices.pivot(index="day", columns="symbol", values=name)
        .reindex(index=trading_days, columns=symbols)
        .to_numpy()
        for name in ["open", "high", "low", "close", "volume"]
    }
    arrays["prior_close"] = (
        prices.pivot(index="day", columns="symbol", values="pre_close")
        .reindex(index=trading_days, columns=symbols)
        .to_numpy()
    )
    for name, column, fill in [("tradable", "tradestatus", 0), ("is_st", "isST", 1)]:
        arrays[name] = (
            status.pivot(index="day", columns="symbol", values=column)
            .reindex(index=trading_days, columns=symbols)
            .fillna(fill)
            .to_numpy()
            == 1
        )
    arrays["growth_board"] = np.array([s.startswith(("300", "301", "688", "689")) for s in symbols])
    level_path = OUT / "index_data/levels.parquet"
    index_receipt = json.loads((OUT / "index_data/receipt.json").read_text(encoding="utf-8"))
    assert sha256(level_path) == index_receipt["levels_sha256"]
    levels = pd.read_parquet(level_path).set_index(["code", "day"]).sort_index()
    all_share = levels.loc["000985"].sort_index()
    market_vol = all_share.close.pct_change().rolling(20).std(ddof=0) * np.sqrt(252)
    rules = config["common"].copy()
    contexts, coverage, snapshots = {}, [], []
    signal_dates = sorted({d for w in windows for d in w["signal_dates"]})
    for candidate in candidates:
        key = (candidate["volume_min"], candidate["scorer"], candidate["market_vol_target"])
        if key in contexts:
            continue
        collection = {}
        for signal in signal_dates:
            rules["volume_trend_20_min"] = candidate["volume_min"]
            frame, incomplete = pool_frame(factors[factors.day == signal], "broader_liquid", rules)
            frame = frame[frame.symbol.isin(symbol_index)].copy()
            assert len(frame) > 0
            if candidate["scorer"] == "P4":
                ranked = score_profile(frame, profile)
            else:
                ranked = frame[["symbol"]].copy()
                risk = (
                    frame[["volatility_value", "drawdown_value", "downside_volatility_value"]]
                    .rank(pct=True)
                    .mean(axis=1)
                )
                ranked["score"] = 0.5 * risk + 0.5 * frame.turnover_value_20.rank(pct=True)
                ranked = ranked.sort_values(["score", "symbol"], ascending=[False, True])
            target = candidate["market_vol_target"]
            vol = float(market_vol.loc[signal])
            assert np.isfinite(vol) and vol > 0
            exposure = 1.0 if target is None else min(1.0, target / vol)
            collection[signal] = {
                "pool": frame.symbol.tolist(),
                "ranking": ranked.symbol.tolist(),
                "exposure": exposure,
                "volatility": {
                    symbol_index[s]: -float(v)
                    for s, v in zip(frame.symbol, frame.volatility_value, strict=True)
                },
                "frame": frame,
            }
            coverage.append(
                {
                    "volume_min": candidate["volume_min"],
                    "scorer": candidate["scorer"],
                    "vol_target": target,
                    "signal": signal,
                    "pool_size": len(frame),
                    "incomplete_excluded": incomplete,
                    "market_vol": vol,
                    "exposure": exposure,
                }
            )
            snapshots.append(
                {
                    "context": list(key),
                    "signal": signal,
                    "ranking": ranked.symbol.tolist(),
                    "exposure": exposure,
                }
            )
        contexts[key] = collection
    (result_dir / "rankings_locked.json").write_text(json.dumps(snapshots), encoding="utf-8")
    pd.DataFrame(coverage).to_csv(result_dir / "coverage.csv", index=False)
    print("All prior-close rankings and exposures frozen", flush=True)
    checks = self_checks()
    checks.update(RESOURCE_CONTEXT)
    # Diagnostics never use labels outside their original development window.
    diagnostic = []
    if args.cohort == "development":
        for window in windows:
            for volume_min in [1.1, 0.0]:
                collection = contexts[(volume_min, "P4", None)]
                for offset in range(0, 30, 5):
                    signal, entry = window["signal_dates"][offset // 5], window["days"][offset]
                    frame = collection[signal]["frame"].copy().set_index("symbol")
                    stocks = np.array([symbol_index[s] for s in frame.index])
                    opening = arrays["open"][day_index[entry], stocks]
                    previous = arrays["close"][day_index[signal], stocks]
                    limit_up = np.where(arrays["growth_board"][stocks], 0.195, 0.095)
                    valid_entry = (
                        np.isfinite(opening)
                        & (opening > 0)
                        & arrays["tradable"][day_index[entry], stocks]
                        & ~arrays["is_st"][day_index[entry], stocks]
                        & (opening / previous - 1 < limit_up)
                    )
                    for horizon in [5, 10, 20]:
                        if offset + horizon > 30:
                            continue
                        exit_day = window["days"][offset + horizon - 1]
                        ending = arrays["close"][day_index[exit_day], stocks]
                        valid = valid_entry & np.isfinite(ending) & (ending > 0)
                        future = pd.Series(
                            ending[valid] / opening[valid] - 1, index=frame.index[valid]
                        )
                        if len(future) < 50:
                            continue
                        for factor in FACTORS:
                            signal_value = frame.loc[future.index, factor]
                            rank = signal_value.rank(pct=True)
                            diagnostic.append(
                                {
                                    "window": window["label"],
                                    "signal": signal,
                                    "entry": entry,
                                    "exit": exit_day,
                                    "volume_min": volume_min,
                                    "horizon": horizon,
                                    "factor": factor,
                                    "stocks": len(future),
                                    "rank_ic": signal_value.rank().corr(future.rank()),
                                    "top_decile_return": future[rank >= 0.9].mean(),
                                    "bottom_decile_return": future[rank <= 0.1].mean(),
                                    "top_minus_bottom": future[rank >= 0.9].mean()
                                    - future[rank <= 0.1].mean(),
                                }
                            )
        pd.DataFrame(diagnostic).to_csv(
            result_dir / "factor_diagnostics.csv", index=False, encoding="utf-8-sig"
        )
        print("Development-only factor diagnostics complete", flush=True)
    baseline_source = pd.concat(
        [
            pd.read_csv(ROOT / "logs/market-only-weight-results-20261014/summary.csv"),
            pd.read_csv(ROOT / "logs/market-only-extended-validation-20261014/results/summary.csv"),
        ],
        ignore_index=True,
    )
    baseline_source = baseline_source[
        (baseline_source.pool == "broader_liquid") & (baseline_source.execution == "continuous_net")
    ]
    old_daily = None
    if args.cohort == "development":
        old_daily = pd.concat(
            [
                pd.read_csv(ROOT / "logs/market-only-weight-results-20261014/daily_nav.csv"),
                pd.read_csv(
                    ROOT / "logs/market-only-extended-validation-20261014/results/daily_nav.csv"
                ),
            ],
            ignore_index=True,
        )
        old_daily = old_daily[
            (old_daily.pool == "broader_liquid") & (old_daily.execution == "continuous_net")
        ]
    summaries, daily_rows, trade_rows, event_rows, decision_rows, index_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    cache, differences = {}, []
    names = [("000300", "H00300"), ("000905", "H00905"), ("000852", "H00852"), ("000985", "H00985")]
    for window in windows:
        indices = {}
        for price_code, tr_code in names:
            price = levels.loc[price_code].reindex(window["days"])
            total = levels.loc[tr_code].reindex(window["days"])
            assert price.open.gt(0).all() and price.close.gt(0).all() and total.close.gt(0).all()
            indices[price_code] = float(price.close.iloc[-1] / price.open.iloc[0] - 1)
            indices[tr_code] = float(total.close.iloc[-1] / total.close.iloc[0] - 1)
            index_rows.append(
                {
                    "window": window["label"],
                    "price_code": price_code,
                    "price_return": indices[price_code],
                    "total_return_code": tr_code,
                    "total_return_close_close": indices[tr_code],
                }
            )
        for candidate in candidates:
            context_key = (
                candidate["volume_min"],
                candidate["scorer"],
                candidate["market_vol_target"],
            )
            collection = contexts[context_key]
            for cost in config["common"]["cost_bps"]:
                references = {}
                reference_modes = [
                    ("same_pool_equal_overlay", False, "equal"),
                    ("same_pool_equal_full", True, "equal"),
                ]
                if candidate["weighting"] == "inverse_vol":
                    reference_modes.append(("same_pool_inverse_vol_overlay", False, "inverse_vol"))
                for label, full, weighting in reference_modes:
                    cache_key = (
                        window["label"],
                        candidate["volume_min"],
                        candidate["rebalance_days"],
                        None if full else candidate["market_vol_target"],
                        cost,
                        weighting,
                    )
                    if cache_key not in cache:
                        cache[cache_key] = replay_rules(
                            window,
                            collection,
                            arrays,
                            symbol_index,
                            day_index,
                            cost,
                            candidate,
                            reference=True,
                            full_exposure=full,
                            reference_weighting=weighting,
                        )
                    references[label] = cache[cache_key]
                strategy = replay_rules(
                    window, collection, arrays, symbol_index, day_index, cost, candidate
                )
                for label, result in [("strategy", strategy), *references.items()]:
                    path, trades, events, residual, decisions = result
                    context = {
                        "window": window["label"],
                        "candidate": candidate["id"],
                        "name": candidate["name"],
                        "cost_bps": cost,
                        "account": label,
                    }
                    row = {
                        **context,
                        "start": window["start"],
                        "end": window["end"],
                        "return": float(path[-1] - 1),
                        "return_close_close": float(path[-1] / path[0] - 1),
                        "max_drawdown": drawdown(path),
                        "trade_notional": sum(t["notional"] for t in trades),
                        "fees": sum(t["fee"] for t in trades),
                        "orders": len(trades),
                        "unfilled": sum(e["action"] == "unfilled_exit_retained" for e in events),
                        "residual": residual,
                        "mean_target_exposure": float(
                            np.mean([d["exposure_target"] for d in decisions])
                        ),
                        "excess_pool": float(
                            path[-1] - references["same_pool_equal_overlay"][0][-1]
                        ),
                        "excess_full_pool": float(
                            path[-1] - references["same_pool_equal_full"][0][-1]
                        ),
                    }
                    for price_code, tr_code in names:
                        row["excess_" + price_code] = row["return"] - indices[price_code]
                        row["excess_" + tr_code] = row["return_close_close"] - indices[tr_code]
                    if "same_pool_inverse_vol_overlay" in references:
                        row["excess_weighted_pool"] = float(
                            path[-1] - references["same_pool_inverse_vol_overlay"][0][-1]
                        )
                    summaries.append(row)
                    daily_rows.extend(
                        {**context, "day": day, "nav": float(value)}
                        for day, value in zip(window["days"], path, strict=True)
                    )
                    if (
                        candidate["id"] == "O00"
                        and args.cohort == "development"
                        and label in ["strategy", "same_pool_equal_overlay"]
                    ):
                        old_profile = (
                            "P4_equal_categories"
                            if label == "strategy"
                            else "same_pool_equal_same_cost"
                        )
                        expected = baseline_source[
                            (baseline_source.window == window["label"])
                            & (baseline_source.profile == old_profile)
                            & (baseline_source.one_way_cost_bps == cost)
                        ].iloc[0]
                        difference = abs(row["return"] - expected["return"])
                        assert difference < 1e-10, (window["label"], cost, label, difference)
                        prior_path = (
                            old_daily[
                                (old_daily.window == window["label"])
                                & (old_daily.profile == old_profile)
                                & (old_daily.one_way_cost_bps == cost)
                            ]
                            .sort_values("day")
                            .nav.to_numpy()
                        )
                        assert np.max(np.abs(prior_path - path)) < 1e-10
                        differences.append(difference)
                    if label == "strategy" and cost == 10:
                        trade_rows.extend(
                            {**context, **t, "symbol": symbols[t["symbol"]]} for t in trades
                        )
                        event_rows.extend(
                            {**context, **e, "symbol": symbols[e["stock"]]} for e in events
                        )
                        decision_rows.extend({**context, **d} for d in decisions)
        print(
            window["label"],
            "all",
            len(candidates),
            "candidates and cost controls complete",
            flush=True,
        )
    for name, rows in [
        ("summary", summaries),
        ("daily_nav", daily_rows),
        ("trades_10bps", trade_rows),
        ("events_10bps", event_rows),
        ("decisions_10bps", decision_rows),
        ("index_window_returns", index_rows),
    ]:
        pd.DataFrame(rows).to_csv(result_dir / (name + ".csv"), index=False, encoding="utf-8-sig")
    inputs = [
        locked_path,
        config_path,
        target_path,
        factor_path,
        status_path,
        market,
        level_path,
        Path(__file__),
        Path(__file__).with_name("rule_account_replay.py"),
        Path(__file__).with_name("prepare_market_weight_batch.py"),
        Path(__file__).with_name("research_resources.py"),
        ROOT / "config/default.toml",
    ]
    checks.update(
        {
            "technical_status": "PASS",
            "cohort": args.cohort,
            "windows": len(windows),
            "candidates": len(candidates),
            "strategy_accounts": len(windows) * len(candidates) * 3,
            "account_rows": len(summaries),
            "baseline_max_return_difference": max(differences) if differences else None,
            "factor_columns_read": list(READ_COLUMNS),
            "status_columns_read": ["day", "symbol", "isST", "tradestatus"],
            "financial_industry_inputs": [],
            "future_labels_outside_development_windows_read": False,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "input_receipts": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
        }
    )
    (result_dir / "qa.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: checks[k]
                for k in [
                    "technical_status",
                    "strategy_accounts",
                    "account_rows",
                    "baseline_max_return_difference",
                    "elapsed_seconds",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
