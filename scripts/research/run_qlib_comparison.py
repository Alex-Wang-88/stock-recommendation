"""Replay frozen external model rankings with exactly the same account constraints."""

from __future__ import annotations

import json
import time

from qlib_external_sources import OUT, ROOT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import duckdb
    import numpy as np
    import pandas as pd
    from account_replay import drawdown
    from prepare_market_weight_batch import READ_COLUMNS, pool_frame, score_profile, sha256
    from rule_account_replay import replay_rules, self_checks

    started = time.monotonic()
    config = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    fit_qa = json.loads((OUT / "fit_qa.json").read_text(encoding="utf-8"))
    assert sha256(OUT / "plan_locked.json") == fit_qa["plan_sha256"]
    assert sha256(OUT / "predictions.parquet") == fit_qa["prediction_sha256"]
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    members = pd.read_parquet(OUT / "membership.parquet")
    previous = ROOT / "logs/market-direction-study-20261015"
    factors = pd.read_parquet(previous / "research_factors.parquet", columns=list(READ_COLUMNS))
    factors["symbol"] = factors.symbol.astype(str).str.zfill(6)
    factors["day"] = factors.day.astype(str).str[:10]
    symbols_wanted = sorted(members.symbol.unique())
    status = pd.read_parquet(
        previous / "execution_status_research.parquet",
        columns=["day", "symbol", "isST", "tradestatus"],
    )
    status["symbol"] = status.symbol.astype(str).str.zfill(6)
    status = status.loc[status.symbol.isin(symbols_wanted)].copy()
    days = sorted({d for w in config["windows"] for d in w["days"] + w["signal_dates"]})
    market = ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet"
    assert sha256(market) == config["market_sha256"]
    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    con.register("wanted_days", pd.DataFrame({"day": days}))
    con.register("wanted_symbols", pd.DataFrame({"symbol": symbols_wanted}))
    prices = con.execute(
        "select b.symbol,cast(b.trade_date as date)::varchar as day,"
        "b.open,b.high,b.low,b.close,b.pre_close,b.volume "
        "from read_parquet(?) b join wanted_days d "
        "on cast(b.trade_date as date)=cast(d.day as date) join wanted_symbols using(symbol)",
        [market.as_posix()],
    ).df()
    con.close()
    assert not prices.duplicated(["day", "symbol"]).any()
    grid = prices.pivot(index="day", columns="symbol", values="close")
    symbols, trading_days = grid.columns.tolist(), grid.index.tolist()
    symbol_index = {s: i for i, s in enumerate(symbols)}
    day_index = {d: i for i, d in enumerate(trading_days)}
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
    levels = (
        pd.read_parquet(ROOT / "logs/market-rule-optimization-20261014/index_data/levels.parquet")
        .set_index(["code", "day"])
        .sort_index()
    )
    market_vol = levels.loc["000985", "close"].pct_change().rolling(20).std(ddof=0) * np.sqrt(252)
    suite = json.loads(
        (ROOT / "config/research/market_only_weights.json").read_text(encoding="utf-8")
    )
    profile = next(p for p in suite["profiles"] if p["id"] == "P4_equal_categories")
    summaries, daily, trades_out, events_out, coverage, snapshots, decisions_out = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    reference_cache = {}
    index_pairs = [
        ("000300", "H00300"),
        ("000905", "H00905"),
        ("000852", "H00852"),
        ("000985", "H00985"),
    ]
    for window in config["windows"]:
        benchmarks = {}
        for price_code, tr_code in index_pairs:
            price = levels.loc[price_code].reindex(window["days"])
            total = levels.loc[tr_code].reindex(window["days"])
            assert price.open.gt(0).all() and price.close.gt(0).all() and total.close.gt(0).all()
            benchmarks[price_code] = float(price.close.iloc[-1] / price.open.iloc[0] - 1)
            benchmarks[tr_code] = float(total.close.iloc[-1] / total.close.iloc[0] - 1)
        for pool in config["pools"]:
            collections = {scorer: {} for scorer in config["scorers"]}
            for signal in window["signal_dates"]:
                pool_members = members.loc[(members.pool == pool) & (members.day == signal)]
                frame, _ = pool_frame(
                    factors.loc[factors.day == signal], "broader_liquid", config["common"]
                )
                frame = frame.loc[frame.symbol.isin(pool_members.symbol)].copy()
                assert frame.symbol.isin(symbol_index).all()
                frame = frame.sort_values("symbol")
                predicted = predictions.loc[
                    (predictions.pool == pool)
                    & (predictions.window == window["label"])
                    & (predictions.day == signal)
                ]
                assert set(predicted.symbol) == set(frame.symbol), (pool, signal)
                merged = frame.merge(
                    predicted[["symbol", "Lasso", "LightGBM"]],
                    on="symbol",
                    validate="one_to_one",
                )
                for scorer in config["scorers"]:
                    ranked = (
                        score_profile(frame, profile)
                        if scorer == "P4"
                        else merged.sort_values([scorer, "symbol"], ascending=[False, True])
                    )
                    collections[scorer][signal] = {
                        "pool": frame.symbol.tolist(),
                        "ranking": ranked.symbol.tolist(),
                        "exposure": 1.0,
                    }
                    snapshots.append(
                        {
                            "pool": pool,
                            "window": window["label"],
                            "signal": signal,
                            "scorer": scorer,
                            "ranking": ranked.symbol.tolist(),
                        }
                    )
                coverage.append(
                    {
                        "pool": pool,
                        "window": window["label"],
                        "signal": signal,
                        "member_count": len(pool_members),
                        "eligible": len(frame),
                        "known_prices": int(
                            pool_members.symbol.isin(
                                prices.loc[prices.day == signal, "symbol"]
                            ).sum()
                        ),
                        "under_25": len(frame) < 25,
                        "membership_update": pool_members.updateDate.max(),
                    }
                )
            for mode, target in config["exposure_modes"].items():
                for scorer, collection in collections.items():
                    for signal, context in collection.items():
                        context["exposure"] = (
                            1.0
                            if target is None
                            else min(1.0, target / float(market_vol.loc[signal]))
                        )
                    candidate = config["portfolio"]
                    for cost in config["cost_bps"]:
                        reference_key = (pool, window["label"], mode, cost)
                        if reference_key not in reference_cache:
                            reference_cache[reference_key] = replay_rules(
                                window,
                                collection,
                                arrays,
                                symbol_index,
                                day_index,
                                cost,
                                candidate,
                                reference=True,
                            )
                        reference = reference_cache[reference_key]
                        strategy = replay_rules(
                            window, collection, arrays, symbol_index, day_index, cost, candidate
                        )
                        results = [(scorer, strategy)]
                        if scorer == "P4":
                            results.append(("PoolEW", reference))
                        for model_name, result in results:
                            path, trades, events, residual, decisions = result
                            identity = {
                                "pool": pool,
                                "window": window["label"],
                                "scorer": model_name,
                                "mode": mode,
                                "cost_bps": cost,
                            }
                            row = {
                                **identity,
                                "start": window["start"],
                                "end": window["end"],
                                "return": float(path[-1] - 1),
                                "return_close_close": float(path[-1] / path[0] - 1),
                                "max_drawdown": drawdown(path),
                                "fees": sum(t["fee"] for t in trades),
                                "trade_notional": sum(t["notional"] for t in trades),
                                "orders": len(trades),
                                "residual_positions": residual,
                                "excess_pool": float(path[-1] - reference[0][-1]),
                                "mean_target_exposure": float(
                                    np.mean([d["exposure_target"] for d in decisions])
                                ),
                            }
                            for price_code, tr_code in index_pairs:
                                row["excess_" + price_code] = row["return"] - benchmarks[price_code]
                                row["excess_" + tr_code] = (
                                    row["return_close_close"] - benchmarks[tr_code]
                                )
                            summaries.append(row)
                            daily.extend(
                                {**identity, "day": day, "nav": float(value)}
                                for day, value in zip(window["days"], path, strict=True)
                            )
                            if cost == 10:
                                trades_out.extend(
                                    {**identity, **t, "symbol": symbols[t["symbol"]]}
                                    for t in trades
                                )
                                events_out.extend(
                                    {**identity, **e, "symbol": symbols[e["stock"]]} for e in events
                                )
                                decisions_out.extend({**identity, **d} for d in decisions)
        print("Account replay complete:", window["label"], flush=True)
    (OUT / "rankings_locked.json").write_text(json.dumps(snapshots), encoding="utf-8")
    for name, rows in [
        ("summary", summaries),
        ("daily_nav", daily),
        ("trades_10bps", trades_out),
        ("events_10bps", events_out),
        ("coverage", coverage),
        ("decisions_10bps", decisions_out),
    ]:
        pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    expected = 2 * 15 * 2 * 3 * 4
    assert len(summaries) == expected
    checks = self_checks()
    checks.update(
        {
            "technical_status": "PASS",
            "resource": RESOURCE_CONTEXT,
            "strategy_accounts": 2 * 15 * 2 * 3 * 3,
            "reference_accounts": len(reference_cache),
            "account_rows": len(summaries),
            "all_scorers_share_pool_and_execution": True,
            "plan_sha256": sha256(OUT / "plan_locked.json"),
            "source_model_prediction_sha256": fit_qa["prediction_sha256"],
            "execution_engine_sha256": sha256(ROOT / "scripts/research/rule_account_replay.py"),
            "production_config_sha256": sha256(ROOT / "config/default.toml"),
            "elapsed_seconds": time.monotonic() - started,
            "fresh_out_of_sample": False,
            "financial_industry_inputs": [],
        }
    )
    (OUT / "replay_qa.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("All accounts complete", len(summaries), flush=True)


if __name__ == "__main__":
    main()
