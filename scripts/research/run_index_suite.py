"""Compare locked structural variants and retain literal old-account parity checks."""

from __future__ import annotations

import json
import time

from design_index_improvement import OUT, PREVIOUS, ROOT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
INDEX_PAIRS = [
    ("000300", "H00300"),
    ("000905", "H00905"),
    ("000852", "H00852"),
    ("000985", "H00985"),
]


def run_accounts(config, factors, predictions, prices, status, levels, candidates, folder, cohort):
    import pandas as pd
    from account_replay import drawdown
    from index_improvement_data import historical_growth_flags, price_arrays
    from prepare_market_weight_batch import pool_frame, score_profile, sha256

    if cohort == "development":
        from rule_account_replay import replay_rules
    else:
        from index_account_replay import replay_rules

    started = time.monotonic()
    folder.mkdir(exist_ok=True, parents=True)
    arrays, symbol_index, day_index = price_arrays(prices, status)
    suite = json.loads(
        (ROOT / "config/research/market_only_weights.json").read_text(encoding="utf-8")
    )
    profile = next(p for p in suite["profiles"] if p["id"] == "P4_equal_categories")
    levels = levels.set_index(["code", "day"]).sort_index()
    previous = pd.read_csv(PREVIOUS / "summary.csv") if cohort == "development" else None
    summaries, paths, decisions_out, trades_out, coverage, ranks, parity = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    cache = {}
    for window in config["windows"]:
        arrays["growth_board"] = historical_growth_flags(
            symbol_index, window["start"], window["end"], legacy_codes=(cohort == "development")
        )
        index_returns, benchmark_dd = {}, {}
        for price_code, tr_code in INDEX_PAIRS:
            price = levels.loc[price_code].reindex(window["days"])
            total = levels.loc[tr_code].reindex(window["days"])
            assert price.open.gt(0).all() and price.close.gt(0).all() and total.close.gt(0).all()
            index_returns[price_code] = float(price.close.iloc[-1] / price.open.iloc[0] - 1)
            index_returns[tr_code] = float(total.close.iloc[-1] / total.close.iloc[0] - 1)
            benchmark_dd[price_code] = drawdown(price.close.to_numpy() / price.open.iloc[0])
        for pool in config["pools"]:
            applicable = [c for c in candidates if c.get("pool", pool) == pool]
            for candidate in applicable:
                contexts = {}
                for signal in window["signal_dates"]:
                    rules = {**config["common"], "volume_trend_20_min": candidate["volume_min"]}
                    frame, _ = pool_frame(
                        factors.loc[(factors.pool == pool) & (factors.day == signal)],
                        "broader_liquid",
                        rules,
                    )
                    frame = frame.loc[frame.symbol.isin(symbol_index)].sort_values("symbol")
                    assert len(frame), (pool, signal, candidate["id"])
                    if candidate["scorer"] == "P4":
                        ranked = score_profile(frame, profile)
                    else:
                        scored = predictions.loc[
                            (predictions.pool == pool)
                            & (predictions.window == window["label"])
                            & (predictions.day == signal)
                        ]
                        merged = frame.merge(
                            scored[["symbol", candidate["scorer"]]],
                            on="symbol",
                            validate="one_to_one",
                        )
                        assert len(merged) == len(frame)
                        ranked = merged.sort_values(
                            [candidate["scorer"], "symbol"], ascending=[False, True]
                        )
                    contexts[signal] = {
                        "pool": frame.symbol.tolist(),
                        "ranking": ranked.symbol.tolist(),
                        "exposure": 1.0,
                    }
                    coverage.append(
                        {
                            "cohort": cohort,
                            "pool": pool,
                            "candidate": candidate["id"],
                            "window": window["label"],
                            "signal": signal,
                            "eligible": len(frame),
                            "under_target": len(frame) < candidate["positions"],
                        }
                    )
                    ranks.append(
                        {
                            "pool": pool,
                            "window": window["label"],
                            "candidate": candidate["id"],
                            "signal": signal,
                            "ranking": ranked.symbol.tolist(),
                        }
                    )
                for cost in config["cost_bps"]:
                    key = (
                        pool,
                        window["label"],
                        candidate["volume_min"],
                        candidate["rebalance_days"],
                        cost,
                    )
                    if key not in cache:
                        cache[key] = replay_rules(
                            window,
                            contexts,
                            arrays,
                            symbol_index,
                            day_index,
                            cost,
                            candidate,
                            reference=True,
                        )
                    result = replay_rules(
                        window, contexts, arrays, symbol_index, day_index, cost, candidate
                    )
                    results = [("strategy", result)]
                    # One same-pool control row for each strategy makes pairwise audit explicit.
                    results.append(("same_pool", cache[key]))
                    for account, replay in results:
                        nav, trades, events, residual, decisions = replay
                        identity = {
                            "cohort": cohort,
                            "pool": pool,
                            "window": window["label"],
                            "candidate": candidate["id"],
                            "family": candidate["family"],
                            "scorer": candidate["scorer"],
                            "cost_bps": cost,
                            "account": account,
                        }
                        own = "000300" if pool == "csi300" else "000905"
                        row = {
                            **identity,
                            "start": window["start"],
                            "end": window["end"],
                            "return": float(nav[-1] - 1),
                            "max_drawdown": drawdown(nav),
                            "return_close_close": float(nav[-1] / nav[0] - 1),
                            "excess_pool": float(nav[-1] - cache[key][0][-1]),
                            "fees": sum(t["fee"] for t in trades),
                            "notional": sum(t["notional"] for t in trades),
                            "orders": len(trades),
                            "residual_positions": residual,
                            "relative_drawdown": drawdown(nav) - benchmark_dd[own],
                            "unfilled_exit_count": sum(
                                e["action"] == "unfilled_exit_retained" for e in events
                            ),
                        }
                        for price_code, tr_code in INDEX_PAIRS:
                            row["excess_" + price_code] = row["return"] - index_returns[price_code]
                            row["excess_" + tr_code] = (
                                row["return_close_close"] - index_returns[tr_code]
                            )
                        summaries.append(row)
                        paths.extend(
                            {**identity, "day": day, "nav": float(value)}
                            for day, value in zip(window["days"], nav, strict=True)
                        )
                        if account == "strategy" and cost == 10:
                            trades_out.extend(
                                {**identity, **t, "symbol": list(symbol_index)[t["symbol"]]}
                                for t in trades
                            )
                            decisions_out.extend({**identity, **d} for d in decisions)
                        if candidate["family"] == "F0" and previous is not None:
                            old_account = candidate["scorer"] if account == "strategy" else "PoolEW"
                            old = previous.loc[
                                (previous.pool == pool)
                                & (previous.window == window["label"])
                                & (previous.scorer == old_account)
                                & (previous["mode"] == "full")
                                & (previous.cost_bps == cost)
                            ]
                            assert len(old) == 1
                            error = abs(row["return"] - old.iloc[0]["return"])
                            assert error < 1e-10, (candidate["id"], window["label"], error)
                            parity.append(error)
            print(cohort, window["label"], pool, "complete", flush=True)
    for name, rows in [
        ("summary", summaries),
        ("daily_nav", paths),
        ("trades_10bps", trades_out),
        ("decisions_10bps", decisions_out),
        ("coverage", coverage),
    ]:
        pd.DataFrame(rows).to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig")
    (folder / "rankings.json").write_text(json.dumps(ranks), encoding="utf-8")
    qa = {
        "technical_status": "PASS",
        "cohort": cohort,
        "strategy_accounts": len(summaries) // 2,
        "unique_reference_accounts": len(cache),
        "reported_rows": len(summaries),
        "baseline_max_return_difference": max(parity) if parity else None,
        "resource": RESOURCE_CONTEXT,
        "elapsed_seconds": time.monotonic() - started,
        "plan_sha256": sha256(OUT / "plan_locked.json"),
        "financial_industry_inputs": [],
    }
    (folder / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


def main():
    from index_improvement_data import load_development

    data = load_development()
    run_accounts(*data, data[0]["candidates"], OUT / "development", "development")


if __name__ == "__main__":
    main()
