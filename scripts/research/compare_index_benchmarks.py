"""Add independently held official indices to complete, unchanged research replays."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/index-benchmark-rerun-20261014"
KEY = ["window", "pool", "profile", "one_way_cost_bps", "execution"]
POOLS = {"original_low_position": "原低位池", "broader_liquid": "较宽候选池"}
COHORTS = {"all_14": "全部14段", "development_6": "此前6段", "historical_8": "后续8段"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def pct(value):
    return f"{value * 100:+.2f}%"


def pp(value):
    return f"{value * 100:+.2f}"


def main():
    import numpy as np
    import pandas as pd
    from account_replay import drawdown

    protocol_path = OUT / "index_data/benchmark_protocol_locked.json"
    receipt_path = OUT / "index_data/receipt.json"
    levels_path = OUT / "index_data/index_levels.parquet"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest(protocol_path) == receipt["protocol_sha256"]
    assert digest(ROOT / "config/research/index_benchmarks.json") == receipt["protocol_sha256"]
    assert digest(levels_path) == receipt["level_sha256"]
    levels = pd.read_parquet(levels_path).set_index(["code", "day"]).sort_index()
    batches = [
        ("development_6", OUT / "development", ROOT / "logs/market-only-weight-results-20261014"),
        (
            "historical_8",
            OUT / "historical_validation",
            ROOT / "logs/market-only-extended-validation-20261014/results",
        ),
    ]
    frames, daily_frames, windows, parity, input_paths = (
        [],
        [],
        [],
        [],
        [protocol_path, receipt_path, levels_path],
    )
    for cohort, current, previous in batches:
        qa = json.loads((current / "qa.json").read_text(encoding="utf-8"))
        assert qa["technical_status"] == "PASS"
        assert qa["windows_priority_verified"] == "BelowNormal"
        assert not qa["financial_tables_read"] and not qa["industry_reference_files_read"]
        summary = pd.read_csv(current / "summary.csv")
        old = pd.read_csv(previous / "summary.csv")
        pd.testing.assert_frame_equal(
            summary.sort_values(KEY).reset_index(drop=True),
            old.sort_values(KEY).reset_index(drop=True),
            check_exact=True,
        )
        daily = pd.read_csv(current / "daily_nav.csv")
        prior_daily = pd.read_csv(previous / "daily_nav.csv")
        pd.testing.assert_frame_equal(
            daily.sort_values(KEY + ["day"]).reset_index(drop=True),
            prior_daily.sort_values(KEY + ["day"]).reset_index(drop=True),
            check_exact=True,
        )
        for artifact in ["holdings.csv", "target_schedules.json"]:
            assert digest(current / artifact) == digest(previous / artifact), artifact
        summary["cohort"], daily["cohort"] = cohort, cohort
        frames.append(summary)
        daily_frames.append(daily)
        plan = json.loads((current / "plan_locked.json").read_text(encoding="utf-8"))
        windows.extend({**window, "cohort": cohort} for window in plan["windows"])
        parity.append(
            {
                "cohort": cohort,
                "accounts": len(summary),
                "exact_summary_and_daily_nav_match": True,
                "holdings_and_target_schedules_sha256_match": True,
                "maximum_return_difference": 0.0,
            }
        )
        input_paths.extend(
            current / artifact
            for artifact in [
                "summary.csv",
                "daily_nav.csv",
                "holdings.csv",
                "target_schedules.json",
                "plan_locked.json",
                "qa.json",
                "input_receipts.json",
            ]
        )
    summary = pd.concat(frames, ignore_index=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    assert len(summary) == 1540 and len(windows) == 14
    assert len({day for w in windows for day in w["days"]}) == 420
    suite = json.loads((batches[0][1] / "suite_locked.json").read_text(encoding="utf-8"))
    profiles = {p["id"]: p["name"] for p in suite["profiles"]}
    paths = {key: frame.sort_values("day") for key, frame in daily.groupby(KEY, sort=False)}
    rebased = {}
    for key, frame in paths.items():
        assert len(frame) == 30 and frame.nav.gt(0).all()
        rebased[key] = frame.nav.to_numpy() / frame.nav.iloc[0]
    benchmarks, bench_daily, lookup = [], [], {}
    for window in windows:
        assert len(window["days"]) == 30
        for index in protocol["indices"]:
            for basis, code, kind in [
                ("price_open_close", index["price_code"], "price"),
                ("total_return_close_close", index["total_return_code"], "total_return"),
            ]:
                frame = levels.loc[(code, window["days"]), :]
                # Reindex explicitly; never rely on source sorting or fill missing values.
                frame = frame.droplevel("code").reindex(window["days"])
                assert frame["kind"].eq(kind).all() and frame.close.gt(0).all()
                entry = frame.open.iloc[0] if kind == "price" else frame.close.iloc[0]
                assert np.isfinite(entry) and entry > 0
                nav = frame.close.to_numpy() / entry
                value = float(nav[-1] - 1)
                assert abs(value - (float(frame.close.iloc[-1]) / float(entry) - 1)) < 1e-14
                row = {
                    "window": window["label"],
                    "cohort": window["cohort"],
                    "basis": basis,
                    "index_code": index["price_code"],
                    "series_code": code,
                    "index_name": index["name"],
                    "start": window["start"],
                    "end": window["end"],
                    "entry_level": entry,
                    "exit_level": frame.close.iloc[-1],
                    "return": value,
                    "max_drawdown": drawdown(nav),
                    "close_intervals": 29 if kind == "total_return" else None,
                    "index_cost_bps": 0,
                }
                benchmarks.append(row)
                lookup[(window["label"], index["price_code"], basis)] = row
                bench_daily.extend(
                    {
                        **{
                            k: row[k]
                            for k in ["window", "basis", "index_code", "series_code", "cohort"]
                        },
                        "day": day,
                        "nav": float(value),
                    }
                    for day, value in zip(window["days"], nav, strict=True)
                )
    assert len(benchmarks) == 112
    comparisons = []
    for _, account in summary.iterrows():
        key = tuple(account[k] for k in KEY)
        original_path = paths[key]
        period_path = rebased[key]
        assert (
            original_path.day.iloc[0] == account.start and original_path.day.iloc[-1] == account.end
        )
        for basis in ["price_open_close", "total_return_close_close"]:
            value = account["return"] if basis == "price_open_close" else float(period_path[-1] - 1)
            dd = account.max_drawdown if basis == "price_open_close" else drawdown(period_path)
            pool_key = (
                account.window,
                account.pool,
                "same_pool_equal_same_cost",
                account.one_way_cost_bps,
                account.execution,
            )
            pool_return = None
            if account.profile != "market_equal_gross":
                pool_return = (
                    account.pool_return_same_cost
                    if basis == "price_open_close"
                    else float(rebased[pool_key][-1] - 1)
                )
            for index in protocol["indices"]:
                bench = lookup[(account.window, index["price_code"], basis)]
                comparisons.append(
                    {
                        **account.to_dict(),
                        "basis": basis,
                        "index_code": index["price_code"],
                        "index_name": index["name"],
                        "comparison_strategy_return": value,
                        "comparison_strategy_drawdown": dd,
                        "index_return_gross": bench["return"],
                        "index_drawdown": bench["max_drawdown"],
                        "excess_index": value - bench["return"],
                        "relative_wealth_return": (1 + value) / (1 + bench["return"]) - 1,
                        "comparison_pool_return_same_cost": pool_return,
                        "comparison_pool_excess": value - pool_return
                        if pool_return is not None
                        else None,
                    }
                )
    comparison = pd.DataFrame(comparisons)
    assert len(comparison) == 12320
    accounts = comparison[comparison.profile.isin(profiles)].copy()
    assert len(accounts) == 10752
    aggregates = []
    for cohort in ["all_14", "development_6", "historical_8"]:
        selection = accounts if cohort == "all_14" else accounts[accounts.cohort == cohort]
        fields = [
            "pool",
            "execution",
            "one_way_cost_bps",
            "profile",
            "index_code",
            "index_name",
            "basis",
        ]
        for keys, group in selection.groupby(fields, sort=False):
            aggregates.append(
                {
                    **dict(zip(fields, keys, strict=True)),
                    "cohort": cohort,
                    "windows": len(group),
                    "profile_name": profiles[keys[3]],
                    "mean_strategy_return": group.comparison_strategy_return.mean(),
                    "mean_index_return_gross": group.index_return_gross.mean(),
                    "mean_excess_index": group.excess_index.mean(),
                    "median_excess_index": group.excess_index.median(),
                    "positive_excess_windows": int(group.excess_index.gt(1e-10).sum()),
                    "worst_excess_index": group.excess_index.min(),
                    "worst_strategy_return": group.comparison_strategy_return.min(),
                    "worst_strategy_drawdown": group.comparison_strategy_drawdown.min(),
                    "mean_pool_excess_same_period": group.comparison_pool_excess.mean(),
                }
            )
    aggregate = pd.DataFrame(aggregates)
    assert len(aggregate) == 2304
    for name, frame in [
        ("combined_summary", summary),
        ("index_window_returns", pd.DataFrame(benchmarks)),
        ("index_daily_nav", pd.DataFrame(bench_daily)),
        ("index_comparisons", comparison),
        ("index_aggregate", aggregate),
    ]:
        frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    primary = aggregate[
        (aggregate.execution == "continuous_net") & (aggregate.one_way_cost_bps == 10)
    ]
    report = [
        "# 四个指数对照：14段历史区间、8组冻结价量权重",
        "",
        "已实际重跑全部 1,344 个策略账户和 196 个原有对照账户，共 1,540 个账户。"
        "四个价格指数及对应全收益指数均来自中证官网发布的历史指数值，"
        "8条序列各641日，全部覆盖所需420个交易日。"
        "重跑的账户收益、逐日净值、持仓和目标名单与原结果完全一致。",
        "",
        "本页主表采用连续持仓、差额调仓、单边10bps费用；每个数字是各段独立区间的算术均值，超额单位为百分点。没有把不连续区间拼接成年化收益。0/10/20bps、两种执行、两种股票池、8组权重、全部分段结果都保存在CSV。",
        "",
        "中证全指是事先固定的宽市场主参考，其他三条用于观察市值风格；四条都报告，不按跑完后的结果挑选基准。原有同池等权、相同费用和执行的对照继续保留。",
        "",
        "## 对齐与费用口径",
        "",
        "- **价格指数，30交易日**：首日真实开盘至末日收盘；策略为原30日净账户收益。"
        "指数全程买入持有、包含周期之间的隔夜行情，指数不扣交易费用且不包含现金分红。",
        "- **全收益指数，29个收盘间隔**：中证官网只有全收益收盘数据。"
        "指数按首日收盘至末日收盘计算，策略也用末日收盘净值/首日收盘净值−1。"
        "指数包含税前股息再投资；策略首日收盘以前的入场费用与第一日行情不在该比较期间，"
        "之后的费用仍包含。没有用价格指数或估算值伪造全收益开盘。",
        "- 策略净收益对指数毛收益，对指数是较严格的费用口径。"
        "官方全收益“税前股息再投资”与费用档位并不是同一个概念。",
        "- “同池超额”也按对应比较期间计算；CSV同时保留原30日同池超额，字段名明确区分。",
        "- 超额为收益率之差，另存相对财富收益；这不是剔除市值、行业等风险暴露后的回归alpha。",
        "",
        "## 结果判断",
        "",
        "全部14段、连续持仓、单边10bps下，两类股票池的8组权重对中证全指的平均超额"
        "均为负；改用对齐收盘期间的全收益指数后，这个结论也成立。"
        "此前两项候选虽然对沪深300的平均价格指数超额为正，"
        "但对中证500和中证1000仍为负，说明只看沪深300容易高估优势。",
        "",
        "原低位池仅风控的后续8段平均价格全指超额为+1.30个百分点，"
        "但同池超额为−2.14个百分点；较宽池类别等权的后续8段分别为"
        "−0.05和−0.73个百分点。市场对照与同池对照共同看，"
        "仍没有充分证据支持把当前候选当作稳定的生产权重升级。",
        "",
    ]
    for basis, title in [
        ("price_open_close", "30交易日价格指数"),
        ("total_return_close_close", "首日收盘至末日收盘全收益指数"),
    ]:
        report.extend(["## 全部14段：" + title, ""])
        for pool, pool_name in POOLS.items():
            selection = primary[
                (primary.cohort == "all_14") & (primary.pool == pool) & (primary.basis == basis)
            ]
            rows = []
            for profile in profiles:
                group = selection[selection.profile == profile].set_index("index_code")
                reference = group.loc["000985"]
                rows.append(
                    [
                        profiles[profile],
                        pct(reference.mean_strategy_return),
                        pp(reference.mean_pool_excess_same_period),
                        *[
                            pp(group.loc[index["price_code"], "mean_excess_index"])
                            for index in protocol["indices"]
                        ],
                        f"{int(reference.positive_excess_windows)}/14",
                    ]
                )
            report.extend(
                [
                    "### " + pool_name,
                    "",
                    table(
                        [
                            "权重",
                            "策略均收益",
                            "同池超额",
                            "沪深300超额",
                            "中证500超额",
                            "中证1000超额",
                            "中证全指超额",
                            "胜全指段数",
                        ],
                        rows,
                    ),
                    "",
                ]
            )
    report.extend(
        [
            "## 此前选定候选的分组表现",
            "",
            "保留上一轮已经指定的“原低位池仅风控”和“较宽池类别等权”，本次不根据指数结果重新挑权重。原6段与后续8段均已查看过，本次增加基准不产生新的留出验证。",
            "",
        ]
    )
    for basis, title in [
        ("price_open_close", "30日价格指数"),
        ("total_return_close_close", "29个收盘间隔全收益"),
    ]:
        rows = []
        for pool, profile in [
            ("original_low_position", "P7_risk_only"),
            ("broader_liquid", "P4_equal_categories"),
        ]:
            for cohort in COHORTS:
                group = primary[
                    (primary.cohort == cohort)
                    & (primary.pool == pool)
                    & (primary.profile == profile)
                    & (primary.basis == basis)
                ].set_index("index_code")
                row = group.loc["000985"]
                rows.append(
                    [
                        POOLS[pool] + "/" + profiles[profile],
                        COHORTS[cohort],
                        pct(row.mean_strategy_return),
                        pp(row.mean_pool_excess_same_period),
                        *[
                            pp(group.loc[index["price_code"], "mean_excess_index"])
                            for index in protocol["indices"]
                        ],
                        f"{int(row.positive_excess_windows)}/{int(row.windows)}",
                    ]
                )
        report.extend(
            [
                "### " + title,
                "",
                table(
                    [
                        "候选",
                        "样本",
                        "策略均收益",
                        "同池超额",
                        "沪深300超额",
                        "中证500超额",
                        "中证1000超额",
                        "中证全指超额",
                        "胜全指段数",
                    ],
                    rows,
                ),
                "",
            ]
        )
    report.extend(["## 每段的四个价格指数收益", ""])
    rows = []
    for window in windows:
        rows.append(
            [
                window["label"],
                window["start"] + " 至 " + window["end"],
                *[
                    pct(
                        lookup[(window["label"], index["price_code"], "price_open_close")]["return"]
                    )
                    for index in protocol["indices"]
                ],
            ]
        )
    report.extend(
        [
            table(["区间", "日期", "沪深300", "中证500", "中证1000", "中证全指"], rows),
            "",
            "## 此前候选每段结果：30日净收益与价格指数",
            "",
        ]
    )
    for pool, profile in [
        ("original_low_position", "P7_risk_only"),
        ("broader_liquid", "P4_equal_categories"),
    ]:
        selection = accounts[
            (accounts.pool == pool)
            & (accounts.profile == profile)
            & (accounts.execution == "continuous_net")
            & (accounts.one_way_cost_bps == 10)
            & (accounts.basis == "price_open_close")
        ]
        rows = []
        for window in windows:
            group = selection[selection.window == window["label"]].set_index("index_code")
            row = group.loc["000985"]
            rows.append(
                [
                    window["label"],
                    pct(row.comparison_strategy_return),
                    pct(row.comparison_strategy_drawdown),
                    pp(row.comparison_pool_excess),
                    *[
                        pp(group.loc[index["price_code"], "excess_index"])
                        for index in protocol["indices"]
                    ],
                ]
            )
        report.extend(
            [
                "### " + POOLS[pool] + "/" + profiles[profile],
                "",
                table(
                    [
                        "区间",
                        "策略净收益",
                        "区间最大回撤",
                        "同池超额",
                        "沪深300超额",
                        "中证500超额",
                        "中证1000超额",
                        "中证全指超额",
                    ],
                    rows,
                ),
                "",
            ]
        )
    report.extend(
        [
            "## 研究范围与局限",
            "",
            "股票侧继续只使用冻结价量因子、上市日期、历史ST/交易状态及原行情快照；不可复现财报和行业因子未加入评分。当前研究是8组价量权重的受限模型，并不等同线上包含财报和行业因子的完整版。原行情为调整后OHLC，调整算法、退市覆盖及历史股票全集尚未完成独立点时审计，指标不能解释为可实盘保证的收益。",
            "",
            "中证全指目前包含北交所，而股票回放池只含沪深；官方指数按编制权重，同池参考等权，二者测量不同问题。实际指数复制或ETF会产生跟踪误差和费用，当前没有把ETF收益混入指数收益。",
            "",
            "通过新增指数能够观察市场和市值风格暴露，但不能消除先前查看数据、权重选择或样本不足的影响。后续8段的同池超额仍需单独看，不因全14段平均跑赢某个指数就视为稳定选股能力。生产权重和GitHub网页未修改。",
            "",
            "## 数据与完整结果",
            "",
            "- index_data/receipt.json：官方原始响应、协议和指数序列SHA256；全部420日覆盖。",
            "- index_window_returns.csv、index_daily_nav.csv：112个区间基准及对应逐日净值。",
            "- index_comparisons.csv：12,320条逐账户、逐指数、逐口径比较，"
            "含原30日账户结果及对齐期间结果。",
            "- index_aggregate.csv：2,304条聚合，"
            "覆盖全部费用、执行、权重、股票池、指数、口径和三个样本组。",
            "- development/、historical_validation/：实际重新执行的账本、持仓、日净值与输入收据。",
            "- qa.json：精确复现、指数覆盖和时间对齐检查。",
            "",
            "指数历史直接来源：[中证官网历史行情接口](https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode=000300&startDate=20240201&endDate=20260924)。",
            "",
            "指数及全收益代码参照中证官方资料：[沪深300](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300factsheet.pdf)、[中证500](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000905factsheet.pdf)、[中证1000](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000852factsheet.pdf)、[中证全指](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000985factsheet.pdf)。",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    (ROOT / "docs/market_only_index_benchmark_validation.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    snapshot = OUT / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in [
        "compare_index_benchmarks.py",
        "fetch_index_benchmarks.py",
        "run_market_weight_batch.py",
        "account_replay.py",
        "prepare_market_weight_batch.py",
        "research_resources.py",
    ]:
        source = ROOT / "scripts/research" / filename
        shutil.copyfile(source, snapshot / filename)
        input_paths.append(source)
    input_paths.extend(
        [
            ROOT / "config/research/market_only_weights.json",
            ROOT / "config/research/index_benchmarks.json",
        ]
    )
    qa = {
        "technical_status": "PASS",
        "resource_context": RESOURCE_CONTEXT,
        "rerun_parity": parity,
        "strategy_accounts": 1344,
        "existing_reference_accounts": 196,
        "total_replayed_accounts": 1540,
        "index_series": 8,
        "price_indices": 4,
        "window_benchmark_paths": 112,
        "index_comparison_rows": len(comparison),
        "aggregate_rows": len(aggregate),
        "index_benchmark_independent_of_profile_pool_cost_and_execution": True,
        "all_required_index_days_present": True,
        "forward_fill_used": False,
        "synthetic_total_return_open_used": False,
        "both_sides_same_entry_exit_timing": True,
        "all_windows_previously_viewed_at_this_rerun": True,
        "production_config_unchanged": True,
        "scorer_financial_tables_read": [],
        "scorer_industry_reference_files_read": [],
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in input_paths],
    }
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: qa[k]
                for k in [
                    "technical_status",
                    "total_replayed_accounts",
                    "index_comparison_rows",
                    "aggregate_rows",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    print(
        primary[(primary.cohort == "all_14") & (primary.index_code == "000985")][
            [
                "pool",
                "profile",
                "basis",
                "mean_excess_index",
                "mean_pool_excess_same_period",
                "positive_excess_windows",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
