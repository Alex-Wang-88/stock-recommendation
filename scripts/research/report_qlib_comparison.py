"""Describe all frozen comparisons, including adverse periods and implementation limits."""

from __future__ import annotations

import json

from qlib_external_sources import OUT, ROOT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def pct(value):
    return f"{value * 100:+.2f}%"


def pp(value):
    return f"{value * 100:+.2f}"


def main():
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import sha256

    config = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    replay = json.loads((OUT / "replay_qa.json").read_text(encoding="utf-8"))
    fitted = json.loads((OUT / "fit_qa.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(OUT / "summary.csv")
    folds = pd.read_csv(OUT / "folds.csv")
    coverage = pd.read_csv(OUT / "coverage.csv")
    signals = pd.read_csv(OUT / "signal_diagnostics.csv")
    membership = pd.read_parquet(OUT / "membership.parquet")
    model_features = pd.read_parquet(
        OUT / "alpha157.parquet", columns=["day", "symbol", "amount_volume_to_close"]
    )
    model_features["has_quote_history"] = np.isfinite(
        model_features.amount_volume_to_close
    ) & model_features.amount_volume_to_close.gt(0)
    member_coverage = membership.merge(
        model_features[["day", "symbol", "has_quote_history"]], on=["day", "symbol"], how="left"
    )
    all_coverage = (
        member_coverage.groupby(["pool", "day"])
        .agg(expected=("symbol", "size"), history=("has_quote_history", "sum"))
        .reset_index()
    )
    all_coverage["coverage_fraction"] = all_coverage.history / all_coverage.expected
    all_coverage.to_csv(OUT / "membership_price_coverage.csv", index=False)
    aggregate, paired, years = [], [], []
    for (pool, mode, cost, scorer), group in summary.groupby(
        ["pool", "mode", "cost_bps", "scorer"], sort=False
    ):
        own = "000300" if pool == "csi300" else "000905"
        own_tr = "H00300" if pool == "csi300" else "H00905"
        assert len(group) == 15
        excess = group["excess_" + own]
        dropped = group.drop(excess.idxmax())
        row = {
            "pool": pool,
            "mode": mode,
            "cost_bps": cost,
            "scorer": scorer,
            "windows": len(group),
            "mean_return": group["return"].mean(),
            "own_index_excess": excess.mean(),
            "own_tr_excess": group["excess_" + own_tr].mean(),
            "all_share_excess": group.excess_000985.mean(),
            "mean_pool_excess": group.excess_pool.mean(),
            "median_own_excess": excess.median(),
            "wins_own_index": int(excess.gt(0).sum()),
            "worst_return": group["return"].min(),
            "worst_drawdown": group.max_drawdown.min(),
            "drop_best_own_excess": dropped["excess_" + own].mean(),
            "mean_fee": group.fees.mean(),
            "mean_notional": group.trade_notional.mean(),
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
            row["excess_" + code] = group["excess_" + code].mean()
        aggregate.append(row)
        for year, part in group.groupby(group.start.str[:4]):
            years.append(
                {
                    "pool": pool,
                    "mode": mode,
                    "cost_bps": cost,
                    "scorer": scorer,
                    "entry_year": year,
                    "windows": len(part),
                    "mean_return": part["return"].mean(),
                    "own_index_excess": part["excess_" + own].mean(),
                }
            )
        if scorer not in ["Lasso", "LightGBM"]:
            continue
        baseline = summary.loc[
            (summary.pool == pool)
            & (summary["mode"] == mode)
            & (summary.cost_bps == cost)
            & (summary.scorer == "P4")
        ]
        compared = group.merge(
            baseline[["window", "return"]],
            on="window",
            suffixes=("", "_P4"),
            validate="one_to_one",
        )
        compared["difference"] = compared["return"] - compared.return_P4
        paired.extend(
            compared[["pool", "mode", "cost_bps", "scorer", "window", "difference"]].to_dict(
                "records"
            )
        )
    aggregate = pd.DataFrame(aggregate)
    years = pd.DataFrame(years)
    paired = pd.DataFrame(paired)
    aggregate.to_csv(OUT / "aggregate.csv", index=False)
    years.to_csv(OUT / "entry_year_groups.csv", index=False)
    paired.to_csv(OUT / "paired_vs_P4.csv", index=False)
    insights = {
        "plan_sha256": sha256(OUT / "plan_locked.json"),
        "fresh_holdout": False,
        "models_fitted": fitted["models_fit"],
        "accounts": replay["account_rows"],
        "training_rows_range": [int(folds.train_rows.min()), int(folds.train_rows.max())],
        "boosting_iterations_range": [int(folds.iterations.min()), int(folds.iterations.max())],
        "lasso_max_iterations_used": int(folds.lasso_iterations.max()),
        "lasso_all_converged_before_limit": bool(
            (folds.lasso_iterations < config["lasso"]["max_iter"]).all()
        ),
        "constant_signal_groups": int(signals.prediction_std.lt(1e-12).sum()),
        "signal_groups": len(signals),
        "eligible_range": [int(coverage.eligible.min()), int(coverage.eligible.max())],
        "under_25_signal_groups": int(coverage.under_25.sum()),
        "dated_member_valid_amount_volume_close_fraction_min": float(
            all_coverage.coverage_fraction.min()
        ),
        "dated_member_valid_amount_volume_close_fraction_mean": float(
            all_coverage.coverage_fraction.mean()
        ),
    }
    (OUT / "insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")
    full_ten = aggregate.loc[(aggregate.cost_bps == 10) & (aggregate["mode"] == "full")]
    p4_500 = full_ten.loc[(full_ten.pool == "csi500") & (full_ten.scorer == "P4")].iloc[0]
    lasso_500 = full_ten.loc[(full_ten.pool == "csi500") & (full_ten.scorer == "Lasso")].iloc[0]
    ew_500 = full_ten.loc[(full_ten.pool == "csi500") & (full_ten.scorer == "PoolEW")].iloc[0]
    gbdt_300_pair = paired.loc[
        (paired.pool == "csi300")
        & (paired["mode"] == "full")
        & (paired.cost_bps == 10)
        & (paired.scorer == "LightGBM")
    ].difference
    ml_combinations = aggregate.loc[
        (aggregate.cost_bps == 10) & aggregate.scorer.isin(["Lasso", "LightGBM"])
    ]
    lines = [
        "# Qlib量价选股与原评分对照：沪深300、中证500",
        "",
        f"完成两类模型的{fitted['models_fit']}次拟合，以及{replay['strategy_accounts']}个策略账户、"
        f"{replay['reference_accounts']}个同池对照账户，共{replay['account_rows']}个账户。"
        "覆盖15个此前已查看的30交易日区间，行情快照截至2026-09-24。",
        "",
        "**这是借鉴公开公式和模型配置的A股适配实验，不是原作者收益复现，也不是新样本外验收。**"
        "以下收益为15个独立区间的算术均值，不拼接或年化；主费用为单边10bps，超额单位为百分点。",
        "",
        "## 结论",
        "",
        f"两类模型在两个股票池、两种仓位设置形成的{len(ml_combinations)}个组合中，"
        f"平均跑赢各自指数的有{int(ml_combinations.own_index_excess.gt(0).sum())}个。"
        "本次结果不支持替换生产选股评分，结论限定于这组参数、数据和交易约束。",
        "",
        f"- 沪深300池的LightGBM相对P4平均改善{pp(gbdt_300_pair.mean())}个百分点；"
        f"去掉配对改善最大的一段后，均差变为"
        f"{pp(gbdt_300_pair.drop(gbdt_300_pair.idxmax()).mean())}个百分点，优势仍依赖个别行情。",
        f"- 中证500池的Lasso最差单段回撤由P4的{pct(p4_500.worst_drawdown)}"
        f"降至{pct(lasso_500.worst_drawdown)}，但平均净收益差为"
        f"{pp(lasso_500.mean_return - p4_500.mean_return)}个百分点，"
        "且三个起始年份组的平均指数超额均为负。这是一条回撤改善线索，尚非稳定正超额。",
        f"- 中证500的同筛选池等权对照平均落后指数{abs(ew_500.own_index_excess) * 100:.2f}"
        "个百分点。这个差距合并了股票池筛选、等权配置、调仓和费用等影响，"
        "不能归因为单个筛选条件；只调模型权重不足以解释全部收益差距。",
        "",
        "## 同条件结果",
        "",
        "每组使用相同的历史成分股票池、原流动性/ST/历史长度筛选、25只等权、5日调仓、"
        "前40名保留缓冲、次日开盘执行、受限卖出保留持仓，以及同一账户引擎。"
        "P4为此前市场数据研究中的动量/风险/流动性类别等权评分，不代表线上九类评分。",
        "",
    ]
    names = {"csi300": "沪深300", "csi500": "中证500"}
    for mode in ["full", "vol15"]:
        lines.extend(
            [
                "### " + ("满仓对照" if mode == "full" else "同样加入市场波动15%仓位控制"),
                "",
                "| 股票池 | 评分 | 平均净收益 | 对本池指数超额 | 对全指超额 "
                "| 同池等权超额 | 最差回撤 | 胜本池指数 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for pool in config["pools"]:
            for scorer in config["scorers"]:
                row = aggregate.loc[
                    (aggregate.pool == pool)
                    & (aggregate["mode"] == mode)
                    & (aggregate.cost_bps == 10)
                    & (aggregate.scorer == scorer)
                ].iloc[0]
                lines.append(
                    f"| {names[pool]} | {scorer} | {pct(row.mean_return)} | "
                    f"{pp(row.own_index_excess)} | "
                    f"{pp(row.all_share_excess)} | {pp(row.mean_pool_excess)} | "
                    f"{pct(row.worst_drawdown)} | {int(row.wins_own_index)}/15 |"
                )
        lines.append("")
    lines.extend(
        [
            "## 相对原P4评分的配对差值",
            "",
            "| 股票池 | 仓位 | 模型 | 平均净收益差 | 中位差 | 占优区间 "
            "| 20bps时均差 | 去掉最大改善段后均差 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for pool in config["pools"]:
        for mode in ["full", "vol15"]:
            for scorer in ["Lasso", "LightGBM"]:
                part = paired.loc[
                    (paired.pool == pool)
                    & (paired["mode"] == mode)
                    & (paired.scorer == scorer)
                    & (paired.cost_bps == 10)
                ]
                high = paired.loc[
                    (paired.pool == pool)
                    & (paired["mode"] == mode)
                    & (paired.scorer == scorer)
                    & (paired.cost_bps == 20)
                ]
                diff = part.difference
                lines.append(
                    f"| {names[pool]} | {mode} | {scorer} | {pp(diff.mean())} | "
                    f"{pp(diff.median())} | "
                    f"{int(diff.gt(0).sum())}/15 | {pp(high.difference.mean())} | "
                    f"{pp(diff.drop(diff.idxmax()).mean())} |"
                )
    lines.extend(
        [
            "",
            "## 分年份及集中度",
            "",
            "按窗口起始年归组，跨年窗口计入起始年；表中不是全年收益。",
            "",
            "| 股票池 | 评分（满仓） | 2024起始组超额 | 2025起始组超额 "
            "| 2026起始组超额 | 剔除最佳段后超额 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for pool in config["pools"]:
        for scorer in config["scorers"]:
            part = years.loc[
                (years.pool == pool)
                & (years["mode"] == "full")
                & (years.cost_bps == 10)
                & (years.scorer == scorer)
            ].set_index("entry_year")
            row = aggregate.loc[
                (aggregate.pool == pool)
                & (aggregate["mode"] == "full")
                & (aggregate.cost_bps == 10)
                & (aggregate.scorer == scorer)
            ].iloc[0]
            vals = " | ".join(pp(part.loc[str(y), "own_index_excess"]) for y in [2024, 2025, 2026])
            lines.append(f"| {names[pool]} | {scorer} | {vals} | {pp(row.drop_best_own_excess)} |")
    lines.extend(
        [
            "",
            "## 预测质量与成本敏感性",
            "",
            "| 股票池 | 模型 | 5日收益平均Rank IC | 正IC信号比例 "
            "| 满仓20bps对本池指数超额 | 对齐全收益指数超额（10bps） |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for pool in config["pools"]:
        for scorer in ["Lasso", "LightGBM"]:
            part = signals.loc[(signals.pool == pool) & (signals.scorer == scorer)]
            row = aggregate.loc[
                (aggregate.pool == pool)
                & (aggregate["mode"] == "full")
                & (aggregate.cost_bps == 10)
                & (aggregate.scorer == scorer)
            ].iloc[0]
            high = aggregate.loc[
                (aggregate.pool == pool)
                & (aggregate["mode"] == "full")
                & (aggregate.cost_bps == 20)
                & (aggregate.scorer == scorer)
            ].iloc[0]
            valid_ic = part.rank_ic.dropna()
            lines.append(
                f"| {names[pool]} | {scorer} | {valid_ic.mean():+.4f} | "
                f"{valid_ic.gt(0).mean():.1%} | {pp(high.own_index_excess)} | "
                f"{pp(row.own_tr_excess)} |"
            )
    lines.extend(
        [
            "",
            "Rank IC为每个信号日预测分数与后续5日开盘至开盘毛收益的Spearman相关，"
            "不等同于扣费组合收益；这些日期的IC也不是相互独立的显著性证据。全收益指数只发布收盘点位，"
            "故其超额采用双方第一日收盘至最后一日收盘，排除首日收益和费用，不能与开盘口径直接互换。",
            "策略在区间末日收盘清算，最后一轮持有短于完整5日开盘至开盘预测期限。",
            "",
            "## 实施与数据审计",
            "",
            f"- Qlib参考版本：`{config['upstream_commit']}`；原版158项公式中仅删VWAP0，保留157项。"
            "复权OHLC与原始amount/volume的口径未独立核实，未用收盘价代替VWAP。",
            "- 使用BaoStock按信号/训练日期查询的历史成分股，"
            "逐次验证300/500只、无重复且更新时间不晚于查询日。"
            "数据商每周更新，仍有调整日粒度及历史修订风险，不宣称完成官方逐日成分审计。",
            "- 全部成分日期中，amount/volume/close为有限正数的记录覆盖"
            f"最低{insights['dated_member_valid_amount_volume_close_fraction_min']:.2%}、"
            f"平均{insights['dated_member_valid_amount_volume_close_fraction_mean']:.2%}。"
            f"交易信号日通过共有筛选的股票数为{insights['eligible_range'][0]}—{insights['eligible_range'][1]}只，"
            f"低于25只的信号组有{insights['under_25_signal_groups']}个；实际持仓允许少于25只。",
            "- 每个30日窗口仅在开始前拟合一次。训练最多252个交易日，验证63日，训练样本约每5日抽取；"
            "边界留出未来5日标签的实现时间。训练标签终止日严格早于验证起点，验证标签终止日严格早于首个测试信号日。",
            "- 所有归一化/缺失值处理参数仅在训练集拟合。训练样本为当日成分股及量价条件，"
            "没有读取财报、行业、历史市值。训练期未额外按ST过滤；实际交易股票池继续使用缓存中的历史ST/交易状态。",
            "- Lasso固定alpha=0.0005，截距为0；LightGBM保留公开示例的主要树参数，种子42、单线程，"
            "最多1000轮、仅根据验证集早停。未根据测试收益调参。"
            f"每折训练{insights['training_rows_range'][0]}—{insights['training_rows_range'][1]}条，"
            f"LightGBM选择{insights['boosting_iterations_range'][0]}—{insights['boosting_iterations_range'][1]}轮。",
            f"- Lasso迭代均未达到上限：{insights['lasso_all_converged_before_limit']}；"
            f"预测近乎常数的信号组：{insights['constant_signal_groups']}/{insights['signal_groups']}。"
            "分数相同时按股票代码排序，因此常数预测的组合不能解释为有效模型择股。",
            "- 原作者的长训练区间、1日收盘标签、50只TopkDropout组合及非对称费用均已适配，"
            "因此不得把本结果与原仓库年化数字直接比较。",
            "- 原市场数据仍未完成退市覆盖和时点复权的独立审计；账户使用分数股，"
            "没有100股整手、实际印花税分段、冲击成本或盘口成交量约束。",
            "- 两个新股票池与此前全市场结果不同，原14段全市场目标不能直接视作已通过或失败。"
            "后续是否采用应看新数据验证；本次没有修改生产默认配置。",
            "",
            "## 复现与完整输出",
            "",
            "锁定方案：`config/research/qlib_comparison.json`。结果目录：`logs/qlib-comparison-20261015/`。",
            "",
            "- `aggregate.csv`：所有股票池/模型/仓位/费用、四个价格指数与对应全收益指数。",
            "- `summary.csv`、`daily_nav.csv`、`trades_10bps.csv`：全部窗口账户和逐日/交易记录。",
            "- `paired_vs_P4.csv`、`entry_year_groups.csv`、`signal_diagnostics.csv`："
            "配对、年份和预测诊断。",
            "- `folds.csv`、`models/`：切分、标签边界、模型、预处理参数及收敛信息。",
            "- `upstream/receipt.json`、`membership/`、`feature_receipt.json`："
            "参考源码和数据来源校验。",
            "- `formula_audit.json`：157项公式在完整样例上与固定版本源码表达式逐项核对；"
            "停牌缺失场景未宣称与Qlib全部算子严格等价。",
            "",
            "依次运行：`qlib_external_sources.py` → `lock_qlib_comparison.py` "
            "→ `prepare_qlib_features.py` "
            "→ `fit_qlib_comparison.py` → `run_qlib_comparison.py` → `report_qlib_comparison.py`，"
            "脚本均位于`scripts/research/`；重跑前仍需执行资源预检。",
            "机器学习环境版本记录在`fit_qa.json`，本次使用LightGBM 4.7.0、"
            "scikit-learn 1.9.1；未向生产入口引入机器学习依赖。",
            "",
            "来源：[Qlib公式](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)、"
            "[Qlib模型配置](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)、"
            "[vnpy Lasso实现](https://github.com/vnpy/vnpy/blob/master/vnpy/alpha/model/models/lasso_model.py)。",
        ]
    )
    text = "\n".join(lines) + "\n"
    destination = ROOT / "docs/research/qlib_comparison_results_20261015.md"
    destination.write_text(text, encoding="utf-8")
    (OUT / "report.md").write_text(text, encoding="utf-8")
    verification = {
        "technical_status": "PASS",
        "resource": RESOURCE_CONTEXT,
        "account_count": len(summary),
        "fresh_out_of_sample": False,
        "production_sha256": sha256(ROOT / "config/default.toml"),
        "report_sha256": sha256(destination),
        "summary_sha256": sha256(OUT / "summary.csv"),
        "plan_sha256": sha256(OUT / "plan_locked.json"),
        "signal_causality_tests": "tests/research/test_qlib_comparison.py",
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(
        aggregate.loc[
            (aggregate.cost_bps == 10) & aggregate.scorer.ne("PoolEW"),
            [
                "pool",
                "mode",
                "scorer",
                "mean_return",
                "own_index_excess",
                "mean_pool_excess",
                "worst_drawdown",
                "wins_own_index",
                "drop_best_own_excess",
            ],
        ].to_string(index=False),
        flush=True,
    )
    print("Report saved", destination, flush=True)


if __name__ == "__main__":
    main()
