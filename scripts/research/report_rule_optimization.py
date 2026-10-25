"""Publish complete finite-study results, failed targets and the frozen holdout result."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-rule-optimization-20261014"


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
    import pandas as pd
    from prepare_market_weight_batch import sha256
    from select_rule_finalist import aggregate

    config = json.loads((OUT / "candidate_suite_locked.json").read_text(encoding="utf-8"))
    candidates = {c["id"]: c for c in config["candidates"]}
    locked = json.loads((OUT / "windows_locked.json").read_text(encoding="utf-8"))
    target = json.loads((OUT / "targets_locked.json").read_text(encoding="utf-8"))
    finalist = json.loads((OUT / "finalist_locked.json").read_text(encoding="utf-8"))
    development = pd.read_csv(OUT / "development/summary.csv")
    primary = pd.read_csv(OUT / "development/primary_targets.csv").set_index("candidate")
    agg = pd.read_csv(OUT / "development/aggregate.csv")
    gates = pd.read_csv(OUT / "development/target_gates.csv").set_index("candidate")
    holdout = pd.read_csv(OUT / "holdout/summary.csv")
    holdout_aggregate = aggregate(holdout)
    holdout_aggregate.to_csv(OUT / "holdout/aggregate.csv", index=False, encoding="utf-8-sig")
    assert len(set(holdout.candidate)) == 2 and set(holdout.candidate) == {
        "O00",
        finalist["candidate_id"],
    }
    assert sha256(OUT / "development/summary.csv") == finalist["development_summary_sha256"]
    assert not set(holdout.window) & set(development.window)
    held_days = set(pd.read_csv(OUT / "holdout/daily_nav.csv").day)
    development_days = set(pd.read_csv(OUT / "development/daily_nav.csv").day)
    assert not held_days & development_days
    qa_dev = json.loads((OUT / "development/qa.json").read_text(encoding="utf-8"))
    qa_holdout = json.loads((OUT / "holdout/qa.json").read_text(encoding="utf-8"))
    assert qa_dev["technical_status"] == qa_holdout["technical_status"] == "PASS"
    for qa in [qa_dev, qa_holdout]:
        assert (
            qa["windows_priority_verified"] == "BelowNormal" and not qa["financial_industry_inputs"]
        )
        for receipt in qa["input_receipts"]:
            assert sha256(Path(receipt["path"])) == receipt["sha256"], receipt["path"]
    source = json.loads((OUT / "data_receipt_development.json").read_text(encoding="utf-8"))
    for receipt in source["sources"]:
        assert sha256(Path(receipt["path"])) == receipt["sha256"]
    factors = pd.read_parquet(
        OUT / "development_factors.parquet", columns=["day", "symbol", "historical_isST"]
    )
    unknown = factors[factors.historical_isST.isna()].groupby("symbol").size().to_dict()
    baseline = primary.loc["O00"]
    income = primary.loc["O04"]
    winner = primary.loc[finalist["candidate_id"]]
    validation = holdout_aggregate[holdout_aggregate.cost_bps == 10].set_index("candidate")
    sensitivity_rows = []
    for candidate in ["O00", "O04", finalist["candidate_id"]]:
        for _, row in agg[agg.candidate == candidate].sort_values("cost_bps").iterrows():
            sensitivity_rows.append(
                [
                    candidate,
                    int(row.cost_bps),
                    pct(row.mean_return),
                    pp(row.mean_excess_000985),
                    pp(row.mean_excess_H00985),
                    pp(row.mean_pool_excess),
                    f"{row.mean_trade_notional:.2f}倍",
                    pct(row.mean_fees),
                ]
            )
    rows = []
    for candidate in candidates:
        row = primary.loc[candidate]
        rows.append(
            [
                candidate,
                candidates[candidate]["name"],
                pct(row.mean_return),
                pp(row.mean_excess_000985),
                pp(row.mean_pool_excess),
                pct(row.worst_return),
                pct(row.worst_drawdown),
                f"{int(row.wins_000985)}/14",
                f"{int(gates.loc[candidate, 'passed_gate_count'])}/7",
            ]
        )
    report = [
        "# 价量策略第一轮规则优化结果",
        "",
        "**研究结论：净收益和风险各有改善，但12组规则均未同时达到冻结目标；留出测试也没有支持生产替换。**",
        "",
        f"14段开发集实际回放504个策略账户、1,134个同池/风险分配对照账户。另在未查看的1段30交易日历史留出区间，"
        f"仅回放已锁定候选{finalist['candidate_id']}与原基线，共6个策略账户和12个对照账户。"
        "合计510个策略账户、1,146个对照账户。费用档位为0/10/20bps，全部连续持仓、差额调仓。",
        "",
        "本页平均收益是14个独立30交易日窗口的算术均值，不拼接、不年化；价格指数首日开盘至末日收盘。"
        "全收益指数只有收盘，双方按首日收盘至末日收盘重新对齐，原始30日收益另保留。",
        "",
        "## 改善与未达项",
        "",
        table(
            ["指标", "原基线O00", "收益方向O04", "风险方向/冻结候选O09"],
            [
                [
                    "平均净收益",
                    pct(baseline.mean_return),
                    pct(income.mean_return),
                    pct(winner.mean_return),
                ],
                [
                    "对中证全指平均超额(百分点)",
                    pp(baseline.mean_excess_000985),
                    pp(income.mean_excess_000985),
                    pp(winner.mean_excess_000985),
                ],
                [
                    "同池相同仓位规则超额(百分点)",
                    pp(baseline.mean_pool_excess),
                    pp(income.mean_pool_excess),
                    pp(winner.mean_pool_excess),
                ],
                [
                    "最差窗口净收益",
                    pct(baseline.worst_return),
                    pct(income.worst_return),
                    pct(winner.worst_return),
                ],
                [
                    "最差窗口回撤",
                    pct(baseline.worst_drawdown),
                    pct(income.worst_drawdown),
                    pct(winner.worst_drawdown),
                ],
                [
                    "跑赢中证全指窗口",
                    f"{int(baseline.wins_000985)}/14",
                    f"{int(income.wins_000985)}/14",
                    f"{int(winner.wins_000985)}/14",
                ],
                [
                    "成交名义金额/初始净值",
                    f"{baseline.mean_trade_notional:.2f}倍",
                    f"{income.mean_trade_notional:.2f}倍",
                    f"{winner.mean_trade_notional:.2f}倍",
                ],
            ],
        ),
        "",
        f"O04采用25只等权、每10日调仓；已有持仓只要仍在共同合格池且排名前40就保留，空位按排名补齐。"
        f"相对基线，平均净收益提高{pp(income.mean_return - baseline.mean_return)}个百分点，"
        "成交金额减少"
        f"{(1 - income.mean_trade_notional / baseline.mean_trade_notional) * 100:.1f}%。"
        "它通过收益、同池超额、20bps费用压力和对齐全收益超额门槛，"
        "但最差亏损、最大回撤和9/14胜率未达标。O04没有作为最终候选去看留出结果。",
        "",
        "O09采用50只等权、10日调仓、前75名保留缓冲，并按信号日前一收盘可知的中证全指20日波动率调整仓位："
        "股票仓位=min(1,15%/指数20日年化波动率)，不加杠杆。开发集调仓时目标仓位平均约76.39%。"
        "它通过最差亏损、回撤和同池超额门槛，仍未通过全指超额、胜率、费用压力及对齐全收益超额。"
        "该方案是事前“风险门槛合格者中选全指超额最高”的退选规则产生的候选，不是达标方案。",
        "",
        f"O09同仓位规则的同池超额为{pp(winner.mean_pool_excess)}个百分点，"
        f"对原全仓同池对照为{pp(winner.mean_full_pool_excess)}个百分点。"
        "两种对照均保留，降低敞口的收益不全部计为选股贡献。",
        "",
        "## 12组完整开发结果：单边10bps",
        "",
        table(
            [
                "编号",
                "规则",
                "净收益均值",
                "全指超额",
                "同池超额",
                "最差净收益",
                "最差回撤",
                "全指胜率",
                "目标通过",
            ],
            rows,
        ),
        "",
        "## 费用与全收益压力测试",
        "",
        table(
            [
                "编号",
                "单边bps",
                "均净收益",
                "价格全指超额",
                "对齐全收益超额",
                "同池超额",
                "成交金额",
                "费用/初始净值",
            ],
            sensitivity_rows,
        ),
        "",
        "## 新历史留出结果",
        "",
        "区间为2025-05-19至2025-06-30，共30交易日，先按日历锁定、与旧14段持有日期不重叠。"
        "评选仅使用开发集，O09的配置与开发结果SHA256在计算留出收益前锁定。"
        "只查看O00与O09，未根据留出结果改挑O04或其他方案。",
        "",
        table(
            ["方案", "净收益", "中证全指超额", "同池超额", "最大回撤"],
            [
                [
                    candidate,
                    pct(validation.loc[candidate, "mean_return"]),
                    pp(validation.loc[candidate, "mean_excess_000985"]),
                    pp(validation.loc[candidate, "mean_pool_excess"]),
                    pct(validation.loc[candidate, "worst_drawdown"]),
                ]
                for candidate in ["O00", finalist["candidate_id"]]
            ],
        ),
        "",
        "该期中证全指价格指数收益为+3.11%。O09的平均目标仓位为100%，因此本期表现差不能简单归因于持有现金；"
        "50只持仓、10日频率及缓冲规则的组合在该期没有改善选股收益。",
        "",
        "现有数据只能找到1段完整、未查看且与已有14段互不重叠的30交易日区间。"
        "该区间穿插在开发历史中，也不是严格按先训练后测试的向前验证。"
        "冻结目标要求至少6段新验证，数量门槛未满足；这一段只能作为补充历史时间留出，不能声称样本外验收通过。",
        "",
        "## 因子诊断",
        "",
        "只使用开发窗口内部的收益标签：以信号次日开盘买入，检查未来5/10/20日收盘收益；超出原窗口的标签不读。"
        "排名相关系数衡量因子高低与后续收益排序是否一致，并列出前后10%组收益差。"
        "这些是可报价股票的毛收益诊断，不代替含停牌/涨跌停/费用的实际账户回放。",
        "",
        "在原放量门槛池、10日标签的70个信号上，20日动量平均排名相关约−0.118，60日动量约−0.094；"
        "低波动因子约+0.045，但其前后10%组平均收益差仍为负。"
        "这提示“过去涨得多”的正向排序和短持有期限可能不匹配，低波动也没有简单的分组单调关系。"
        "单个信号的标签有重叠，70次不是70个独立样本，当前未做统计显著性宣称。",
        "",
        "下一轮研究可优先检验短期反转与中期趋势分别建模、再组合低换手规则；"
        "应重新预先固定有限规则，记录累积试验次数，使用额外未查看数据或前向模拟。"
        "不能把本轮已查看的留出区间重复用作新的独立验收，也不能仅因本轮动量相关为负就认定反向买入有效。",
        "",
        "## 实现与验证",
        "",
        "- 新引擎O00在全部14段、三档费用的账户收益和逐日净值复现原结果，"
        f"最大收益误差{qa_dev['baseline_max_return_difference']:.3g}。",
        "- 账户按现金约束求解费用后目标仓位，先卖后买；缓冲使用实际持仓；"
        "无法退出的库存继续占用仓位并计入净值。",
        "- 波动率分配限制单股目标权重4%（股票少于25只时无法满足该上限，"
        "会采用1/股票数的可行上限）；股权数量允许分数，未建模100股整手与账户规模。",
        "- 测试覆盖排名缓冲减少换手、停牌库存占位、0/10/20bps资金守恒、"
        "实际订单波动率权重上限；报告外另存测试与静态检查回执。",
        "- 排序、个股波动率权重、市场仓位均只读信号日前数据；"
        "开盘执行延续前轮的报价与历史交易状态假设。",
        "- 不可复现的财报、行业、题材和资金流没有加入评分。历史状态只读日期、代码、ST和交易状态。",
        "",
        "## 数据限制与交付",
        "",
        f"开发因子中未知历史状态代码及信号数：{json.dumps(unknown)}。这些候选在共同池中排除，未用当前名称填补历史ST。"
        "本次留出候选的历史状态覆盖完整。",
        "",
        "股票价格来自冻结的前复权归档；项目日更源码注明腾讯qfq及此前ADJUST_PREV。"
        "复权漂移、退市覆盖与历史股票全集尚未完成独立点时审计；20bps是费用压力情景，未证明能覆盖所有实盘滑点、整手约束和冲击成本。"
        "价量受限模型的研究结果也不等同线上包含财报和行业数据的完整版。",
        "",
        "文件：candidate_suite_locked.json、targets_locked.json、windows_locked.json锁定设计；"
        "development/含全部账户、逐日净值、10bps交易账本、因子诊断、年度稳定性及7项验收结果；"
        "holdout/含锁定候选与基线；finalist_locked.json与finalist_research_config.json供复现；"
        "qa.json和code_snapshot/保存检查与脚本。",
        "",
        "历史状态来源：[BaoStock官方API](https://github.com/zxygithub/baostock/blob/master/docs/pythonAPI.md)。"
        "指数历史直接来自[中证官网](https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode=000985&startDate=20230101&endDate=20260924)，"
        "8条价格/全收益序列的原始响应及SHA256见index_data/。",
        "",
    ]
    text = "\n".join(report)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    (ROOT / "docs/research/market_only_rule_optimization_results.md").write_text(
        text, encoding="utf-8"
    )
    export = {
        **finalist,
        "status": "research_only_validation_not_passed",
        "development_qa": qa_dev["technical_status"],
        "holdout_windows": 1,
        "required_validation_windows": target["evaluation_policy"][
            "validation_window_minimum_for_preliminary_acceptance"
        ],
        "production_replacement_supported": False,
        "factor_weights": {"momentum": 1 / 3, "risk": 1 / 3, "liquidity": 1 / 3},
        "source_suite": "market_only_v1",
        "primary_cost_bps": 10,
        "baseline_reference": "O00",
        "holdout_mean_return_10bps": float(validation.loc["O09", "mean_return"]),
    }
    (OUT / "finalist_research_config.json").write_text(
        json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files = [
        "run_rule_optimization.py",
        "rule_account_replay.py",
        "prepare_rule_data.py",
        "lock_rule_optimization.py",
        "select_rule_finalist.py",
        "report_rule_optimization.py",
        "research_resources.py",
        "extend_market_weight_windows.py",
        "prepare_market_weight_batch.py",
        "fetch_market_status.py",
    ]
    snapshot = OUT / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in files:
        shutil.copyfile(ROOT / "scripts/research" / filename, snapshot / filename)
    shutil.copyfile(
        ROOT / "tests/research/test_rule_account_replay.py",
        snapshot / "test_rule_account_replay.py",
    )
    qa = {
        "technical_status": "PASS",
        "optimization_target_status": "NOT_MET",
        "resource_context": RESOURCE_CONTEXT,
        "development_strategy_accounts": 504,
        "holdout_strategy_accounts": 6,
        "development_accounts": len(development),
        "holdout_accounts": len(holdout),
        "candidate_variants": 12,
        "finalist_locked_before_holdout": True,
        "all_target_passers": int(gates.all_development_gates_passed.sum()),
        "validation_windows_available": 1,
        "validation_windows_required": 6,
        "formal_out_of_sample_status": "NOT_PASSED",
        "future_labels_outside_development_windows_read": False,
        "production_replacement_supported": False,
        "baseline_max_return_difference": qa_dev["baseline_max_return_difference"],
        "all_input_receipts_verified": True,
        "production_config_sha256": sha256(ROOT / "config/default.toml"),
        "candidate_suite_sha256": sha256(OUT / "candidate_suite_locked.json"),
        "target_sha256": sha256(OUT / "targets_locked.json"),
        "finalist_sha256": sha256(OUT / "finalist_locked.json"),
        "code_snapshots": [{"name": name, "sha256": sha256(snapshot / name)} for name in files],
    }
    assert (
        qa["candidate_suite_sha256"] == locked["candidate_sha256"]
        and qa["target_sha256"] == locked["target_sha256"]
    )
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: qa[k]
                for k in [
                    "technical_status",
                    "optimization_target_status",
                    "candidate_variants",
                    "formal_out_of_sample_status",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
