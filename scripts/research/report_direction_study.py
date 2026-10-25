"""Publish all direction hypotheses and scoped, outlier-aware research findings."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-direction-study-20261015"


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

    suite_path = OUT / "candidate_suite_locked.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    lock = json.loads((OUT / "windows_locked.json").read_text(encoding="utf-8"))
    qa_run = json.loads((OUT / "research/qa.json").read_text(encoding="utf-8"))
    insights = json.loads((OUT / "insights.json").read_text(encoding="utf-8"))
    assert sha256(suite_path) == lock["candidate_sha256"]
    assert qa_run["technical_status"] == insights["technical_status"] == "PASS"
    assert qa_run["baseline_max_return_difference"] < 1e-10
    for receipt in qa_run["input_receipts"]:
        assert sha256(Path(receipt["path"])) == receipt["sha256"], receipt["path"]
    cfg = {c["id"]: c for c in suite["candidates"]}
    aggregate = pd.read_csv(OUT / "aggregate.csv")
    primary = aggregate[
        (aggregate.cohort == "all_15_viewed") & (aggregate.cost_bps == 10)
    ].set_index("candidate")
    original = aggregate[
        (aggregate.cohort == "original_14") & (aggregate.cost_bps == 10)
    ].set_index("candidate")
    gates = pd.read_csv(OUT / "original14_target_gates.csv").set_index("candidate")
    pairs = pd.read_csv(OUT / "paired_comparisons.csv")
    pairs = pairs[pairs.cohort == "all_15_viewed"].set_index("id")
    frame = pd.read_csv(OUT / "research/summary.csv")
    strategy = frame[(frame.account == "strategy") & (frame.cost_bps == 10)]
    metadata = pd.read_csv(OUT / "window_metadata.csv")
    assert len(metadata) == 15 and not metadata.window.duplicated().any()
    stratified = pd.read_csv(OUT / "stratified_results.csv")
    leader_id = insights["top_mean_excess_candidates"][0]
    leader = primary.loc[leader_id]
    best = (
        strategy[strategy.candidate == leader_id]
        .sort_values("excess_000985", ascending=False)
        .iloc[0]
    )
    all_depend_on_best = bool(primary.mean_excess_without_best_window.lt(0).all())
    assert all_depend_on_best
    assert primary.minimum_year_mean_excess.lt(0).all()
    assert set(pairs[pairs.status != "mixed"].index) == {
        "reverse_vs_chase",
        "remove_volume_reverse_risk",
    }
    text = [
        "# 第二轮方向与规则研究：21组新规则、3组对照",
        "",
        "新增21组不同规则，保留3组旧方案；在15个已查看的30交易日区间、0/10/20bps费用下，"
        "实际回放1,080个策略账户和2,205个对照账户，共3,285个账户。"
        "3个旧对照的收益与逐日净值复现一致。数据仍使用截至2026-09-24的冻结快照。",
        "",
        "**发现两条较一致的相对表现线索，但没有找到经过新样本外验证的稳定盈利规律。**"
        "24个方案分别去掉各自超额最高的一段后，平均全指超额全部变为负值；每个方案也都存在平均超额为负的起始年份组。",
        "",
        "本页主口径为单边10bps、连续持仓、差额调仓，收益为独立30日区间算术均值，超额单位为百分点。"
        "不拼接、不复利成年化收益。四个价格指数及对应全收益指数继续全部比较。",
        "",
        "## 相对一致的两条线索",
        "",
        "1. **5日反转优于同条件的5日追涨。** R02与R01除了排序方向外，"
        "股票池、持仓数、5日调仓、排名缓冲和费用相同。"
        f"反转在{int(pairs.loc['reverse_vs_chase', 'positive_windows'])}/15段占优，"
        f"平均净收益高{pp(pairs.loc['reverse_vs_chase', 'mean_effect'])}个百分点；"
        "各起始年份平均差均为正，20bps费用和剔除效果最强一段后方向也保持一致。"
        f"但R02本身对全指平均超额仅{pp(primary.loc['R02', 'mean_excess_000985'])}个百分点，"
        f"最差回撤{pct(primary.loc['R02', 'worst_drawdown'])}，"
        "不能把“比追涨少亏或多赚”直接当成稳健正超额。",
        "",
        f"2. **对反转＋风控流动性组合，保留放量门槛较好。** R20相对R08只取消量比≥1.1条件。"
        f"取消后有{int(pairs.loc['remove_volume_reverse_risk', 'negative_windows'])}/15段变差，"
        "平均净收益下降"
        f"{abs(pairs.loc['remove_volume_reverse_risk', 'mean_effect']) * 100:.2f}个百分点；"
        "三个起始年份、20bps费用和去掉最差影响一段后方向一致。"
        "这条结论限定于这个组合；纯5日反转取消放量门槛的结果没有通过同样的一致性标准。",
        "",
        "以上“相对一致”是事先固定的敏感性筛选：平均数和中位数同向、至少2/3区间同向、每个起始年份均差同向、"
        "20bps同向、剔除最强影响段后仍同向。16组成对比较全部披露；这不是显著性检验或因果识别，也没有消除多次试验偏差。",
        "",
        "## 目前较有吸引力的研究候选",
        "",
        f"{leader_id}：{cfg[leader_id]['name']}。保持原P4类别等权评分，25只等权、10日调仓、前40名保留缓冲；"
        "仓位=min(1,15%/信号日前中证全指20日年化波动率)，无杠杆。"
        "它是这批已查看数据中平均全指超额最高的方案，并未据此升级生产策略。",
        "",
    ]
    comparison_rows = []
    for candidate in ["C00", "C01", leader_id, "R02", "R05"]:
        row = primary.loc[candidate]
        comparison_rows.append(
            [
                candidate,
                cfg[candidate]["name"],
                pct(row.mean_return),
                pp(row.mean_excess_000985),
                pp(row.mean_pool_excess),
                pct(row.worst_drawdown),
                f"{int(row.wins_000985)}/15",
            ]
        )
    text += [
        table(
            ["方案", "规则", "平均净收益", "全指超额", "同仓位同池超额", "最差回撤", "胜全指段数"],
            comparison_rows,
        ),
        "",
        "### 按原14段冻结目标验收",
        "",
    ]
    row, baseline = original.loc[leader_id], original.loc["C00"]
    text += [
        f"在原14段，{leader_id}平均净收益{pct(row.mean_return)}，相对基线{pct(baseline.mean_return)}"
        f"提高{pp(row.mean_return - baseline.mean_return)}个百分点；"
        f"全指超额{pp(row.mean_excess_000985)}个百分点，"
        f"最差回撤{pct(row.worst_drawdown)}。"
        f"7项开发目标通过{int(gates.loc[leader_id, 'passed_gates'])}/7项，"
        f"仍只有{int(row.wins_000985)}/14段跑赢全指，未达到9/14段门槛。24组均未全部通过原目标。",
        "",
        f"原14段的对沪深300、中证500、中证1000超额分别为{pp(row.mean_excess_000300)}、"
        f"{pp(row.mean_excess_000905)}、{pp(row.mean_excess_000852)}个百分点。"
        "完整三档费用、四个价格指数和四个对齐全收益指数结果见aggregate.csv。",
        "",
        "### 去掉最强行情后，优势仍不成立",
        "",
        f"{leader_id}最强的一段为{best['start']}至{best['end']}，净收益{pct(best['return'])}，"
        f"对全指超额{pp(best.excess_000985)}个百分点。剔除这一段，剩余14段平均全指超额为"
        f"{pp(leader.mean_excess_without_best_window)}个百分点。"
        "这说明当前平均正超额较依赖少数强行情，尚未形成覆盖多数环境的优势。",
        "",
        "按区间起始年分组如下；跨年区间归入起始年，表内是被抽取区间的平均值，**不是全年收益**。",
        "",
    ]
    year_rows = []
    for _, year in stratified[
        (stratified.candidate == leader_id)
        & (stratified.cost_bps == 10)
        & (stratified.grouping == "year")
    ].iterrows():
        year_rows.append(
            [
                str(year["group"]),
                int(year.windows),
                pct(year.mean_return),
                pp(year.mean_excess_000985),
                pct(year.worst_drawdown),
            ]
        )
    text += [
        table(["起始年", "区间数", "平均净收益", "平均全指超额", "最差回撤"], year_rows),
        "",
        "## 调仓周期没有统一最优值",
        "",
        "原P4方案降低换手有帮助，但不能把“10日比5日好”推广给所有信号。"
        f"纯5日反转按5/10/20日调仓的平均净收益分别为{pct(primary.loc['R02', 'mean_return'])}、"
        f"{pct(primary.loc['R03', 'mean_return'])}、{pct(primary.loc['R16', 'mean_return'])}。"
        "5日改10日的效果在2024/2025起始年份为负、2026为正，没有通过跨期一致性筛选。"
        "持有周期需要与信号匹配，不能只靠降低交易费用判断。",
        "",
        "中期趋势剔除最近5日、趋势加入短期回撤，多数区间相对改善，但2025起始年份仍反向；"
        "25只改50只、波动率倒数分配及均线状态切换均未形成跨期一致的收益改善。"
        "回撤改善和平均净收益改善分别报告，不能互相替代。",
        "",
        "## 24组完整结果：全部15段、单边10bps",
        "",
    ]
    all_rows = []
    for candidate in cfg:
        row = primary.loc[candidate]
        all_rows.append(
            [
                candidate,
                cfg[candidate]["name"],
                pct(row.mean_return),
                pp(row.mean_excess_000985),
                pct(row.worst_return),
                pct(row.worst_drawdown),
                f"{int(row.wins_000985)}/15",
                pp(row.mean_excess_without_best_window),
            ]
        )
    text += [
        table(
            [
                "编号",
                "规则",
                "均净收益",
                "全指超额",
                "最差净收益",
                "最差回撤",
                "胜全指段数",
                "剔除最佳段后超额",
            ],
            all_rows,
        ),
        "",
        "## 全部16组成对比较",
        "",
        "差值为“替代规则净收益−原规则净收益”，单位为百分点；每对尽量只改一个条件。"
        "同一窗口指数相同，因此净收益差也等于对同一指数的超额差。"
        "涉及股票池变化时，配套同池基准也会变化，不能将整个收益差解释为纯排序贡献。",
        "",
    ]
    pair_rows = []
    for spec in suite["paired_hypotheses"]:
        row = pairs.loc[spec["id"]]
        label = {
            "consistent_positive": "较一致改善",
            "consistent_negative": "较一致变差",
            "mixed": "不一致",
        }[row.status]
        pair_rows.append(
            [
                spec["name"],
                pp(row.mean_effect),
                pp(row.median_effect),
                int(row.positive_windows),
                int(row.negative_windows),
                pp(row["20bps_mean_effect"]),
                label,
            ]
        )
    text += [
        table(
            ["变化", "均差", "中位差", "改善段数", "变差段数", "20bps均差", "一致性筛选"], pair_rows
        ),
        "",
        "## 市场状态和逐段表现",
        "",
        f"按全窗口全指收益事后划分，上涨(>5%)有{insights['regime_counts'].get('up', 0)}段、"
        f"震荡有{insights['regime_counts'].get('sideways', 0)}段、"
        f"下跌(<−5%)仅{insights['regime_counts'].get('down', 0)}段。"
        "这组事后标签只用于描述，未用于买卖；强下跌样本太少，不能据此认定普遍熊市规律。"
        "另存入场前全指是否高于60日均线的7段/8段分组，交易状态切换只使用该信号当时可见的均线信息。",
        "",
    ]
    window_rows = []
    focus = ["C00", "C01", leader_id, "R02", "R01"]
    for window in sorted(lock["windows"], key=lambda w: w["start"]):
        group = strategy[strategy.window == window["label"]].set_index("candidate")
        window_rows.append(
            [
                window["start"] + "至" + window["end"],
                *[pp(group.loc[c, "excess_000985"]) for c in focus],
            ]
        )
    text += [
        table(["日期", *[c + "全指超额" for c in focus]], window_rows),
        "",
        "## 解释边界与复现",
        "",
        "本轮采用上一轮诊断启发的新假设，属于第二轮自适应研究。两轮规则累计33种（12旧规则＋21新规则），"
        "另有更早的8组权重及其他历史实验；3个重复对照不算新规则。"
        "上一轮2025-05-19至06-30留出区间现已查看，明确归入研究数据。"
        "15段都不是本轮新的独立样本外数据，不能把研究目标通过项或排名最高解释成样本外成功。",
        "",
        "研究仍只使用可重算价量因子、上市日期和历史ST/交易状态。"
        "补充5/10日收益及“60日趋势剔除最近5日”时只使用信号日前价格；"
        "未来价格扰动测试不改变历史信号，原20/60日收益与缓存完全一致。"
        "财报、行业、题材、资金流未进入评分。",
        "",
        "价格指数从首日真实开盘买入持有至末日收盘，指数毛收益对策略费用后收益。"
        "全收益指数只公布收盘，因此双方按首日收盘至末日收盘的29个间隔对齐；"
        "首日行情和入场费用在该补充比较期之前。仓位规则也应用于同池对照，原全仓对照继续保留。",
        "",
        "股票归档为前复权价格；退市覆盖、复权点时版本、整手及冲击成本仍未完成独立审计。"
        "历史状态未知的股票沿用共同排除规则，未用当前名称回填ST。"
        "所有账户继续遵守现金约束和无法退出持仓留存规则。",
        "",
        "短期反转与流动性的关系有既有研究背景，但当前A股价量模型并非论文复现，效果只能由本项目数据检验。"
        "[研究背景](https://www.nber.org/papers/w30917)。"
        "对历史回测、敏感性分析和前视/幸存者偏差的处理参照[CFA方法说明](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/backtesting-and-simulation)。",
        "",
        "交付：aggregate.csv（原14段与全部15段的全部费用和指数）、paired_comparisons.csv（成对一致性）、"
        "stratified_results.csv及paired_stratified_results.csv（年份和状态）、research/（账本/净值/信号）、"
        "original14_target_gates.csv、insights.json、候选配置、QA和脚本快照。",
        "",
    ]
    report = "\n".join(text)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/research/market_direction_patterns_20261015.md").write_text(
        report, encoding="utf-8"
    )
    export = {
        "status": "research_only_selected_on_viewed_data",
        "candidate": cfg[leader_id],
        "candidate_suite_sha256": sha256(suite_path),
        "formal_out_of_sample_passed": False,
        "original14_targets_passed": bool(gates.loc[leader_id, "all_development_targets_passed"]),
        "all15_mean_index_excess": float(leader.mean_excess_000985),
        "all15_excess_without_best_window": float(leader.mean_excess_without_best_window),
        "prior_factor_suite": "market_only_v1",
        "read_original_P4_weights_unchanged": True,
    }
    (OUT / ("research_candidate_" + leader_id + ".json")).write_text(
        json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    snapshot = OUT / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    files = [
        "design_direction_study.py",
        "prepare_direction_data.py",
        "direction_scoring.py",
        "run_direction_batch.py",
        "analyze_direction_study.py",
        "report_direction_study.py",
        "research_resources.py",
        "rule_account_replay.py",
        "prepare_market_weight_batch.py",
        "select_rule_finalist.py",
    ]
    for name in files:
        shutil.copyfile(ROOT / "scripts/research" / name, snapshot / name)
    for name in ["test_direction_study.py", "test_direction_pattern_analysis.py"]:
        shutil.copyfile(ROOT / "tests/research" / name, snapshot / name)
    qa = {
        "technical_status": "PASS",
        "resource_context": RESOURCE_CONTEXT,
        "new_rule_variants": 21,
        "controls": 3,
        "windows": 15,
        "strategy_accounts": 1080,
        "total_account_rows": len(frame),
        "paired_hypotheses": 16,
        "consistent_relative_patterns": len(pairs[pairs.status != "mixed"]),
        "all24_fail_without_own_best_window": all_depend_on_best,
        "all_original14_target_passers": insights["original14_target_passers"],
        "formal_out_of_sample_status": insights["formal_out_of_sample_status"],
        "control_max_return_difference": qa_run["baseline_max_return_difference"],
        "candidate_suite_sha256": sha256(suite_path),
        "target_sha256": sha256(OUT / "targets_locked.json"),
        "production_config_sha256": sha256(ROOT / "config/default.toml"),
        "financial_industry_inputs": [],
        "cumulative_rule_variants_two_rounds": 33,
        "code_snapshots": [{"name": name, "sha256": sha256(snapshot / name)} for name in files],
    }
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: qa[k]
                for k in [
                    "technical_status",
                    "new_rule_variants",
                    "consistent_relative_patterns",
                    "all24_fail_without_own_best_window",
                    "formal_out_of_sample_status",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
