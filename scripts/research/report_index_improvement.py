"""Report development changes and untouched tests without reselecting their winners."""

from __future__ import annotations

import json

from design_index_improvement import OUT, ROOT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def p(value):
    return f"{value * 100:+.2f}%"


def q(value):
    return f"{value * 100:+.2f}"


def main():
    import pandas as pd
    from prepare_market_weight_batch import sha256
    from select_index_finalists import summarize

    plan = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    selected = json.loads((OUT / "finalists_locked.json").read_text(encoding="utf-8"))
    assert selected["plan_sha256"] == sha256(OUT / "plan_locked.json")
    scope = json.loads((OUT / "validation_windows_locked.json").read_text(encoding="utf-8"))
    development = pd.read_csv(OUT / "development/target_gates.csv")
    pairs = pd.read_csv(OUT / "development/paired_structural_changes.csv")
    summaries, aggregates = {}, {}
    for cohort in ["external", "recent"]:
        summaries[cohort] = pd.read_csv(OUT / cohort / "summary.csv")
        aggregates[cohort] = summarize(summaries[cohort])
        aggregates[cohort].to_csv(OUT / cohort / "aggregate.csv", index=False)
    verdicts = []
    for candidate in selected["finalists"]:
        pool, cid = candidate["pool"], candidate["id"]
        row = (
            aggregates["external"]
            .loc[
                (aggregates["external"].pool == pool)
                & (aggregates["external"].candidate == cid)
                & (aggregates["external"].cost_bps == 10)
            ]
            .iloc[0]
        )
        high = (
            aggregates["external"]
            .loc[
                (aggregates["external"].pool == pool)
                & (aggregates["external"].candidate == cid)
                & (aggregates["external"].cost_bps == 20)
            ]
            .iloc[0]
        )
        gates = plan["external_gates"]
        checks = {
            "mean": bool(row.mean_excess >= gates["mean_own_index_excess_min"]),
            "wins": bool(row.wins >= gates["wins_min"]),
            "cost": bool(high.mean_excess >= gates["mean_20bps_own_index_excess_min"]),
            "total_return": bool(row.tr_excess >= gates["aligned_total_return_index_excess_min"]),
            "drop_best": bool(row.drop_best_excess >= gates["drop_best_mean_excess_min"]),
        }
        verdicts.append(
            {
                "pool": pool,
                "candidate": cid,
                "development_all_gates": candidate["development_all_gates_passed"],
                "external_checks": checks,
                "external_all_gates": all(checks.values()),
                "recent_days": len(scope["recent_window"]["days"]),
                "prospective_30day_validation_complete": False,
                "production_release": False,
            }
        )
    labels = {"csi300": "沪深300", "csi500": "中证500"}
    large_result = aggregates["external"].loc[
        (aggregates["external"].pool == "csi300")
        & (aggregates["external"].candidate == "F5_P4")
        & (aggregates["external"].cost_bps == 10)
    ].iloc[0]
    medium_result = aggregates["external"].loc[
        (aggregates["external"].pool == "csi500")
        & (aggregates["external"].candidate == "F1_Lasso")
        & (aggregates["external"].cost_bps == 10)
    ].iloc[0]
    lines = [
        "# 以跑赢指数为目标的结构优化（2026-10-16）",
        "",
        "**本轮未达到稳定跑赢指数的目标。**"
        f"六段新历史中，沪深300候选平均指数超额{q(large_result.mean_excess)}个百分点，"
        f"跑赢{int(large_result.wins)}/6段；中证500候选平均超额"
        f"{q(medium_result.mean_excess)}个百分点，跑赢{int(medium_result.wins)}/6段。"
        "旧样本中的改善不能据此视为可迁移的收益。",
        "",
        "本轮固定18个方案，在两个历史成分股池、15个已查看的30交易日区间中进行开发对照。"
        "复用已冻结的P4/Lasso/LightGBM评分，检验放量门槛、调仓周期及持仓分散度，未重新搜索模型参数。",
        "",
        "随后每个股票池只封存一个候选，验证6个此前未查看的历史区间，再检查冻结数据之后的9个交易日。"
        "历史验证时间早于开发样本，属于反向时间的历史压力检验；不能替代未来模拟跟踪。",
        "",
        "本轮开发区没有方案全部通过8项冻结门槛，因此两个候选均为研究候选，未替换生产默认评分。"
        "以下30日收益为独立窗口的算术均值，不拼接、复利或年化；主费用为买卖各0.1%，超额为百分点。",
        "",
        "## 旧样本中的具体改善",
        "",
        "| 股票池 | 方案 | 平均净收益 | 对本池指数超额 | 跑赢区间 | 20bps超额 "
        "| 去掉最佳段后超额 | 最差单段回撤 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    display = {
        "csi300": ["F0_P4", "F1_Lasso", "F5_P4"],
        "csi500": ["F0_Lasso", "F1_Lasso", "F3_Lasso"],
    }
    for pool, ids in display.items():
        for cid in ids:
            row = development.loc[(development.pool == pool) & (development.candidate == cid)].iloc[
                0
            ]
            lines.append(
                f"| {labels[pool]} | {cid} | {p(row.mean_return)} | {q(row.mean_excess)} | "
                f"{int(row.wins)}/15 | {q(row.excess_20bps)} | "
                f"{q(row.drop_best_excess)} | {p(row.worst_drawdown)} |"
            )
    lines.extend(
        [
            "",
            "F0为原规则；F1只取消量比≥1.1；F3在取消量比门槛后改为10日调仓、50只、前75名缓冲；"
            "F5为10日调仓、100只、前150名缓冲。其它ST、上市时间、250日历史及成交额筛选保留。",
            "",
            "开发门槛同时要求：平均指数超额≥0.5个百分点、中位超额≥0、至少9/15段跑赢、"
            "单边20bps平均超额≥0、对齐全收益指数超额≥0、剔除最佳段后仍非负、"
            "各起始年份组平均超额非负、单段回撤比对应指数恶化不超过2个百分点。",
            "",
            "## 结构变化的配对影响",
            "",
            "以下只改变同评分方法下的规则；扩展持仓时同时使用对应的固定排名缓冲。",
            "",
            "| 股票池 | 评分 | 新规则−旧规则 | 平均净收益差 | 改善段数 | 平均费用差 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in pairs.loc[(pairs.cost_bps == 10) & pairs.scorer.eq("Lasso")].iterrows():
        lines.append(
            f"| {labels[row.pool]} | Lasso | {row.new_family}−{row.old_family} | "
            f"{q(row.mean_difference)} | {int(row.wins)}/15 | {q(row.fee_difference)} |"
        )
    lines.extend(
        [
            "",
            "## 封存候选及未看历史验证",
            "",
            "候选只按开发区冻结规则选出：先看是否全门槛通过；若没有，则优先改善最弱的年份/剔除最佳段/高费用表现。"
            "验证结束后不更换候选，未按新历史结果重新挑选18个方案。",
            "",
            "| 股票池 | 封存候选 | 6段平均净收益 | 平均指数超额 | 跑赢段数 "
            "| 20bps超额 | 对齐全收益超额 | 最差单段回撤 | 验证门槛 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for verdict in verdicts:
        pool, cid = verdict["pool"], verdict["candidate"]
        part = aggregates["external"]
        row = part.loc[(part.pool == pool) & (part.candidate == cid) & (part.cost_bps == 10)].iloc[
            0
        ]
        high = part.loc[(part.pool == pool) & (part.candidate == cid) & (part.cost_bps == 20)].iloc[
            0
        ]
        lines.append(
            f"| {labels[pool]} | {cid} | {p(row.mean_return)} | {q(row.mean_excess)} | "
            f"{int(row.wins)}/6 | {q(high.mean_excess)} | {q(row.tr_excess)} | "
            f"{p(row.worst_drawdown)} | "
            f"{sum(verdict['external_checks'].values())}/5 |"
        )
    lines.extend(
        [
            "",
            "验证的5项门槛为平均超额≥0.5个百分点、至少4/6段跑赢、20bps仍非负、"
            "对齐全收益指数超额非负、剔除最佳段后超额非负。它们是小样本筛选门槛，不是统计盈利保证。",
            "",
            "| 历史验证区间 | 沪深300指数收益 | 300候选超额 | 中证500指数收益 "
            "| 500候选超额 | 原500 Lasso超额 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for window in scope["external_windows"]:
        rows = summaries["external"].loc[
            (summaries["external"].window == window["label"])
            & (summaries["external"].account == "strategy")
            & (summaries["external"].cost_bps == 10)
        ]
        own_rows = {}
        for candidate in selected["finalists"]:
            own_rows[candidate["pool"]] = rows.loc[
                (rows.pool == candidate["pool"]) & (rows.candidate == candidate["id"])
            ].iloc[0]
        large, medium = own_rows["csi300"], own_rows["csi500"]
        baseline = rows.loc[(rows.pool == "csi500") & (rows.candidate == "F0_Lasso")].iloc[0]
        lines.append(
            f"| {window['start']}—{window['end']} | "
            f"{p(large['return'] - large.excess_000300)} | {q(large.excess_000300)} | "
            f"{p(medium['return'] - medium.excess_000905)} | {q(medium.excess_000905)} | "
            f"{q(baseline.excess_000905)} |"
        )
    lines.extend(
        [
            "",
            "## 超额的来源核对",
            "",
            "下表把对指数超额拆成同池等权账户的指数偏离与候选相对同池账户的差。"
            "同池对照也包含筛选、等权和交易费用，因此后一项仍同时含选股与换手差异，不能直接称为纯因子收益。",
            "",
            "| 样本 | 股票池 | 候选对指数 | 同池等权对指数 | 候选对同池等权 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for cohort, label in [("development", "开发15段"), ("external", "新历史6段")]:
        for candidate in selected["finalists"]:
            source = development if cohort == "development" else aggregates["external"]
            row = source.loc[
                (source.pool == candidate["pool"])
                & (source.candidate == candidate["id"])
                & (source.cost_bps == 10)
            ].iloc[0]
            lines.append(
                f"| {label} | {labels[candidate['pool']]} | {q(row.mean_excess)} | "
                f"{q(row.mean_excess - row.mean_pool_excess)} | {q(row.mean_pool_excess)} |"
            )
    lines.extend(
        [
            "",
            "## 冻结数据结束后的短期检查",
            "",
            f"区间为{scope['recent_window']['start']}至{scope['recent_window']['end']}，"
            f"仅{len(scope['recent_window']['days'])}个交易日，不能按完整30日目标验收。",
            "",
            "| 股票池 | 封存候选 | 净收益 | 对本池指数超额 | 对齐全收益指数超额 | 最大回撤 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in selected["finalists"]:
        part = aggregates["recent"]
        row = part.loc[
            (part.pool == candidate["pool"])
            & (part.candidate == candidate["id"])
            & (part.cost_bps == 10)
        ].iloc[0]
        lines.append(
            f"| {labels[candidate['pool']]} | {candidate['id']} | {p(row.mean_return)} | "
            f"{q(row.mean_excess)} | {q(row.tr_excess)} | {p(row.worst_drawdown)} |"
        )
    download = json.loads((OUT / "validation_download_receipt.json").read_text(encoding="utf-8"))
    no_quotes = [x for x in download["outcomes"] if x["status"] == "no_quotes"]
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert verification["technical_status"] == "PASS"
    ledger_error = max(
        c["independent_cash_quantity_nav_error"] for c in verification["cohorts"].values()
    )
    prior_gaps = sum(
        c.get("quote_audit", {}).get("prior_close_gaps_over_10bps", 0)
        for c in verification["cohorts"].values()
    )
    lines.extend(
        [
            "",
            "## 实现和边界",
            "",
            "- 开发区实际回放1,620个策略账户及360个独立同池对照；"
            "表内重复展示的同池对照不重复计为计算账户。"
            "原规则复现最大收益差约为9.9e-17。",
            f"- 从逐笔交易独立重算现金和数量后，每日净值最大差为{ledger_error:.3g}；"
            "手续费账本、费用敏感性、训练标签边界及封存哈希检查通过。"
            f"新报价中昨收与上一有效收盘偏差超过10bps的记录共{prior_gaps}条，"
            "逐项保存在各验证目录的`quote_prior_close_gaps.csv`。"
            "这些记录不在账户评估日内，但可能进入回看特征和训练数据；"
            "特殊公司行动和复权连续性尚未逐项修复核验。",
            "- 验证报价来自BaoStock前复权日线；历史成分股按日期查询，"
            "含当时存在、现在可能已退出的股票。"
            f"下载无报价任务数为{len(no_quotes)}；完整逐项下载状态保存在回执中。"
            "数据商每周成分更新、少量499只记录及复权版本仍限制精度，不使用最新名单补历史。",
            "- 验证使用各窗口之前的数据重新拟合相同模型参数；"
            "训练/验证/测试边界清除尚未完整实现的标签。"
            "财报、行业、市值数据仍未进入选股。特征缺失处理仅从训练期拟合。",
            "- 创业板2020-08-24之前按10%规则，之后按20%规则；科创板采用20%。"
            "另修正验证引擎将创业板/科创板ST卖出误按5%限制处理的问题，买入仍排除ST。"
            "近期验证按2026-07-06起主板ST涨跌幅10%的新规则处理。"
            "验证按完整30开头代码范围识别创业板，包括302132。"
            "开发旧引擎及旧结果保留，验证候选与对照共同使用修正后的执行逻辑。",
            "- 使用分数股，没有整手、最低佣金、盘中盘口与冲击模型；单边10/20bps是统一费用场景。"
            "全收益指数仅有收盘数据，双方采用第一日收盘至末日收盘对齐，不能与开盘主口径互换。",
            "- 六段历史结果是以前未看过的压力检验；其日期早于用于设计方案的2024—2026数据，"
            "仍有反向时间选择风险。近期9日区间虽晚于冻结快照，也属于事后历史模拟，不是已运行的实时模拟账户。",
            "- 本轮尚未满足全部开发门槛，目标不得标为已经完成；"
            "研究候选单独保存，生产配置保持不变。",
            "",
            "## 下一步研究方向",
            "",
            "当前证据支持先检查评分的选股增益与收益来源："
            "沪深300候选在开发和新历史中均落后同池等权，中证500的放量门槛改善未迁移到新历史。"
            "后续应以指数增强为框架，明确主动偏离和换手预算，检验评分是否能在同池对照上增加收益。"
            "这是待验证的研究方向，不是已经证明有效的新方案。",
            "",
            "现有15+6段数据现在都已看过，只能继续用于开发和压力检验。"
            "下一轮应先冻结评分、执行约束与验收口径，再进行时间顺序向前的验证；"
            "不能通过在这21段上反复选择赢家来宣称已经解决超额问题。",
            "",
            "完整输出位于`logs/index-improvement-20261016/`："
            "`development/target_gates.csv`披露全部方案；`finalists_locked.json`记录封存选择；"
            "`external/`与`recent/`包含逐日净值、交易、同池对照和所有四个指数及对应全收益指数。",
            "",
            "规则来源：[深交所2020-08-21说明](https://www.szse.cn/aboutus/trends/conference/t20200821_580925.html)、"
            "[创业板交易特别规定说明](https://investor.szse.cn/institute/rules/t20200807_580310.html)。",
            "主板ST规则来源：[上交所2026年规则修订说明]"
            "(https://big5.sse.com.cn/site/cht/www.sse.com.cn/aboutus/mediacenter/hotandd/"
            "c/c_20260424_10816474.shtml)、[深交所风险警示板2026年指南]"
            "(https://www.szse.cn/lawrules/service/member/t20260630_621404.html)。",
            "代码范围来源：[深交所证券代码编码规范]"
            "(https://www.szse.cn/marketServices/technicalservice/doc/P020241212550140892927.pdf)。",
        ]
    )
    text = "\n".join(lines) + "\n"
    destination = ROOT / "docs/research/index_improvement_results_20261016.md"
    destination.write_text(text, encoding="utf-8")
    (OUT / "report.md").write_text(text, encoding="utf-8")
    (OUT / "verdicts.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    research = {
        "research_only": True,
        "production_release": False,
        "selected": selected,
        "verdicts": verdicts,
    }
    (ROOT / "config/research/index_improvement_finalists_20261016.json").write_text(
        json.dumps(research, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "Historical validation",
        aggregates["external"]
        .loc[
            aggregates["external"].cost_bps == 10,
            [
                "pool",
                "candidate",
                "mean_return",
                "mean_excess",
                "wins",
                "drop_best_excess",
                "worst_drawdown",
            ],
        ]
        .to_string(index=False),
        flush=True,
    )
    print(
        "Recent partial",
        aggregates["recent"]
        .loc[
            aggregates["recent"].cost_bps == 10,
            ["pool", "candidate", "mean_return", "mean_excess", "worst_drawdown"],
        ]
        .to_string(index=False),
        flush=True,
    )
    print("Saved", destination, flush=True)


if __name__ == "__main__":
    main()
