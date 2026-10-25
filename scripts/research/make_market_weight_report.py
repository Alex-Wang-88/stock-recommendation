"""Summarize every declared account without selecting extra weights after returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL_NAMES = {"original_low_position": "原低位池", "broader_liquid": "较宽候选池"}
EXEC_NAMES = {
    "cycle_close_open": "5日收盘退出/次期开盘入场",
    "continuous_net": "连续持仓、差额调仓",
}
WINDOW_NAMES = {
    "sideways": "震荡",
    "recovery_reversal": "先跌后修复",
    "strong_up": "强上涨",
    "strong_down": "强下跌",
    "recent_holdout": "最近期（已查看）",
    "unseen_calendar": "1—3月（已查看）",
}


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def percent(value):
    return f"{value * 100:+.2f}%"


def points(value):
    return f"{value * 100:+.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=ROOT / "logs/market-only-weight-results-20261014"
    )
    args = parser.parse_args()
    out = args.results
    frame = pd.read_csv(out / "summary.csv")
    suite = json.loads((out / "suite_locked.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "plan_locked.json").read_text(encoding="utf-8"))
    qa = json.loads((out / "qa.json").read_text(encoding="utf-8"))
    assert qa["technical_status"] == "PASS" and len(frame) == qa["completed_account_jobs"]
    window_count = len(plan["windows"])
    window_names = {
        window["label"]: window.get(
            "display_name", WINDOW_NAMES.get(window["label"], window["label"])
        )
        for window in plan["windows"]
    }
    fresh = bool(qa.get("fresh_validation_included"))
    names = {profile["id"]: profile["name"] for profile in suite["profiles"]}
    order = list(names)
    accounts = frame.loc[frame.profile.isin(names)].copy()
    groups = accounts.groupby(["pool", "execution", "one_way_cost_bps", "profile"], sort=False)
    aggregates = []
    for (pool, execution, bps, profile), group in groups:
        baseline = accounts.loc[
            (accounts.pool == pool)
            & (accounts.execution == execution)
            & (accounts.one_way_cost_bps == bps)
            & (accounts.profile == order[0])
        ].set_index("window")
        selected = group.set_index("window")
        delta = selected["return"] - baseline["return"]
        aggregates.append(
            {
                "pool": pool,
                "execution": execution,
                "one_way_cost_bps": bps,
                "profile": profile,
                "name": names[profile],
                "windows": len(group),
                "mean_return": group["return"].mean(),
                "mean_pool_alpha": group.excess_pool_same_cost.mean(),
                "median_pool_alpha": group.excess_pool_same_cost.median(),
                "mean_market_alpha": group.excess_market_gross.mean(),
                "positive_pool_alpha_windows": int((group.excess_pool_same_cost > 1e-10).sum()),
                "better_than_P0_windows": int((delta > 1e-10).sum()),
                "mean_delta_P0": delta.mean(),
                "worst_window_drawdown": group.max_drawdown.min(),
                "worst_window_return": group["return"].min(),
                "mean_traded_notional": group.total_traded_notional.mean(),
                "mean_fees": group.total_fees.mean(),
                "unfilled_exit_events": group.unfilled_exit_events.sum(),
                "max_residual_positions": group.residual_positions.max(),
            }
        )
    aggregate = pd.DataFrame(aggregates)
    aggregate.to_csv(out / "aggregate.csv", index=False, encoding="utf-8-sig")
    primary = aggregate.loc[
        (aggregate.execution == "continuous_net") & (aggregate.one_way_cost_bps == 10)
    ]
    primary_leaders = primary.loc[primary.groupby("pool").mean_pool_alpha.idxmax()]
    headline = "；".join(
        f"{POOL_NAMES[row.pool]}：{row['name']}平均同池超额"
        f"{points(row.mean_pool_alpha)}个百分点、"
        f"{int(row.positive_pool_alpha_windows)}/{int(row.windows)}段为正、"
        f"最差窗口回撤{percent(row.worst_window_drawdown)}"
        for _, row in primary_leaders.iterrows()
    )
    headline_note = (
        "这两组可作为后续候选，但胜率和回撤尚未支持稳定超额的结论。"
        if (primary_leaders.mean_pool_alpha > 0).all()
        else "本批领先组的平均同池超额仍为负，尚不支持任何被测权重具有稳定增益。"
    )
    blocks = [
        "# 八组价量权重：完整批量回放结果",
        f"已完成 {qa['completed_account_jobs']} 个账户：{len(accounts)}个策略账户、"
        f"{int((frame.profile == 'same_pool_equal_same_cost').sum())}个同池等权账户、"
        f"{int((frame.profile == 'market_equal_gross').sum())}个市场毛收益参考。"
        "计算与隔离校验 PASS。"
        "生产默认配置未修改；本轮不读取财报、行业文件或PE。",
        "每侧10基点、连续差额调仓下，" + headline + "。" + headline_note,
        "八组参数在计算本轮收益前已固定。因子仅包括20/60日收益、量比、20日波动率、60日回撤、20日下行波动率和20日日均成交额。类别内先做同池横截面分位平均，再按权重相加。",
        "每段30个交易日、5日一期、最多25只；八组共享完整因子候选集合和资格条件。原低位池要求250日位置≤35%、距高点回落≥20%；较宽池仅取消两项低位限制。成本为每侧0、10、20基点。主表使用每侧10基点，即0.10%。",
        (
            f"本轮{window_count}段为选窗时尚未查看策略收益的新历史区间，与此前六段持有日不重叠，"
            "八组权重在本轮选窗前已经固定。它们是向较早历史时期扩展的时间验证，"
            "不是向未来的滚动验证。"
            if fresh
            else f"所有{window_count}段均已查看，身份统一为开发复验。这次没有新的未查看时间区间。"
        )
        + f"均值是{window_count}段30日账户收益/超额的算术平均，未年化，"
        "不能把它解释为连续实盘组合收益。",
        "## 时间区间",
        table(
            ["区间", "开始", "结束", "身份"],
            [
                [window_names[window["label"]], window["start"], window["end"], window["role"]]
                for window in plan["windows"]
            ],
        ),
        "## 固定权重",
        table(
            ["方案", "动量", "风控", "流动性"],
            [
                [
                    profile["id"].split("_")[0] + " " + profile["name"],
                    *[
                        f"{profile['weights'][key] * 100:.2f}%"
                        for key in ("momentum", "risk", "liquidity")
                    ],
                ]
                for profile in suite["profiles"]
            ],
        ),
        f"## {window_count}段均值与风险：所有方案",
    ]
    leaders = []
    for pool, pool_name in POOL_NAMES.items():
        for execution, execution_name in EXEC_NAMES.items():
            selected = aggregate.loc[
                (aggregate.pool == pool)
                & (aggregate.execution == execution)
                & (aggregate.one_way_cost_bps == 10)
            ].set_index("profile")
            rows = [
                [
                    names[profile],
                    percent(selected.loc[profile, "mean_return"]),
                    points(selected.loc[profile, "mean_pool_alpha"]),
                    points(selected.loc[profile, "median_pool_alpha"]),
                    f"{int(selected.loc[profile, 'positive_pool_alpha_windows'])}/{window_count}",
                    points(selected.loc[profile, "mean_market_alpha"]),
                    percent(selected.loc[profile, "worst_window_drawdown"]),
                ]
                for profile in order
            ]
            blocks += [
                f"### {pool_name} · {execution_name}",
                table(
                    [
                        "方案",
                        "平均净收益",
                        "平均同池超额/百分点",
                        "同池超额中位数/百分点",
                        f"胜过同池/{window_count}",
                        "平均对市场毛收益超额/百分点",
                        "最差窗口回撤",
                    ],
                    rows,
                ),
            ]
            winner = selected.sort_values("mean_pool_alpha", ascending=False).iloc[0]
            leaders.append(
                {
                    "pool": pool,
                    "execution": execution,
                    "profile": winner.name,
                    "name": winner["name"],
                    "mean_return": float(winner.mean_return),
                    "mean_pool_alpha": float(winner.mean_pool_alpha),
                    "positive_pool_alpha_windows": int(winner.positive_pool_alpha_windows),
                    "mean_market_alpha": float(winner.mean_market_alpha),
                    "worst_window_drawdown": float(winner.worst_window_drawdown),
                }
            )
    blocks += [
        "同池超额使用相同费用、相同执行规则的候选池等权账户；两层候选池分别有自己的基准。市场参考为存档中具有历史报价/交易状态且可入场的沪深股票等权账户毛收益，不是官方指数。排序方案和基准均处理无法成交的持仓；未卖出的资产继续计价并占用资金。",
        "## 扣成本收益：逐段公开全部八组",
    ]
    for pool, pool_name in POOL_NAMES.items():
        for execution, execution_name in EXEC_NAMES.items():
            subset = frame.loc[
                (frame.pool == pool)
                & (frame.execution == execution)
                & (frame.one_way_cost_bps == 10)
            ].set_index(["window", "profile"])
            rows = []
            for window in plan["windows"]:
                key = window["label"]
                market = frame.loc[
                    (frame.window == key)
                    & (frame.profile == "market_equal_gross")
                    & (frame.execution == execution),
                    "return",
                ].iloc[0]
                rows.append(
                    [
                        window_names[key],
                        *[percent(subset.loc[(key, profile), "return"]) for profile in order],
                        percent(subset.loc[(key, "same_pool_equal_same_cost"), "return"]),
                        percent(market),
                    ]
                )
            blocks += [
                f"### {pool_name} · {execution_name}",
                table(
                    [
                        "行情",
                        *[profile.split("_")[0] for profile in order],
                        "同池净收益",
                        "市场毛收益",
                    ],
                    rows,
                ),
            ]
    blocks += [
        "## 成本敏感性：平均同池超额",
        "各成本下对照均采用同成本同池等权账户，不用零成本基准给有成本策略打分。",
    ]
    for pool, pool_name in POOL_NAMES.items():
        for execution, execution_name in EXEC_NAMES.items():
            subset = aggregate.loc[
                (aggregate.pool == pool) & (aggregate.execution == execution)
            ].set_index(["profile", "one_way_cost_bps"])
            rows = [
                [
                    names[profile],
                    *[points(subset.loc[(profile, bps), "mean_pool_alpha"]) for bps in (0, 10, 20)],
                ]
                for profile in order
            ]
            blocks += [
                f"### {pool_name} · {execution_name}",
                table(["方案", "零成本/百分点", "每侧10基点/百分点", "每侧20基点/百分点"], rows),
            ]
    blocks += [
        "## 交易与时点影响",
        "连续差额账户和5日收盘退出账户的隔夜敞口不同。下表收益变化包括时点和成本等影响，不能全部归因于减少成交。连续账户按成交金额收买入费；收盘退出账户沿用上一轮按分配资金扣除买入成本的公式，同一执行方式内各方案口径一致。",
    ]
    rows = []
    for pool, pool_name in POOL_NAMES.items():
        subset = aggregate.loc[
            (aggregate.pool == pool) & (aggregate.one_way_cost_bps == 10)
        ].set_index(["profile", "execution"])
        for profile in order:
            cycle = subset.loc[(profile, "cycle_close_open")]
            net = subset.loc[(profile, "continuous_net")]
            decrease = 1 - net.mean_traded_notional / cycle.mean_traded_notional
            rows.append(
                [
                    pool_name,
                    names[profile],
                    points(net.mean_return - cycle.mean_return),
                    f"{decrease * 100:.1f}%",
                    f"{net.mean_fees * 10000:.1f}",
                    int(net.unfilled_exit_events),
                    int(net.max_residual_positions),
                ]
            )
    blocks += [
        table(
            [
                "池",
                "方案",
                "差额账户平均收益变化/百分点",
                "平均成交额减少",
                "差额费用/初始资金基点",
                "差额退出未成交次数",
                "最大期末残留头寸",
            ],
            rows,
        ),
        "## 已完成校验与剩余边界",
        f"{len(frame)}个声明任务全部完成，权重与因子缓存SHA256在执行前后保持一致；"
        "被剔除字段扰动不影响评分；八组共享候选池和完整数据门槛；独立的周期收益公式与现金账本一致；停牌保留、费用守恒、现金非负和持仓上限检查通过。详情见qa.json和input_receipts.json。",
        "剔除财报和行业因子消除了本轮对这些输入的依赖，不能消除历史股票池存续偏差或复权版本问题。涨跌停用日线阈值保守近似，缺少逐笔成交/排队、整手约束和真实费用拟合。"
        + (
            "新增历史样本用于检验原先预选的方案；从本轮八组中再次择优得到的组，仍需另行验证。"
            if fresh
            else "已查看的区间不能据此宣称找到最优权重或完成样本外验证。"
        ),
        "## 本批中的领先组（按预先固定的平均同池超额排序）",
        table(
            [
                "池",
                "执行",
                "领先组",
                "平均同池超额/百分点",
                f"胜过同池/{window_count}",
                "最差窗口回撤",
            ],
            [
                [
                    POOL_NAMES[row["pool"]],
                    EXEC_NAMES[row["execution"]],
                    row["name"],
                    points(row["mean_pool_alpha"]),
                    f"{row['positive_pool_alpha_windows']}/{window_count}",
                    percent(row["worst_window_drawdown"]),
                ]
                for row in leaders
            ],
        ),
        "该排序只用于缩小后续候选范围，仍应结合逐段稳定性、市场参考和回撤审视；本轮没有更改线上权重。",
        "## 输出文件与复现",
        f"- summary.csv：{len(frame)}个账户的收益、超额、费用与交易约束。\n"
        "- aggregate.csv：96个方案/池/执行/成本汇总组合。\n"
        "- daily_nav.csv：全部账户每日净值。\n"
        "- holdings.csv、target_schedules.json、cycles.csv：冻结的目标股票与入场排除。\n"
        "- trades_10bps.csv：全部策略账户在每侧10基点下的实际成交账本。\n"
        "- account_events.csv：各成本下未成交/保留事件。\n"
        "- qa.json、input_receipts.json、plan_locked.json、suite_locked.json：校验与不可变协议。",
        "```powershell\n.venv\\Scripts\\python.exe "
        "scripts/research/run_market_weight_batch.py "
        "--output logs/market-only-weight-results-new\n"
        ".venv\\Scripts\\python.exe scripts/research/make_market_weight_report.py "
        "--results logs/market-only-weight-results-new\n```",
        "默认复用本地日志里的锁定配置、因子、价格和历史交易状态缓存；这些大输入未纳入Git。脚本不会覆盖已存在的summary.csv，复验应指定新的输出目录。",
    ]
    (out / "report.md").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    (out / "cohort_leaders.json").write_text(
        json.dumps(leaders, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        aggregate.loc[
            (aggregate.execution == "continuous_net") & (aggregate.one_way_cost_bps == 10),
            [
                "pool",
                "profile",
                "mean_return",
                "mean_pool_alpha",
                "mean_market_alpha",
                "positive_pool_alpha_windows",
                "worst_window_drawdown",
            ],
        ].to_string(index=False)
    )
    print(out / "report.md")


if __name__ == "__main__":
    main()
