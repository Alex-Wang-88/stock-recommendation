"""Compare prior development results with the fixed-weight new historical time cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from make_market_weight_report import EXEC_NAMES, POOL_NAMES, percent, points, table

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-results",
        type=Path,
        default=ROOT / "logs/market-only-extended-validation-20261014/results",
    )
    parser.add_argument(
        "--old-results", type=Path, default=ROOT / "logs/market-only-weight-results-20261014"
    )
    args = parser.parse_args()
    new = pd.read_csv(args.new_results / "summary.csv")
    old = pd.read_csv(args.old_results / "summary.csv")
    new["cohort"] = "new_time_validation"
    old["cohort"] = "development_previously_viewed"
    plan = json.loads((args.new_results / "plan_locked.json").read_text(encoding="utf-8"))
    suite = json.loads((args.new_results / "suite_locked.json").read_text(encoding="utf-8"))
    assert suite == json.loads((args.old_results / "suite_locked.json").read_text(encoding="utf-8"))
    assert plan["fresh_validation_included"] is True
    names = {profile["id"]: profile["name"] for profile in suite["profiles"]}
    old_days = set(pd.read_csv(args.old_results / "daily_nav.csv").day)
    new_days = set(pd.read_csv(args.new_results / "daily_nav.csv").day)
    assert not new_days & old_days
    combined = pd.concat([old, new], ignore_index=True)
    combined.to_csv(args.new_results / "combined_summary.csv", index=False, encoding="utf-8-sig")
    rows = []
    for pool in POOL_NAMES:
        for execution in EXEC_NAMES:
            for bps in (0, 10, 20):
                for profile in names:
                    prior = old.loc[
                        (old.pool == pool)
                        & (old.execution == execution)
                        & (old.one_way_cost_bps == bps)
                        & (old.profile == profile)
                    ]
                    fresh = new.loc[
                        (new.pool == pool)
                        & (new.execution == execution)
                        & (new.one_way_cost_bps == bps)
                        & (new.profile == profile)
                    ]
                    all_rows = pd.concat([prior, fresh], ignore_index=True)
                    rows.append(
                        {
                            "pool": pool,
                            "execution": execution,
                            "one_way_cost_bps": bps,
                            "profile": profile,
                            "name": names[profile],
                            "old_windows": len(prior),
                            "new_windows": len(fresh),
                            "old_mean_pool_alpha": prior.excess_pool_same_cost.mean(),
                            "new_mean_pool_alpha": fresh.excess_pool_same_cost.mean(),
                            "new_median_pool_alpha": fresh.excess_pool_same_cost.median(),
                            "new_positive_alpha_windows": int(
                                (fresh.excess_pool_same_cost > 1e-10).sum()
                            ),
                            "new_mean_return": fresh["return"].mean(),
                            "new_mean_market_alpha": fresh.excess_market_gross.mean(),
                            "new_worst_drawdown": fresh.max_drawdown.min(),
                            "all_mean_pool_alpha": all_rows.excess_pool_same_cost.mean(),
                            "all_positive_alpha_windows": int(
                                (all_rows.excess_pool_same_cost > 1e-10).sum()
                            ),
                            "all_mean_return": all_rows["return"].mean(),
                            "all_worst_drawdown": all_rows.max_drawdown.min(),
                        }
                    )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.new_results / "cohort_comparison.csv", index=False, encoding="utf-8-sig")
    candidates = plan["previously_selected_candidates"]
    selected = comparison.loc[
        (comparison.execution == "continuous_net")
        & (comparison.one_way_cost_bps == 10)
        & comparison.apply(lambda row: row.profile == candidates[row.pool], axis=1)
    ]
    conclusion = (
        "新八段中，两组此前预选方案的平均同池超额均转为负值。"
        "新增证据不支持直接用这两组调整线上权重，应暂缓部署。"
        if (selected.new_mean_pool_alpha < 0).all()
        else "此前预选方案的表现见下表；应同时检查同池超额方向、区间胜率和回撤，"
        "不以平均净收益为正作为通过验证的条件。"
    )
    if (comparison.new_mean_pool_alpha < 0).all():
        conclusion += "全部八组在新八段中，按两种执行方式和三档成本计算的平均同池超额均为负。"
    blocks = [
        "# 新增八段历史验证：与原六段对照",
        conclusion,
        "八组权重保持上轮锁定值，新增八段各30个交易日，与原六段持有日完全不重叠，共新增240天。"
        "这批日期按季度/日历规则在策略收益计算前选定；历史ST与交易状态通过BaoStock补齐。",
        "新旧两批均不使用财报、行业、PE、题材或资金评分。原六段为开发样本，新八段为本次固定权重的历史时间验证。"
        "这些新月份较开发期更早，不是向未来的滚动验证；历史股票池完整性与复权版本限制仍存在。",
        "## 之前预选的两组：先看它们能否通过新样本",
        "主表为每侧10基点、连续差额调仓。原低位池预选P7仅风控；较宽池预选P4类别等权。"
        "它们在本轮日期锁定时已明确记录，没有用新结果改选权重。",
        table(
            [
                "池/预选方案",
                "原六段同池超额/百分点",
                "新八段同池超额/百分点",
                "新八段中位数/百分点",
                "新样本胜过同池",
                "新样本平均净收益",
                "新样本最差回撤",
            ],
            [
                [
                    POOL_NAMES[row.pool] + " / " + row["name"],
                    points(row.old_mean_pool_alpha),
                    points(row.new_mean_pool_alpha),
                    points(row.new_median_pool_alpha),
                    f"{int(row.new_positive_alpha_windows)}/{int(row.new_windows)}",
                    percent(row.new_mean_return),
                    percent(row.new_worst_drawdown),
                ]
                for _, row in selected.iterrows()
            ],
        ),
        "## 新的时间区间",
        table(
            ["区间", "开始", "结束"],
            [
                [window["display_name"], window["start"], window["end"]]
                for window in plan["windows"]
            ],
        ),
        "## 全部八组：两批分开报告",
    ]
    for pool, pool_name in POOL_NAMES.items():
        for execution, execution_name in EXEC_NAMES.items():
            group = comparison.loc[
                (comparison.pool == pool)
                & (comparison.execution == execution)
                & (comparison.one_way_cost_bps == 10)
            ].set_index("profile")
            blocks += [
                f"### {pool_name} · {execution_name}",
                table(
                    [
                        "方案",
                        "原六段平均同池超额/百分点",
                        "新八段平均同池超额/百分点",
                        "新八段中位数/百分点",
                        "新样本胜过同池",
                        "新样本最差回撤",
                        "合并14段平均同池超额/百分点",
                    ],
                    [
                        [
                            names[profile],
                            points(group.loc[profile, "old_mean_pool_alpha"]),
                            points(group.loc[profile, "new_mean_pool_alpha"]),
                            points(group.loc[profile, "new_median_pool_alpha"]),
                            f"{int(group.loc[profile, 'new_positive_alpha_windows'])}/"
                            f"{int(group.loc[profile, 'new_windows'])}",
                            percent(group.loc[profile, "new_worst_drawdown"]),
                            points(group.loc[profile, "all_mean_pool_alpha"]),
                        ]
                        for profile in names
                    ],
                ),
            ]
    blocks += [
        "## 解读口径",
        "优先检验预选方案在新八段的同池扣成本超额、胜率与回撤，不用合并14段的好看均值掩盖新旧反转。"
        "新八段中表现最好的其他权重只能作为后续候选，不能把本轮择优后的收益再当作该新权重的独立验证。"
        "市场参考为可取得历史数据的沪深股票等权毛收益，不是官方指数；两层池的同池基准也不同。",
        "没有调整生产默认配置。若新增证据不支持之前建议，就暂缓部署；若方向一致，也仍需更长和向未来的样本。",
        "完整的新八段逐段收益、全部成本和交易账本见同目录 "
        "report.md、summary.csv、trades_10bps.csv。"
        "新旧分组数值见 cohort_comparison.csv，合并但保留身份的账户明细见 combined_summary.csv。",
        "历史状态数据接口参考：[BaoStock官方包说明](https://pypi.org/project/baostock/)。",
    ]
    (args.new_results / "validation_comparison.md").write_text(
        "\n\n".join(blocks) + "\n", encoding="utf-8"
    )
    print(
        selected[
            [
                "pool",
                "profile",
                "old_mean_pool_alpha",
                "new_mean_pool_alpha",
                "new_positive_alpha_windows",
                "new_mean_return",
                "new_worst_drawdown",
            ]
        ].to_string(index=False)
    )
    print(args.new_results / "validation_comparison.md")


if __name__ == "__main__":
    main()
