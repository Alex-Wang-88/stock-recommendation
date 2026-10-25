"""Lock a second, finite hypothesis suite before calculating its strategy returns."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-direction-study-20261015"
PREVIOUS = ROOT / "logs/market-rule-optimization-20261014"


def design():
    previous = json.loads((PREVIOUS / "candidate_suite_locked.json").read_text(encoding="utf-8"))
    source = {c["id"]: c for c in previous["candidates"]}
    controls = [
        {**source[old], "id": new, "previous_id": old, "family": "control"}
        for new, old in [("C00", "O00"), ("C01", "O04"), ("C02", "O09")]
    ]
    specs = [
        ("R01", "5日追涨对照", "trend5", 5, {}),
        ("R02", "5日反转/5日调仓", "reverse5", 5, {}),
        ("R03", "5日反转/10日调仓", "reverse5", 10, {}),
        ("R04", "10日反转/10日调仓", "reverse10", 10, {}),
        ("R05", "20日反转/10日调仓", "reverse20", 10, {}),
        ("R06", "中期趋势剔除最近5日", "trend60_skip5", 10, {}),
        ("R07", "中期趋势+短期回撤", "trend_pullback", 10, {}),
        ("R08", "短期反转+风控流动性", "reverse_risk", 10, {}),
        ("R09", "5日及20日双反转", "dual_reverse", 10, {}),
        ("R10", "过去均线状态切换", "regime_switch", 10, {}),
        ("R11", "反转风控/50只等权", "reverse_risk", 10, {"positions": 50, "retain_rank": 75}),
        (
            "R12",
            "反转风控/50只波动率分配",
            "reverse_risk",
            10,
            {"positions": 50, "retain_rank": 75, "weighting": "inverse_vol"},
        ),
        ("R13", "反转风控/市场波动仓位控制", "reverse_risk", 10, {"market_vol_target": 0.15}),
        ("R14", "趋势回撤/市场波动仓位控制", "trend_pullback", 10, {"market_vol_target": 0.15}),
        ("R15", "状态切换/市场波动仓位控制", "regime_switch", 10, {"market_vol_target": 0.15}),
        ("R16", "5日反转/20日调仓", "reverse5", 20, {}),
        ("R17", "10日反转/5日调仓", "reverse10", 5, {}),
        ("R18", "60日原始趋势对照", "trend60", 10, {}),
        ("R19", "原低换手方案/25只仓位控制", "P4", 10, {"market_vol_target": 0.15}),
        ("R20", "反转风控/取消放量门槛", "reverse_risk", 10, {"volume_min": 0.0}),
        ("R21", "5日反转/取消放量门槛", "reverse5", 10, {"volume_min": 0.0}),
    ]
    candidates = controls + [
        {
            "id": code,
            "name": name,
            "family": scorer,
            "scorer": scorer,
            "rebalance_days": days,
            "positions": 25,
            "retain_rank": 40,
            "volume_min": 1.1,
            "weighting": "equal",
            "market_vol_target": None,
            **options,
        }
        for code, name, scorer, days, options in specs
    ]
    pairs = [
        ("reverse_vs_chase", "R01", "R02", "5日反转相对5日追涨"),
        ("reverse5_10_vs_5", "R02", "R03", "5日反转：10日相对5日调仓"),
        ("reverse5_20_vs_10", "R03", "R16", "5日反转：20日相对10日调仓"),
        ("reverse10_10_vs_5", "R17", "R04", "10日反转：10日相对5日调仓"),
        ("skip_recent5", "R18", "R06", "中期趋势剔除最近5日"),
        ("trend_plus_pullback", "R06", "R07", "中期趋势加入回撤及风控"),
        ("reverse_plus_risk", "R03", "R08", "短期反转加入风控及流动性"),
        ("broader_holdings", "R08", "R11", "25只改为50只持仓"),
        ("inverse_volatility", "R11", "R12", "50只等权改为波动率倒数分配"),
        ("market_risk_reverse", "R08", "R13", "反转风控加入市场仓位控制"),
        ("market_risk_trend", "R07", "R14", "趋势回撤加入市场仓位控制"),
        ("market_risk_switch", "R10", "R15", "状态切换加入市场仓位控制"),
        ("market_risk_original", "C01", "R19", "原低换手方案加入市场仓位控制"),
        ("remove_volume_reverse_risk", "R08", "R20", "反转风控取消放量门槛"),
        ("remove_volume_reverse", "R03", "R21", "5日反转取消放量门槛"),
        ("state_switch_vs_trend", "R07", "R10", "均线状态切换相对固定趋势回撤"),
    ]
    return {
        "schema_version": 2,
        "defined_on": "2026-10-15",
        "expected_candidates": 24,
        "scope": "second finite adaptive research round: 21 new rules plus 3 controls",
        "common": previous["common"],
        "additional_factors": ["ret_5", "ret_10", "ret_60_skip5"],
        "scorer_definitions": {
            "trend5": "pct_rank(ret_5)",
            "reverse5": "pct_rank(-ret_5)",
            "reverse10": "pct_rank(-ret_10)",
            "reverse20": "pct_rank(-ret_20)",
            "trend60": "pct_rank(ret_60)",
            "trend60_skip5": "pct_rank(close[t-5]/close[t-60]-1)",
            "trend_pullback": ".5*rank(ret_60_skip5)+.3*rank(-ret_5)+.2*mean(three risk ranks)",
            "reverse_risk": ".5*rank(-ret_5)+.3*mean(three risk ranks)+.2*rank(turnover_value_20)",
            "dual_reverse": ".5*rank(-ret_5)+.5*rank(-ret_20)",
            "regime_switch": (
                "trend_pullback if CSI All Share previous-close level > "
                "previous-close MA60; otherwise reverse_risk"
            ),
            "P4": "unchanged original market-only P4 scorer",
        },
        "candidates": candidates,
        "paired_hypotheses": [
            {"id": code, "base": base, "alternative": alt, "name": name}
            for code, base, alt, name in pairs
        ],
        "pattern_consistency_rule": {
            "definition": "sensitivity screen, not a statistical-significance or causal claim",
            "mean_and_median_effect_same_sign": True,
            "same_direction_fraction_min": 2 / 3,
            "every_calendar_year_mean_same_direction": True,
            "20bps_mean_same_direction": True,
            "mean_without_best_effect_window_same_direction": True,
            "two_sided_reporting": True,
        },
        "analysis": {
            "cohorts": ["original_14", "all_15_viewed"],
            "primary_cost_bps": 10,
            "market_regimes": (
                "realized full-window index return >5%, <-5%, otherwise sideways; "
                "descriptive only, never a trading input"
            ),
            "signal_regime": (
                "index close vs MA60 observed before window entry; also report separately"
            ),
            "robustness": (
                "year, realized market regime, median, win fraction, worst loss/drawdown, "
                "remove best window, 20bps costs, paired changes"
            ),
            "held_out_years": (
                "do not claim new held-out data; all 15 periods viewed in earlier rounds"
            ),
        },
        "attempts": {
            "previous_rule_variants": 12,
            "new_rule_variants": 21,
            "cumulative_rule_variants": 33,
            "earlier_weight_profiles": 8,
            "interpretation": (
                "counts describe these two rule rounds; "
                "do not erase earlier full-factor experiments"
            ),
        },
        "validation_status": (
            "all existing 15 periods are research data; no new out-of-sample acceptance"
        ),
        "no_production_change": True,
    }


def main():
    from prepare_market_weight_batch import sha256

    OUT.mkdir(exist_ok=True)
    path = ROOT / "config/research/market_direction_candidates.json"
    suite = design()
    assert len(suite["candidates"]) == 24
    assert len({c["id"] for c in suite["candidates"]}) == 24
    serialized = json.dumps(suite, ensure_ascii=False, indent=2)
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == suite
    else:
        path.write_text(serialized, encoding="utf-8")
    old = json.loads((PREVIOUS / "windows_locked.json").read_text(encoding="utf-8"))
    windows = [
        {**w, "prior_role": w["role"], "role": "previously_viewed_research"} for w in old["windows"]
    ]
    lock = {
        "market_sha256": old["market_sha256"],
        "candidate_sha256": sha256(path),
        "target_sha256": sha256(PREVIOUS / "targets_locked.json"),
        "windows": windows,
        "new_holdout_included": False,
        "source_previous_study": str(PREVIOUS),
        "hypotheses_locked_before_current_batch_returns": True,
    }
    destination = OUT / "windows_locked.json"
    if destination.exists():
        assert json.loads(destination.read_text(encoding="utf-8")) == lock
    else:
        destination.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "candidate_suite_locked.json").write_bytes(path.read_bytes())
    (OUT / "targets_locked.json").write_bytes((PREVIOUS / "targets_locked.json").read_bytes())
    print(
        "Locked 24 candidates, 16 paired comparisons and 15 previously viewed windows", flush=True
    )


if __name__ == "__main__":
    main()
