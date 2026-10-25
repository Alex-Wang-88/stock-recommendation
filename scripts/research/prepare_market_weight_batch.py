"""Validate and freeze a price-only weight suite and future batch plan; no return replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMBERS = {
    "momentum": ("ret_20", "ret_60", "volume_trend_20"),
    "risk": ("volatility_value", "drawdown_value", "downside_volatility_value"),
    "liquidity": ("turnover_value_20",),
}
FACTORS = tuple(name for names in MEMBERS.values() for name in names)
POOL_FIELDS = (
    "day",
    "symbol",
    "history_days",
    "listed_days",
    "close_position_250",
    "dist_to_52w_high",
    "historical_isST",
    "historical_trade_status",
)
READ_COLUMNS = tuple(dict.fromkeys(POOL_FIELDS + FACTORS))


def validate_suite(suite: dict) -> None:
    if suite.get("schema_version") != 1:
        raise ValueError("Unsupported suite schema")
    supplied = {category: tuple(names) for category, names in suite["factor_members"].items()}
    if supplied != MEMBERS:
        raise ValueError("Factor members must match the fixed seven market-only inputs")
    ids, weight_sets = set(), set()
    if len(suite["profiles"]) != 8:
        raise ValueError("This version predeclares exactly eight profiles")
    for profile in suite["profiles"]:
        if profile["id"] in ids:
            raise ValueError("Duplicate profile id")
        ids.add(profile["id"])
        weights = profile["weights"]
        if set(weights) != set(MEMBERS):
            raise ValueError("Only momentum, risk and liquidity weights are allowed")
        if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights.values()):
            raise ValueError("Weights must be finite and nonnegative")
        if abs(sum(weights.values()) - 1) > 1e-12:
            raise ValueError("Profile weights must sum to one")
        key = tuple(weights[category] for category in MEMBERS)
        if key in weight_sets:
            raise ValueError("Duplicate weight profile")
        weight_sets.add(key)


def project_factors(frame):
    """Physical whitelist: neither financial nor industry/duplicate factors reach scoring."""
    import numpy as np
    import pandas as pd

    missing = set(("symbol",) + FACTORS) - set(frame.columns)
    if missing:
        raise ValueError(f"Required market factors missing: {sorted(missing)}")
    out = frame.loc[:, ["symbol", *FACTORS]].copy()
    if out.symbol.duplicated().any():
        raise ValueError("Duplicate symbols within one signal/pool")
    for name in FACTORS:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if not np.isfinite(out.loc[:, list(FACTORS)].to_numpy(dtype=float)).all():
        raise ValueError("Incomplete rows must be removed from all profiles' common pool first")
    return out


def score_profile(frame, profile):
    sys.path.insert(0, str(ROOT / "src"))
    from recommend.rank.score import add_scores

    return add_scores(project_factors(frame), profile["weights"]).sort_values(
        ["score", "symbol"], ascending=[False, True]
    )


def pool_frame(frame, pool_id: str, rules: dict):
    import numpy as np
    import pandas as pd

    missing = set(READ_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing price/status eligibility fields: {sorted(missing)}")
    out = frame.loc[:, list(READ_COLUMNS)].copy()
    for name in READ_COLUMNS:
        if name not in ("day", "symbol"):
            out[name] = pd.to_numeric(out[name], errors="coerce")
    complete = np.isfinite(out.loc[:, list(FACTORS)].to_numpy(dtype=float)).all(axis=1)
    eligible = (
        (out.history_days >= rules["history_days_min"])
        & (out.listed_days >= rules["listed_days_min"])
        & (out.volume_trend_20 >= rules["volume_trend_20_min"])
        & (out.turnover_value_20 >= rules["turnover_value_20_min"])
        & (out.historical_isST == rules["historical_isST"])
        & (out.historical_trade_status == rules["historical_trade_status"])
    )
    if pool_id == "original_low_position":
        eligible &= (out.close_position_250 <= 0.35) & (out.dist_to_52w_high <= -0.20)
    elif pool_id != "broader_liquid":
        raise ValueError(f"Unknown candidate pool: {pool_id}")
    return out.loc[eligible & complete].copy(), int((eligible & ~complete).sum())


def verify_isolation(suite: dict, sample) -> dict:
    import numpy as np

    polluted = sample.copy()
    for name in (
        "industry",
        "industry_score",
        "industry_leader_score",
        "yoy_net_income",
        "yoy_eps_basic",
        "roe_avg",
        "current_ratio",
        "cfo_to_np",
        "pe_ttm_value",
        "pe_industry_value",
        "relative_ret_20",
        "relative_ret_60",
        "ret_since_924",
        "theme_heat_max",
        "main_net_inflow",
        "margin_rz_chg_20",
    ):
        polluted[name] = np.arange(len(sample)) * 1e9 + 1e12
    for profile in suite["profiles"]:
        left = score_profile(sample, profile)
        right = score_profile(polluted, profile)
        assert left.symbol.tolist() == right.symbol.tolist()
        assert np.array_equal(left.score.to_numpy(), right.score.to_numpy())
        assert np.allclose(left.score_coverage, 1.0)
        assert left.score.between(0.0, 1.0 + 1e-12).all()
    incomplete = sample.copy()
    incomplete.loc[incomplete.index[0], "ret_20"] = np.nan
    try:
        score_profile(incomplete, suite["profiles"][-1])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing common factor silently reweighted")
    forbidden = json.loads(json.dumps(suite))
    forbidden["profiles"][0]["weights"]["industry"] = 0.0
    try:
        validate_suite(forbidden)
    except ValueError:
        pass
    else:
        raise AssertionError("Industry dependency accepted")
    return {
        "weights_unique_finite_nonnegative_sum_one": True,
        "financial_industry_and_duplicate_factor_perturbations_do_not_affect_scores": True,
        "all_profiles_require_same_complete_factor_pool": True,
        "score_coverage_one_on_common_pool": True,
        "forbidden_category_rejected_even_at_zero_weight": True,
        "full_batch_returns_executed": False,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "config/research/market_only_weights.json"
    )
    parser.add_argument(
        "--factor-file",
        type=Path,
        default=ROOT / "logs/local-optimization-test-20261014/candidate_factors_final.parquet",
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=ROOT / "logs/local-optimization-test-20261014/windows_locked.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / datetime.now().strftime("market-only-weight-design-%Y%m%d"),
    )
    parser.add_argument(
        "--window-role",
        choices=["development_previously_viewed", "new_time_validation"],
        default="development_previously_viewed",
    )
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    validate_suite(suite)
    # Small validation load only; no financial tables, industry references, or PE are read.
    import pandas as pd

    factors = pd.read_parquet(args.factor_file, columns=list(READ_COLUMNS))
    factors["symbol"] = factors.symbol.astype(str).str.zfill(6)
    factors["day"] = factors.day.astype(str).str[:10]
    if factors.duplicated(["day", "symbol"]).any():
        raise ValueError("Duplicate day/symbol rows in factor snapshot")
    window_source = json.loads(args.windows.read_text(encoding="utf-8"))
    source_windows = window_source["windows"]
    fresh_validation = args.window_role == "new_time_validation"
    excluded_days = set()
    if fresh_validation:
        if window_source.get("selected_before_strategy_returns") is not True:
            raise ValueError(
                "New validation needs a selection receipt from before strategy returns"
            )
        if window_source.get("suite_sha256") != sha256(args.suite):
            raise ValueError("Weights differ from the suite frozen when dates were selected")
        excluded_file = Path(window_source["excluded_windows_file"])
        if sha256(excluded_file) != window_source["excluded_windows_sha256"]:
            raise ValueError("Previously viewed date receipt changed")
        excluded_days = {
            day
            for window in json.loads(excluded_file.read_text(encoding="utf-8"))["windows"]
            for day in window["days"]
        }
    windows = []
    for original in source_windows:
        window = dict(original)
        window["source_role"] = original.get("role", "previously_viewed")
        window["role"] = args.window_role
        if len(window["days"]) != suite["batch_protocol"]["window_trading_days"]:
            raise ValueError("Wrong window length")
        if not (
            len(window["signal_dates"]) == len(window["entry_dates"]) == len(window["exit_dates"])
        ):
            raise ValueError("Inconsistent schedule lengths")
        for signal, entry in zip(window["signal_dates"], window["entry_dates"], strict=True):
            if signal >= entry:
                raise ValueError("Signal must precede entry")
        windows.append(window)
    holding_days = [day for window in windows for day in window["days"]]
    if len(set(holding_days)) != len(holding_days):
        raise ValueError("Windows overlap")
    if set(holding_days) & excluded_days:
        raise ValueError("New validation overlaps previously viewed holding dates")
    settings = suite["batch_protocol"]
    coverage, pool_cache = [], {}
    for signal in sorted({day for window in windows for day in window["signal_dates"]}):
        group = factors.loc[factors.day == signal].copy()
        if group.empty:
            raise ValueError(f"Signal data absent: {signal}")
        for pool in settings["pool_groups"]:
            selected, incomplete = pool_frame(group, pool["id"], settings["shared_pool_rules"])
            if selected.empty:
                raise ValueError(f"No complete candidates: {signal}, {pool['id']}")
            pool_cache[(signal, pool["id"])] = selected
            coverage.append(
                {
                    "signal": signal,
                    "pool": pool["id"],
                    "common_complete_candidates": len(selected),
                    "incomplete_candidates_excluded_for_all_profiles": incomplete,
                }
            )
    first = pool_cache[sorted(pool_cache)[0]].head(100)
    checks = verify_isolation(suite, first)
    checks["signals"] = len({key[0] for key in pool_cache})
    checks["pool_groups"] = len(settings["pool_groups"])
    jobs = []
    for window, pool, profile, bps, execution in product(
        windows,
        settings["pool_groups"],
        suite["profiles"] + [{"id": "same_pool_equal_same_cost"}],
        settings["one_way_cost_bps"],
        settings["executions"],
    ):
        jobs.append(
            {
                "window": window["label"],
                "role": window["role"],
                "pool": pool["id"],
                "profile": profile["id"],
                "one_way_cost_bps": bps,
                "execution": execution,
            }
        )
    market_jobs = [
        {
            "window": window["label"],
            "profile": "market_equal_gross",
            "execution": execution,
            "one_way_cost_bps": 0,
            "role": window["role"],
        }
        for window, execution in product(windows, settings["executions"])
    ]
    plan = {
        "status": "prepared_not_executed",
        "suite_id": suite["suite_id"],
        "suite_file": str(args.suite.resolve()),
        "suite_sha256": sha256(args.suite),
        "factor_file": str(args.factor_file.resolve()),
        "factor_sha256": sha256(args.factor_file),
        "factor_columns_actually_read": list(READ_COLUMNS),
        "source_windows_file": str(args.windows.resolve()),
        "source_windows_sha256": sha256(args.windows),
        "windows": windows,
        "settings": settings,
        "coverage": coverage,
        "profile_account_jobs": sum(job["profile"] != "same_pool_equal_same_cost" for job in jobs),
        "same_pool_benchmark_jobs": sum(
            job["profile"] == "same_pool_equal_same_cost" for job in jobs
        ),
        "market_reference_jobs": len(market_jobs),
        "account_jobs": jobs,
        "market_reference_jobs_plan": market_jobs,
        "fresh_validation_included": fresh_validation,
        "previously_selected_candidates": window_source.get("previously_selected_candidates", {}),
        "remaining_data_limitations": [
            "历史股票存档仍不保证完整时点股票池。",
            "日线复权口径和涨跌停成交近似仍需要单独说明。",
            "新增区间是对固定权重的历史时间验证，并非向未来的滚动验证。"
            if fresh_validation
            else "已查看的六段只能作为开发复验；没有在本轮新增未查看的验证数据。",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in [
        ("suite_locked.json", suite),
        ("batch_plan.json", plan),
        ("qa.json", checks),
    ]:
        (args.output / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    pd.DataFrame(coverage).to_csv(
        args.output / "factor_coverage.csv", index=False, encoding="utf-8-sig"
    )
    print(
        json.dumps(
            {
                "status": "design_and_isolation_checks_passed",
                "profiles": len(suite["profiles"]),
                "signals": checks["signals"],
                "profile_account_jobs": plan["profile_account_jobs"],
                "benchmark_jobs": plan["same_pool_benchmark_jobs"] + len(market_jobs),
                "full_batch_executed": False,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
