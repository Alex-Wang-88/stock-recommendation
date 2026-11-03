"""Immutable forward cohorts with next-open fills, lots, cash and delayed exits.

This deliberately does not turn overlapping daily cohorts into a compounded NAV.
Raw daily quote snapshots are the execution evidence; adjusted research bars are not.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, time, timedelta
from pathlib import Path

from .daily_data import atomic_json
from .daily_strategy import canonical_hash
from .market import SHANGHAI_TIMEZONE


def order_cost(notional: float, protocol: dict) -> float:
    return (
        max(notional * protocol["cost_bps_per_side"] / 10000, protocol["minimum_cost_per_order"])
        if notional > 0
        else 0.0
    )


def buy_fill(quote: dict | None, budget: float, protocol: dict) -> dict:
    result = {"quantity": 0, "price": None, "cost": 0.0, "spent": 0.0, "reason": "无报价"}
    if not quote:
        return result
    price, prior = quote.get("open", 0), quote.get("pre_close", 0)
    name = str(quote.get("name", "")).upper()
    if "ST" in name or "退" in name:
        result["reason"] = "买入日风险名称，取消买入"
        return result
    if price <= 0 or prior <= 0 or quote.get("volume", 0) <= 0:
        result["reason"] = "停牌或无有效开盘"
        return result
    if price / prior >= 1.095:
        result["reason"] = "开盘接近涨停，保留现金"
        return result
    lot = protocol["lot_size"]
    quantity = int(budget / (price * lot)) * lot
    while quantity > 0 and quantity * price + order_cost(quantity * price, protocol) > budget:
        quantity -= lot
    if quantity <= 0:
        result["reason"] = "预算不足一手，保留现金"
        return result
    cost = order_cost(quantity * price, protocol)
    return {
        "quantity": quantity,
        "price": price,
        "cost": cost,
        "spent": quantity * price + cost,
        "reason": "次日开盘模拟成交",
    }


def _deadline_after(day: str) -> datetime:
    following = datetime.fromisoformat(day).date() + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    # Conservative on exchange holidays: never accept a late historical quote capture.
    return datetime.combine(following, time(9, 30), SHANGHAI_TIMEZONE)


def _snapshot_timely(snapshot: dict, day: str) -> bool:
    try:
        captured = datetime.fromisoformat(snapshot["fetched_at"])
        return (
            snapshot["as_of"] == day
            and captured.tzinfo is not None
            and captured
            >= datetime.combine(datetime.fromisoformat(day), time(15, 30), SHANGHAI_TIMEZONE)
            and captured < _deadline_after(day)
        )
    except (KeyError, ValueError, TypeError):
        return False


def materialize_entry(signal: dict, entry_day: str, snapshot: dict, path: Path) -> dict:
    """Create once from a timely, frozen raw quote snapshot; verify before every read."""
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        checksum = result.pop("sha256", None)
        if checksum != canonical_hash(result) or result["signal_hash"] != signal["signal_hash"]:
            raise ValueError("Entry ledger integrity failure")
        result["sha256"] = checksum
        return result
    protocol = signal["protocol"]
    recorded = datetime.fromisoformat(signal["recorded_at"])
    if recorded >= datetime.combine(
        datetime.fromisoformat(entry_day), time(9, 30), SHANGHAI_TIMEZONE
    ):
        raise ValueError("Signal was not frozen before entry")
    if not _snapshot_timely(snapshot, entry_day):
        raise ValueError("Entry quote was captured late; cannot backfill a forward fill")
    picks = signal["selection"]["picks"]
    budget = protocol["initial_capital"] / max(len(picks), 1)
    fills = {
        p["symbol"]: buy_fill(snapshot["quotes"].get(p["symbol"]), budget, protocol) for p in picks
    }
    result = {
        "signal_hash": signal["signal_hash"],
        "entry_day": entry_day,
        "quote_hash": snapshot["sha256"],
        "fills": fills,
        "cash": protocol["initial_capital"] - sum(f["spent"] for f in fills.values()),
    }
    result["sha256"] = canonical_hash(result)
    # Caller holds the single daily-job lock. An existing ledger is never updated.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def _pool_return(symbols: list[str], entry: dict, exit_: dict, rate: float) -> float | None:
    if not symbols:
        return 0.0
    returns = []
    for symbol in symbols:
        a, b = entry.get(symbol), exit_.get(symbol)
        if not a or not b:
            return None
        if a["open"] <= 0 or a["pre_close"] <= 0 or a["volume"] <= 0:
            returns.append(0.0)
        elif a["open"] / a["pre_close"] >= 1.095 or "ST" in str(a["name"]).upper():
            returns.append(0.0)
        else:
            returns.append(b["close"] / a["open"] * (1 - rate) / (1 + rate) - 1)
    return sum(returns) / len(returns)


def evaluate_cohort(
    signal: dict, days: list[str], quotes: dict, index: dict, ledger_dir: Path
) -> dict:
    protocol, selection = signal["protocol"], signal["selection"]
    as_of = selection["as_of"]
    result = {
        "as_of": as_of,
        "recorded_at": signal["recorded_at"],
        "protocol_hash": signal["protocol_hash"],
        "signal_hash": signal["signal_hash"],
        "picks": selection["picks"],
        "status": "waiting_entry",
        "entry_day": None,
        "exit_day": None,
        "net_return": None,
        "benchmark_return": None,
        "excess": None,
        "pool_return": None,
        "pool_excess": None,
        "fees": 0.0,
        "filled": 0,
        "pending_exits": 0,
        "issues": [],
        "positions": [],
    }
    if as_of not in days or days.index(as_of) + 1 >= len(days):
        return result
    entry_position = days.index(as_of) + 1
    entry_day = days[entry_position]
    result["entry_day"] = entry_day
    entry_snapshot = quotes.get(entry_day)
    if not entry_snapshot:
        result.update(status="data_gap", issues=[f"缺少 {entry_day} 原始开盘留证"])
        return result
    try:
        entry = materialize_entry(
            signal, entry_day, entry_snapshot, ledger_dir / f"{signal['signal_hash']}.json"
        )
    except ValueError as error:
        result.update(status="data_gap", issues=[str(error)])
        return result
    cash, fees = entry["cash"], sum(f["cost"] for f in entry["fills"].values())
    target = entry_position + protocol["holding_sessions"] - 1
    has_target = target < len(days)
    result["target_exit_day"] = days[target] if has_target else None
    result["sessions_observed"] = min(len(days) - entry_position, protocol["holding_sessions"])
    last_exit = days[target] if has_target else days[-1]
    positions = []
    fatal = False
    for symbol, fill in entry["fills"].items():
        position = {
            "symbol": symbol,
            **fill,
            "exit_day": None,
            "exit_price": None,
            "exit_cost": 0.0,
            "status": "cash",
        }
        if fill["quantity"] == 0:
            if fill["reason"] == "无报价":
                fatal = True
                result["issues"].append(f"{symbol} 买入日缺少报价，成交状态未知")
            positions.append(position)
            continue
        result["filled"] += 1
        prior_close = None
        latest_price = fill["price"]
        position["status"] = "holding"
        for day_index in range(entry_position, len(days)):
            day = days[day_index]
            snapshot = quotes.get(day)
            quote = snapshot.get("quotes", {}).get(symbol) if snapshot else None
            if not snapshot or not _snapshot_timely(snapshot, day) or not quote:
                fatal = True
                result["issues"].append(f"{symbol} {day} 原始报价留证缺失或过晚")
                position["status"] = "data_gap"
                break
            if prior_close and quote["pre_close"] > 0:
                if abs(quote["pre_close"] / prior_close - 1) > 0.005:
                    fatal = True
                    position["status"] = "corporate_action_review"
                    result["issues"].append(f"{symbol} {day} 昨收不连续，公司行动待核对")
                    break
            latest_price = quote["close"] if quote["close"] > 0 else latest_price
            prior_close = latest_price
            if day_index < target:
                continue
            can_exit = (
                quote["volume"] > 0
                and quote["close"] > 0
                and quote["pre_close"] > 0
                and quote["close"] / quote["pre_close"] > 0.905
            )
            if can_exit:
                notional = fill["quantity"] * quote["close"]
                cost = order_cost(notional, protocol)
                cash += notional - cost
                fees += cost
                position.update(
                    exit_day=day, exit_price=quote["close"], exit_cost=cost, status="closed"
                )
                last_exit = max(last_exit, day)
                break
            position["status"] = "exit_blocked"
        position["marked_value"] = (
            fill["quantity"] * latest_price if position["status"] != "closed" else 0.0
        )
        positions.append(position)
    result["positions"] = positions
    result["fees"] = fees
    result["pending_exits"] = sum(p["status"] == "exit_blocked" for p in positions)
    result["marked_return"] = (cash + sum(p.get("marked_value", 0) for p in positions)) / protocol[
        "initial_capital"
    ] - 1
    if fatal:
        result["status"] = "data_gap"
    elif not has_target:
        result["status"] = "holding"
    elif result["pending_exits"]:
        result["status"] = "exit_blocked"
    else:
        result.update(
            status="matured", exit_day=last_exit, net_return=cash / protocol["initial_capital"] - 1
        )
        a, b = index.get(entry_day), index.get(last_exit)
        if a and b and a["open"] > 0:
            result["benchmark_return"] = b["close"] / a["open"] - 1
            result["excess"] = result["net_return"] - result["benchmark_return"]
        exit_snapshot = quotes.get(last_exit, {})
        result["pool_return"] = _pool_return(
            selection["pool_symbols"],
            entry_snapshot["quotes"],
            exit_snapshot.get("quotes", {}),
            protocol["cost_bps_per_side"] / 10000,
        )
        if result["pool_return"] is not None:
            result["pool_excess"] = result["net_return"] - result["pool_return"]
    return result


def evaluate_daily(directory: Path, days: list[str], protocol: dict) -> dict:
    protocol_hash = canonical_hash(protocol)
    quotes = {}
    for path in sorted((directory / "quotes").glob("*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        checksum = snapshot.pop("sha256", None)
        if checksum != canonical_hash(snapshot):
            raise ValueError(f"Quote snapshot integrity failure: {path.name}")
        snapshot["sha256"] = checksum
        quotes[snapshot["as_of"]] = snapshot
    benchmark_path = directory / "benchmark.json"
    benchmark = (
        json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else {}
    )
    index = {r["day"]: r for r in benchmark.get("rows", [])}
    cohorts = []
    for path in sorted((directory / "signals").glob("????-??-??__*.json")):
        signal = json.loads(path.read_text(encoding="utf-8"))
        checksum = signal.pop("signal_hash", None)
        if checksum != canonical_hash(signal):
            raise ValueError(f"Signal integrity failure: {path.name}")
        signal["signal_hash"] = checksum
        if signal["protocol_hash"] != protocol_hash:
            continue
        cohorts.append(evaluate_cohort(signal, days, quotes, index, directory / "entries"))
    matured = [c for c in cohorts if c["status"] == "matured"]
    paired = [c for c in matured if c["excess"] is not None]
    pool_paired = [c for c in matured if c["pool_excess"] is not None]
    gaps = sum(c["status"] == "data_gap" for c in cohorts)
    blocked = sum(c["status"] == "exit_blocked" for c in cohorts)

    def mean(rows: list, key: str) -> float | None:
        return sum(r[key] for r in rows) / len(rows) if rows else None

    excess = mean(paired, "excess")
    pool_excess = mean(pool_paired, "pool_excess")
    enough = len(paired) >= protocol["acceptance"]["minimum_mature_cohorts_for_observation"]
    complete = len(paired) == len(matured) == len(pool_paired) and gaps == blocked == 0
    status = "积累样本中"
    if gaps or blocked:
        status = "存在待核对样本，暂不评价达标"
    elif enough and complete:
        status = (
            "达到观察门槛，仍需持续验证" if excess > 0 and pool_excess > 0 else "未达到观察目标"
        )
    result = {
        "protocol_hash": protocol_hash,
        "cohorts": cohorts,
        "summary": {
            "signals": len(cohorts),
            "matured": len(matured),
            "paired": len(paired),
            "data_gaps": gaps,
            "blocked": blocked,
            "pending": len(cohorts) - len(matured) - gaps,
            "mean_net_return": mean(matured, "net_return"),
            "mean_excess": excess,
            "mean_pool_excess": pool_excess,
            "win_rate": sum(c["excess"] > 0 for c in paired) / len(paired) if paired else None,
            "worst_cohort": min((c["net_return"] for c in matured), default=None),
            "acceptance": status,
        },
        "benchmark_source": benchmark.get("source"),
        "limitations": [
            "每日批次有重叠，不拼接成年化或连续账户收益。",
            "费用为单边20bps且每笔至少5元的统一情景；日线模拟不含盘口冲击。",
            "股票与指数均为价格收益，未计现金分红；公司行动异常留待核对。",
            "同池对照使用分数单位，不含整手及最低费用，属于近似参考。",
            "达到60批成熟样本只是观察门槛，不代表统计显著或未来收益保证。",
        ],
    }
    if any(not math.isfinite(c["net_return"]) for c in matured):
        raise ValueError("Non-finite realized return")
    atomic_json(directory / "evaluation.json", result)
    return result
