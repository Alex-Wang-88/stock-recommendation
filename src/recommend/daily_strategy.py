"""Frozen, deterministic daily picks. No language model, financial API or fitted weights."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .market import is_forward_eod_ready

MAIN_BOARD = re.compile(r"(?:60|00)\d{4}")
FACTOR_LABELS = {"trend": "中期趋势", "stability": "低波动", "pullback": "短期回撤"}


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    weights = protocol["weights"]
    if set(weights) != set(FACTOR_LABELS):
        raise ValueError("Expected exactly trend, stability, pullback weights")
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    if not math.isclose(sum(weights.values()), 1):
        raise ValueError("Weights must sum to one")
    if not 1 <= protocol["selection_count"] <= 10:
        raise ValueError("Selection count must be between 1 and 10")
    if protocol["holding_sessions"] < 2:
        raise ValueError("Holding period must be at least two sessions")
    return protocol


def build_features(bars: pd.DataFrame, as_of: str, minimum_history: int = 250) -> pd.DataFrame:
    """Use only prices up to as_of; one common, complete factor set for every stock."""
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame = frame.loc[frame.trade_date.le(as_of)].sort_values(["symbol", "trade_date"])
    if frame.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("Duplicate symbol/date in market data")
    rows = []
    for symbol, group in frame.groupby("symbol", sort=True):
        if not MAIN_BOARD.fullmatch(symbol) or group.iloc[-1].trade_date != as_of:
            continue
        group = group.tail(minimum_history)
        numeric = group[["open", "high", "low", "close", "volume", "amount"]].apply(
            pd.to_numeric, errors="coerce"
        )
        # Suspended/stale rows cannot silently become a fully observed price series.
        if len(group) < minimum_history or not np.isfinite(numeric.to_numpy()).all():
            continue
        if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
            continue
        close = numeric.close
        ret = close.pct_change(fill_method=None)
        last = numeric.iloc[-1]
        rows.append(
            {
                "symbol": symbol,
                "close": float(close.iloc[-1]),
                "trend": float(close.iloc[-21] / close.iloc[-121] - 1),
                "stability": -float(ret.tail(60).std(ddof=0)),
                "pullback": -float(close.iloc[-1] / close.iloc[-6] - 1),
                "ma120": float(close.tail(120).mean()),
                "amount_20": float(numeric.amount.tail(20).mean()),
                "active": bool(last.volume > 0 and last.amount > 0),
                "valid_ohlc": bool(
                    last.low
                    <= min(last.open, last.close)
                    <= max(last.open, last.close)
                    <= last.high
                ),
                "history_days": len(group),
            }
        )
    return pd.DataFrame(rows)


def select_daily(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    protocol: dict,
    as_of: str,
) -> dict:
    """Filter first, rank once in the same pool, then enforce industry diversity."""
    result = {
        "as_of": as_of,
        "picks": [],
        "pool_symbols": [],
        "pool_count": 0,
        "market_breadth": None,
        "reason": "完整行情不足",
        "funnel": {},
    }
    if features.empty:
        return result
    meta = metadata.copy()
    meta["symbol"] = meta["symbol"].astype(str).str.zfill(6)
    if meta.symbol.duplicated().any():
        raise ValueError("Duplicate symbol in dated metadata")
    frame = features.merge(meta, on="symbol", how="left", validate="one_to_one")
    for key in ("name", "industry", "listed_date", "status_date"):
        if key not in frame:
            frame[key] = None
    if "quote_active" not in frame:
        frame["quote_active"] = True
    eligible_name = frame.name.notna() & ~frame.name.fillna("").str.contains(
        r"ST|退", case=False, regex=True
    )
    dated = frame.status_date.eq(as_of)
    listed = pd.to_datetime(frame.listed_date, errors="coerce")
    old_enough = (pd.Timestamp(as_of) - listed).dt.days.ge(365)
    base = (
        eligible_name
        & dated
        & old_enough
        & frame.active
        & frame.valid_ohlc
        & frame.quote_active.fillna(False).astype(bool)
    )
    breadth_frame = frame.loc[base]
    if breadth_frame.empty:
        result["reason"] = "缺少同日名称/ST状态或完整行情"
        return result
    breadth = float((breadth_frame.close > breadth_frame.ma120).mean())
    result["market_breadth"] = breadth
    tests = {
        "同日状态及有效行情": base,
        "价格与流动性": frame.close.ge(protocol["minimum_price"])
        & frame.amount_20.ge(protocol["minimum_amount_20"]),
        "中期趋势": frame.trend.gt(0) & frame.close.gt(frame.ma120),
        "完整评分因子": np.isfinite(frame[list(FACTOR_LABELS)]).all(axis=1),
    }
    mask = pd.Series(True, index=frame.index)
    for name, condition in tests.items():
        mask &= condition
        result["funnel"][name] = int(mask.sum())
    pool = frame.loc[mask].copy()
    result["pool_symbols"] = sorted(pool.symbol.tolist())
    result["pool_count"] = len(pool)
    if breadth < protocol["minimum_market_breadth"]:
        result["reason"] = "市场站上120日均线的比例不足40%，本日不新增候选"
        return result
    if pool.empty:
        result["reason"] = "没有满足趋势、流动性和数据完整性条件的股票"
        return result
    pool["score"] = 0.0
    for factor, weight in protocol["weights"].items():
        pool[f"rank_{factor}"] = pool[factor].rank(pct=True, method="average")
        pool["score"] += pool[f"rank_{factor}"] * weight
    pool = pool.sort_values(["score", "symbol"], ascending=[False, True], kind="stable")
    counts: Counter = Counter()
    for row in pool.to_dict("records"):
        industry = row.get("industry")
        industry = str(industry) if pd.notna(industry) and str(industry).strip() else "行业未识别"
        if counts[industry] >= protocol["maximum_per_industry"]:
            continue
        counts[industry] += 1
        result["picks"].append(
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "industry": industry,
                "close": row["close"],
                "score": row["score"],
                "trend_return": row["trend"],
                "volatility_60": -row["stability"],
                "return_5": -row["pullback"],
                "amount_20": row["amount_20"],
                "parts": [
                    {
                        "key": factor,
                        "label": FACTOR_LABELS[factor],
                        "weight": weight,
                        "percentile": row[f"rank_{factor}"],
                        "contribution": row[f"rank_{factor}"] * weight,
                    }
                    for factor, weight in protocol["weights"].items()
                ],
                "reason": (
                    f"中期趋势 {row['trend']:+.1%}；近60日波动 {(-row['stability']):.2%}；"
                    f"近5日涨跌 {(-row['pullback']):+.1%}；行业分散后入选"
                ),
            }
        )
        if len(result["picks"]) == protocol["selection_count"]:
            break
    result["reason"] = f"符合条件 {len(pool)} 只，按评分与行业分散选出 {len(result['picks'])} 只"
    return result


def freeze_signal(directory: Path, selection: dict, protocol: dict, now: datetime) -> Path:
    """Exclusive create: never rewrite a forward signal after observing future prices."""
    if not is_forward_eod_ready(selection["as_of"], now):
        raise ValueError("Signal is outside the forward recording window")
    directory.mkdir(parents=True, exist_ok=True)
    protocol_hash = canonical_hash(protocol)
    for previous in directory.glob("protocol-*.json"):
        saved = json.loads(previous.read_text(encoding="utf-8"))
        if saved["version"] == protocol["version"] and canonical_hash(saved) != protocol_hash:
            raise ValueError("Protocol changed without a new version")
    locked = directory / f"protocol-{protocol_hash}.json"
    if not locked.exists():
        with locked.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, ensure_ascii=False, indent=2, allow_nan=False)
    path = directory / f"{selection['as_of']}__{protocol_hash[:16]}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["protocol_hash"] != protocol_hash:
            raise ValueError("Frozen protocol mismatch")
        return path
    signal = {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "recorded_at": now.isoformat(),
        "selection": selection,
    }
    signal["signal_hash"] = canonical_hash(signal)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(signal, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return path
