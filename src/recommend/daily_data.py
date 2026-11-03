"""Dated raw quotes and official index history for the independent daily strategy."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from .bars import TencentSession, tencent_code
from .daily_strategy import MAIN_BOARD, canonical_hash

INDEX_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_quotes(text: str, as_of: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if "~" not in line:
            continue
        cells = line.split('"', 1)[-1].split("~")
        if len(cells) < 31:
            continue
        symbol, stamp = cells[2], cells[30]
        if not MAIN_BOARD.fullmatch(symbol) or len(stamp) < 14:
            continue
        try:
            observed = datetime.strptime(stamp[:14], "%Y%m%d%H%M%S")
            values = [float(cells[i]) for i in (3, 4, 5, 6)]
        except (ValueError, TypeError):
            continue
        if observed.date().isoformat() != as_of:
            continue
        if not all(math.isfinite(v) and v >= 0 for v in values):
            continue
        close, previous, open_, volume = values
        # Suspended/delisted quotes can keep a 09:00 source stamp. An explicit
        # zero-open, zero-volume response is a no-trade observation, not a fresh price.
        if observed.hour < 15 and not (open_ == 0 and volume == 0):
            continue
        rows.append(
            {
                "symbol": symbol,
                "name": cells[1],
                "status_date": as_of,
                "observed_at": stamp[:14],
                "open": open_,
                "close": close,
                "pre_close": previous,
                "volume": volume * 100,
                "session_traded": bool(volume > 0 and open_ > 0),
                "source": "https://qt.gtimg.cn/",
            }
        )
    return rows


def fetch_quotes(symbols: list[str], as_of: str, now: datetime) -> dict:
    client = TencentSession()
    records = []
    failures = []
    for offset in range(0, len(symbols), 60):
        batch = symbols[offset : offset + 60]
        try:
            response = client.get("https://qt.gtimg.cn/q=" + ",".join(map(tencent_code, batch)))
            response.raise_for_status()
            response.encoding = "gbk"
            records.extend(parse_quotes(response.text, as_of))
        except requests.RequestException as error:
            failures.append(f"batch {offset}: {type(error).__name__}")
        time.sleep(0.08)
    mapping = {row["symbol"]: row for row in records}
    payload = {
        "as_of": as_of,
        "fetched_at": now.isoformat(),
        "expected": len(symbols),
        "coverage": len(mapping) / max(len(symbols), 1),
        "errors": failures,
        "quotes": mapping,
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def save_quote_snapshot(directory: Path, payload: dict) -> Path:
    """Keep the first complete EOD snapshot, so historical fills do not move on reruns."""
    path = directory / f"{payload['as_of']}.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("coverage", 0) >= 0.95:
            return path
    atomic_json(path, payload)
    return path


def refresh_benchmark(path: Path, as_of: str, now: datetime) -> dict:
    """Price-index actual open and close; never manufacture a total-return opening price."""
    start = (datetime.fromisoformat(as_of) - timedelta(days=730)).strftime("%Y%m%d")
    params = {"indexCode": "000985", "startDate": start, "endDate": as_of.replace("-", "")}
    response = requests.get(
        INDEX_URL, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=35
    )
    response.raise_for_status()
    frame = pd.DataFrame(response.json().get("data", []))
    if frame.empty or not {"indexCode", "tradeDate", "open", "close"} <= set(frame):
        raise ValueError("Official benchmark response is empty or has changed schema")
    if not frame.indexCode.astype(str).eq("000985").all():
        raise ValueError("Benchmark code mismatch")
    dates = pd.to_datetime(frame.tradeDate.astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    if dates.duplicated().any():
        raise ValueError("Duplicate benchmark dates")
    rows = []
    for day, open_, close in zip(dates, frame.open, frame.close, strict=True):
        try:
            open_, close = float(open_), float(close)
        except (ValueError, TypeError):
            continue
        if day <= as_of and all(math.isfinite(v) and v > 0 for v in (open_, close)):
            rows.append({"day": day, "open": open_, "close": close})
    if not rows:
        raise ValueError("No valid official index OHLC")
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged = {r["day"]: r for r in previous.get("rows", [])}
    merged.update({r["day"]: r for r in rows})
    result = {
        "code": "000985",
        "name": "中证全指价格指数",
        "source": INDEX_URL,
        "fetched_at": now.isoformat(),
        "params": params,
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "rows": sorted(merged.values(), key=lambda r: r["day"]),
    }
    atomic_json(path, result)
    return result
