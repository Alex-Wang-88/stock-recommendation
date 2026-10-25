"""Cache published CSI price and total-return levels for the frozen research windows."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import winreg

    import pandas as pd
    import requests

    root = Path(__file__).resolve().parents[2]
    out = root / "logs/index-benchmark-rerun-20261014/index_data"
    out.mkdir(parents=True, exist_ok=True)
    config_path = root / "config/research/index_benchmarks.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plans = [
        root / "logs/market-only-weight-design-20261014/batch_plan.json",
        root / "logs/market-only-extended-validation-20261014/plan/batch_plan.json",
    ]
    windows = [w for p in plans for w in json.loads(p.read_text(encoding="utf-8"))["windows"]]
    wanted_days = {day for window in windows for day in window["days"]}
    assert len(windows) == 14 and len(wanted_days) == 420
    start, end = min(wanted_days), max(wanted_days)
    # Save the protocol before requesting full histories.
    snapshot = out / "benchmark_protocol_locked.json"
    if snapshot.exists():
        assert snapshot.read_bytes() == config_path.read_bytes()
    else:
        snapshot.write_bytes(config_path.read_bytes())
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
    ) as key:
        enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
        proxy = winreg.QueryValueEx(key, "ProxyServer")[0]
    if enabled and "=" not in proxy:
        session.proxies = {"http": "http://" + proxy, "https": "http://" + proxy}
    receipts, frames = [], []
    for index in config["indices"]:
        for kind, code in [
            ("price", index["price_code"]),
            ("total_return", index["total_return_code"]),
        ]:
            raw = out / f"{code}_raw.json"
            params = {
                "indexCode": code,
                "startDate": start.replace("-", ""),
                "endDate": end.replace("-", ""),
            }
            if not raw.exists():
                response = session.get(config["source"], params=params, timeout=45)
                response.raise_for_status()
                payload = response.json()
                assert isinstance(payload.get("data"), list) and payload["data"], (code, payload)
                raw.write_bytes(response.content)
            payload = json.loads(raw.read_bytes())
            frame = pd.DataFrame(payload["data"])
            assert set(frame.indexCode) == {code}
            assert frame.indexNameEnAll.str.contains("Total Return").all() == (
                kind == "total_return"
            )
            frame["day"] = pd.to_datetime(frame.tradeDate, format="%Y%m%d").dt.strftime("%Y-%m-%d")
            assert not frame.day.duplicated().any()
            assert wanted_days <= set(frame.day), (code, sorted(wanted_days - set(frame.day)))
            for column in ["open", "high", "low", "close"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            assert frame.close.notna().all() and (frame.close > 0).all()
            if kind == "price":
                assert frame.open.notna().all() and (frame.open > 0).all()
            selected = frame[["day", "open", "high", "low", "close"]].copy()
            selected["code"], selected["price_code"], selected["kind"], selected["name"] = (
                code,
                index["price_code"],
                kind,
                index["name"],
            )
            frames.append(selected)
            receipts.append(
                {
                    "code": code,
                    "kind": kind,
                    "source": config["source"],
                    "params": params,
                    "raw_file": str(raw),
                    "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "rows": len(frame),
                    "required_days_covered": len(wanted_days),
                    "observed_non_null_open_rows": int(frame.open.notna().sum()),
                }
            )
            print(f"{code}: {len(frame)} published days; all 420 required days covered", flush=True)
    levels = pd.concat(frames, ignore_index=True).sort_values(["code", "day"])
    levels.to_csv(out / "index_levels.csv", index=False, encoding="utf-8-sig")
    levels.to_parquet(out / "index_levels.parquet", index=False)
    receipt = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "source": config["source"],
        "resource_context": RESOURCE_CONTEXT,
        "protocol_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "level_sha256": hashlib.sha256((out / "index_levels.parquet").read_bytes()).hexdigest(),
        "window_count": len(windows),
        "days_per_window": 30,
        "records": receipts,
        "checks": {
            "coverage_pass": True,
            "forward_fill_used": False,
            "synthetic_total_return_open_used": False,
        },
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Published index history coverage PASS", flush=True)


if __name__ == "__main__":
    main()
