"""Prepare only historical execution status and published index levels for rule research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-rule-optimization-20261014"


def main():
    import pandas as pd
    from prepare_market_weight_batch import ROOT, sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["status-request", "merge-status", "indices"])
    parser.add_argument("--cohort", choices=["development", "holdout"], default="development")
    args = parser.parse_args()
    locked = json.loads((OUT / "windows_locked.json").read_text(encoding="utf-8"))
    if args.mode == "status-request":
        holdouts = [w for w in locked["windows"] if w["role"] != "development_previously_viewed"]
        days = sorted({d for w in holdouts for d in w["days"] + w["signal_dates"]})
        signal_days = {d for w in holdouts for d in w["signal_dates"]}
        factors = pd.read_parquet(
            OUT / "price_factors_pre_status.parquet", columns=["day", "symbol"]
        )
        symbols = sorted(
            set(factors.loc[factors.day.isin(signal_days), "symbol"].astype(str).str.zfill(6))
        )
        symbols = [s for s in symbols if s.startswith(("0", "3", "6"))]
        request = {
            "source": "BaoStock query_history_k_data_plus",
            "start": min(days),
            "end": max(days),
            "wanted_days": days,
            "symbols": symbols,
            "fields": "date,code,isST,tradestatus",
            "windows_sha256": sha256(OUT / "windows_locked.json"),
        }
        (OUT / "status_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
        print("Validation status request:", len(symbols), "symbols,", len(days), "days", flush=True)
        return
    if args.mode == "indices":
        import winreg

        import requests

        directory = OUT / "index_data"
        directory.mkdir(exist_ok=True)
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0"
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        ) as key:
            enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
            proxy = winreg.QueryValueEx(key, "ProxyServer")[0]
        if enabled and "=" not in proxy:
            session.proxies = {"http": "http://" + proxy, "https": "http://" + proxy}
        indices = json.loads(
            (ROOT / "config/research/index_benchmarks.json").read_text(encoding="utf-8")
        )["indices"]
        all_frames, receipts = [], []
        for index in indices:
            for kind, code in [
                ("price", index["price_code"]),
                ("total_return", index["total_return_code"]),
            ]:
                raw = directory / (code + ".json")
                params = {"indexCode": code, "startDate": "20230101", "endDate": "20260924"}
                url = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
                if not raw.exists():
                    response = session.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    assert response.json().get("data")
                    raw.write_bytes(response.content)
                payload = json.loads(raw.read_bytes())
                frame = pd.DataFrame(payload["data"])
                assert set(frame.indexCode) == {code}
                frame["day"] = pd.to_datetime(frame.tradeDate, format="%Y%m%d").dt.strftime(
                    "%Y-%m-%d"
                )
                selected = frame[["day", "open", "close"]].copy()
                selected["open"] = pd.to_numeric(selected.open, errors="coerce")
                selected["close"] = pd.to_numeric(selected.close, errors="raise")
                selected["code"], selected["kind"] = code, kind
                assert not selected.day.duplicated().any() and selected.close.gt(0).all()
                all_frames.append(selected)
                receipts.append(
                    {
                        "code": code,
                        "source": url,
                        "params": params,
                        "rows": len(selected),
                        "sha256": sha256(raw),
                    }
                )
                print(code, len(selected), "published days", flush=True)
        levels = pd.concat(all_frames, ignore_index=True)
        levels.to_parquet(directory / "levels.parquet", index=False)
        (directory / "receipt.json").write_text(
            json.dumps(
                {
                    "resource_context": RESOURCE_CONTEXT,
                    "records": receipts,
                    "levels_sha256": sha256(directory / "levels.parquet"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    sources = [
        ROOT / "logs/historical-weight-validation-20261014/historical_status_valuation.parquet",
        ROOT
        / "logs/market-only-extended-validation-20261014/status-data/historical_status.parquet",
    ]
    if args.cohort == "holdout":
        sources.append(OUT / "status-data/historical_status.parquet")
    history = pd.concat(
        [pd.read_parquet(p, columns=["date", "symbol", "isST", "tradestatus"]) for p in sources],
        ignore_index=True,
    )
    history["day"] = history.date.astype(str).str[:10]
    history["symbol"] = history.symbol.astype(str).str.zfill(6)
    duplicates = history[history.duplicated(["day", "symbol"], keep=False)]
    if not duplicates.empty:
        assert (
            duplicates.groupby(["day", "symbol"])[["isST", "tradestatus"]]
            .nunique()
            .le(1)
            .all()
            .all()
        )
    history = history.drop_duplicates(["day", "symbol"])
    windows = [
        w
        for w in locked["windows"]
        if (w["role"] == "development_previously_viewed") == (args.cohort == "development")
    ]
    days = {d for w in windows for d in w["days"] + w["signal_dates"]}
    history = history.loc[history.day.isin(days), ["day", "symbol", "isST", "tradestatus"]]
    history.to_parquet(OUT / ("execution_status_" + args.cohort + ".parquet"), index=False)
    factors = pd.read_parquet(OUT / "price_factors_pre_status.parquet")
    signals = {d for w in windows for d in w["signal_dates"]}
    factors = factors.loc[factors.day.isin(signals)].copy()
    factors["symbol"] = factors.symbol.astype(str).str.zfill(6)
    status = history.rename(
        columns={"isST": "historical_isST", "tradestatus": "historical_trade_status"}
    )
    factors = factors.merge(status, on=["day", "symbol"], how="left", validate="one_to_one")
    factor_destination = OUT / (args.cohort + "_factors.parquet")
    factors.to_parquet(factor_destination, index=False)
    coverage = factors.groupby("day").agg(
        candidates=("symbol", "size"), known_status=("historical_isST", "count")
    )
    coverage.to_csv(OUT / ("status_coverage_" + args.cohort + ".csv"))
    print(coverage.to_string(), flush=True)
    (OUT / ("data_receipt_" + args.cohort + ".json")).write_text(
        json.dumps(
            {
                "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources],
                "status_columns_read": ["date", "symbol", "isST", "tradestatus"],
                "financial_industry_inputs": [],
                "factor_sha256": sha256(factor_destination),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
