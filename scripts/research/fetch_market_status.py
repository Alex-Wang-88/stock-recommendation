"""Resume historical ST/trading-status download for predeclared new time windows."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import socket
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
BS = None
REQUEST = None
FOLDER = None


def below_normal():
    configure_background_job()


def initialize(request, folder):
    global BS, REQUEST, FOLDER
    below_normal()
    socket.setdefaulttimeout(20)
    import baostock as bs

    BS, REQUEST, FOLDER = bs, request, Path(folder)
    with contextlib.redirect_stdout(io.StringIO()):
        login = BS.login()
    if login.error_code != "0":
        raise RuntimeError("BaoStock login error " + login.error_code)


def fetch(symbol):
    import pandas as pd

    destination = FOLDER / (symbol + ".parquet")
    if destination.exists():
        return {"symbol": symbol, "status": "cached"}
    code = ("sh." if symbol.startswith("6") else "sz.") + symbol
    for attempt in range(2):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = BS.query_history_k_data_plus(
                    code,
                    REQUEST["fields"],
                    start_date=REQUEST["start"],
                    end_date=REQUEST["end"],
                    frequency="d",
                    adjustflag="3",
                )
                if result.error_code != "0":
                    raise RuntimeError("query error " + result.error_code)
                rows = []
                while result.next():
                    rows.append(result.get_row_data())
            frame = pd.DataFrame(rows, columns=result.fields)
            if frame.empty:
                return {"symbol": symbol, "status": "no_quotes"}
            frame = frame.loc[frame.date.isin(REQUEST["wanted_days"])].copy()
            frame["symbol"] = symbol
            for column in ("isST", "tradestatus"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame.to_parquet(destination, index=False)
            return {"symbol": symbol, "status": "ok", "rows": len(frame)}
        except Exception as error:
            if attempt:
                return {"symbol": symbol, "status": "failed", "error": type(error).__name__}
            with contextlib.redirect_stdout(io.StringIO()):
                BS.login()
            time.sleep(0.25)
    return {"symbol": symbol, "status": "failed"}


def main():
    from prepare_market_weight_batch import ROOT, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        type=Path,
        default=ROOT / "logs/market-only-extended-validation-20261014/status_request.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs/market-only-extended-validation-20261014/status-data",
    )
    parser.add_argument("--workers", type=int, default=4, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    below_normal()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output / "per-symbol"
    folder.mkdir(exist_ok=True)
    scope = args.output / "request_locked.json"
    if scope.exists():
        assert json.loads(scope.read_text(encoding="utf-8")) == request, "Cache scope differs"
    else:
        scope.write_text(json.dumps(request, indent=2), encoding="utf-8")
    outcomes = []
    start = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize, initargs=(request, str(folder))
    ) as pool:
        jobs = [pool.submit(fetch, symbol) for symbol in request["symbols"]]
        for i, future in enumerate(as_completed(jobs), 1):
            outcomes.append(future.result())
            if i % 100 == 0 or i == len(jobs):
                unavailable = sum(row["status"] in ("failed", "no_quotes") for row in outcomes)
                print(
                    f"Historical status {i}/{len(jobs)}; unavailable {unavailable}; "
                    f"elapsed {time.monotonic() - start:.0f}s",
                    flush=True,
                )
    import duckdb

    files = [str(path) for path in folder.glob("*.parquet")]
    if not files:
        raise RuntimeError("No historical status data received")
    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    combined = con.execute("select * from read_parquet(?)", [files]).df()
    assert not combined.duplicated(["date", "symbol"]).any()
    destination = args.output / "historical_status.parquet"
    combined.to_parquet(destination, index=False)
    receipt = {
        "source": "BaoStock",
        "fields": request["fields"],
        "source_docs": "https://pypi.org/project/baostock/",
        "request_sha256": sha256(scope),
        "records": len(combined),
        "symbols": int(combined.symbol.nunique()),
        "workers": args.workers,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "outcomes": outcomes,
        "status_sha256": sha256(destination),
    }
    (args.output / "download_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Historical status complete: {len(combined)} records", flush=True)


if __name__ == "__main__":
    main()
