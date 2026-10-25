"""Download only after finalists are sealed; resume bounded network workers."""

from __future__ import annotations

import contextlib
import io
import json
import socket
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from design_index_improvement import OUT
from prepare_index_validation import dated_members
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
BS = None
REQUESTS = None


def fold_for(window, calendar):
    t = calendar.index(window["signal_dates"][0])
    valid_start, train_end = t - 69, t - 76
    train_start = max(65, train_end - 251)
    assert train_end - train_start >= 100
    return {
        "window": window["label"],
        "test_first_signal": window["signal_dates"][0],
        "train_start": calendar[train_start],
        "train_end": calendar[train_end],
        "valid_start": calendar[valid_start],
        "valid_end": calendar[t - 7],
        "test_signals": window["signal_dates"],
    }


def make_request():
    import baostock as bs
    import pandas as pd
    from prepare_market_weight_batch import sha256

    chosen_path = OUT / "finalists_locked.json"
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    plan = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    scope = json.loads((OUT / "validation_windows_locked.json").read_text(encoding="utf-8"))
    members = pd.read_parquet(OUT / "validation_membership.parquet")
    folds = []
    socket.setdefaulttimeout(25)
    with contextlib.redirect_stdout(io.StringIO()):
        login = bs.login()
    assert login.error_code == "0"
    try:
        for cohort, windows, calendar in [
            ("external", scope["external_windows"], scope["external_calendar"]),
            (
                "recent",
                [scope["recent_window"]],
                [d for d in scope["calendar"] if d >= "2023-01-01"],
            ),
        ]:
            for window in windows:
                fold = fold_for(window, calendar)
                folds.append({"cohort": cohort, **fold})
            needed = {}
            for candidate in chosen["finalists"]:
                if candidate["scorer"] == "P4":
                    continue
                dates = sorted(
                    {
                        d
                        for w in windows
                        for d in calendar[65::5]
                        if fold_for(w, calendar)["train_start"]
                        <= d
                        <= fold_for(w, calendar)["valid_end"]
                    }
                )
                needed[candidate["pool"]] = dates
            for pool, dates in needed.items():
                for i, day in enumerate(dates):
                    frame = dated_members(bs, pool, day)
                    frame["day"], frame["pool"] = day, pool
                    frame["symbol"] = frame.code.str.split(".").str[-1]
                    members = pd.concat(
                        [members, frame[["day", "pool", "symbol", "updateDate"]]], ignore_index=True
                    )
                    if i % 30 == 0:
                        print(
                            cohort, pool, "training membership", i + 1, "/", len(dates), flush=True
                        )
        members = members.drop_duplicates(["day", "pool", "symbol"])
        members.to_parquet(OUT / "validation_all_membership.parquet", index=False)
        basics_path = OUT / "validation_ipo_dates.csv"
        if not basics_path.exists():
            result = bs.query_stock_basic()
            assert result.error_code == "0"
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            basic = pd.DataFrame(rows, columns=result.fields)
            basic[["code", "ipoDate"]].to_csv(basics_path, index=False)
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            bs.logout()
    jobs = {}
    for cohort, start, end in [
        ("external", plan["external_data_start"], plan["external_data_end"]),
        ("recent", "2023-01-01", "2026-10-15"),
    ]:
        days = members.day.between(start, end)
        jobs[cohort] = {
            "start": start,
            "end": end,
            "symbols": sorted(members.loc[days, "symbol"].unique()),
            "fields": "date,code,open,high,low,close,preclose,volume,amount,tradestatus,isST",
            "adjustflag": "2",
        }
    request = {
        "finalists_sha256": sha256(chosen_path),
        "plan_sha256": sha256(OUT / "plan_locked.json"),
        "jobs": jobs,
        "folds": folds,
        "workers": 2,
        "historical_size_tolerance": (
            "up to two absent constituent records; never fill with current members"
        ),
    }
    path = OUT / "validation_download_request.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == request
    else:
        path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    return request


def initialize(request):
    global BS, REQUESTS
    configure_background_job()
    import baostock as bs

    socket.setdefaulttimeout(30)
    BS, REQUESTS = bs, request
    with contextlib.redirect_stdout(io.StringIO()):
        login = BS.login()
    assert login.error_code == "0"


def fetch_one(job):
    import pandas as pd

    cohort, symbol = job
    request = REQUESTS["jobs"][cohort]
    folder = OUT / "validation_quotes" / cohort
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / (symbol + ".parquet")
    if path.exists():
        try:
            cached = pd.read_parquet(path, columns=["day"])
            assert len(cached) and not cached.day.duplicated().any()
            assert cached.day.between(request["start"], request["end"]).all()
            return {"cohort": cohort, "symbol": symbol, "status": "cached", "rows": len(cached)}
        except Exception:
            # Preserve a partial file if a prior downloader was interrupted during a write.
            path.replace(path.with_suffix(".incomplete"))
    code = ("sh." if symbol.startswith("6") else "sz.") + symbol
    for attempt in range(2):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = BS.query_history_k_data_plus(
                    code,
                    request["fields"],
                    start_date=request["start"],
                    end_date=request["end"],
                    frequency="d",
                    adjustflag=request["adjustflag"],
                )
                if result.error_code != "0":
                    raise RuntimeError(result.error_code + " " + result.error_msg)
                rows = []
                while result.next():
                    rows.append(result.get_row_data())
            frame = pd.DataFrame(rows, columns=result.fields)
            if frame.empty:
                return {"cohort": cohort, "symbol": symbol, "status": "no_quotes"}
            for field in request["fields"].split(",")[2:]:
                frame[field] = pd.to_numeric(frame[field], errors="coerce")
            frame = frame.rename(columns={"date": "day", "preclose": "pre_close"})
            frame["symbol"] = symbol
            assert not frame.day.duplicated().any()
            assert frame.day.between(request["start"], request["end"]).all()
            pending = path.with_suffix(".pending")
            frame.to_parquet(pending, index=False)
            pending.replace(path)
            return {"cohort": cohort, "symbol": symbol, "status": "ok", "rows": len(frame)}
        except Exception as error:
            if attempt:
                return {
                    "cohort": cohort,
                    "symbol": symbol,
                    "status": "failed",
                    "error": str(error)[:150],
                }
            with contextlib.redirect_stdout(io.StringIO()):
                BS.login()
            time.sleep(0.2)
    raise AssertionError("unreachable")


def main():
    import argparse

    from prepare_market_weight_batch import sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["request", "fetch"])
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=2)
    args = parser.parse_args()
    if args.mode == "request":
        request = make_request()
        print({c: len(j["symbols"]) for c, j in request["jobs"].items()}, flush=True)
        return
    request_path = OUT / "validation_download_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert sha256(OUT / "finalists_locked.json") == request["finalists_sha256"]
    started, outcomes = time.monotonic(), []
    tasks = [(c, s) for c, r in request["jobs"].items() for s in r["symbols"]]
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize, initargs=(request,)
    ) as pool:
        futures = [pool.submit(fetch_one, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            outcomes.append(future.result())
            if i % 100 == 0 or i == len(tasks):
                failures = sum(x["status"] == "failed" for x in outcomes)
                print(
                    f"Validation quotes {i}/{len(tasks)}; failures {failures}; "
                    f"{time.monotonic() - started:.0f}s",
                    flush=True,
                )
    receipt = {
        "source": "BaoStock query_history_k_data_plus, qfq adjustflag=2",
        "request_sha256": sha256(request_path),
        "outcomes": outcomes,
        "elapsed_seconds": time.monotonic() - started,
        "resource": RESOURCE_CONTEXT,
        "workers": args.workers,
        "financial_industry_inputs": [],
    }
    (OUT / "validation_download_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    assert not any(x["status"] == "failed" for x in outcomes), (
        "Resume failed downloads before validation"
    )


if __name__ == "__main__":
    main()
