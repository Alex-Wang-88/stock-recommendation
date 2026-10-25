"""Fetch only calendars and dated constituent lists; do not inspect validation returns."""

from __future__ import annotations

import contextlib
import io
import json
import socket
import time

from design_index_improvement import OUT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def dated_members(bs, pool, day):
    import pandas as pd

    folder = OUT / "validation_membership"
    folder.mkdir(exist_ok=True)
    path = folder / f"{pool}_{day}.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)
    query = bs.query_hs300_stocks if pool == "csi300" else bs.query_zz500_stocks
    for attempt in range(3):
        try:
            result = query(date=day)
            if result.error_code != "0":
                raise RuntimeError(f"Membership query failed: {pool} {day} {result.error_code}")
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            break
        except (RuntimeError, OSError):
            if attempt == 2:
                raise
            with contextlib.redirect_stdout(io.StringIO()):
                bs.login()
            time.sleep(0.2)
    frame = pd.DataFrame(rows, columns=result.fields)
    expected = 300 if pool == "csi300" else 500
    assert expected - 2 <= len(frame) <= expected, (pool, day, len(frame))
    assert not frame.code.duplicated().any() and frame.updateDate.le(day).all()
    frame.to_csv(path, index=False)
    return frame


def build_window(calendar, start, count, label, role):
    first = next(i for i, d in enumerate(calendar) if d >= start)
    days = calendar[first : first + count]
    assert len(days) == count and first >= 1
    return {
        "label": label,
        "role": role,
        "start": days[0],
        "end": days[-1],
        "days": days,
        "signal_dates": [calendar[first + offset - 1] for offset in range(0, count, 5)],
    }


def main():
    import baostock as bs
    import pandas as pd
    from prepare_market_weight_batch import sha256

    socket.setdefaulttimeout(25)
    config = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        login = bs.login()
    assert login.error_code == "0"
    path = OUT / "validation_windows_locked.json"
    try:
        if path.exists():
            locked = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = bs.query_trade_dates(start_date="2017-07-01", end_date="2026-10-15")
            assert result.error_code == "0"
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            dates = pd.DataFrame(rows, columns=result.fields)
            calendar = sorted(dates.loc[dates.is_trading_day == "1", "calendar_date"])
            external_calendar = [d for d in calendar if d <= config["external_data_end"]]
            windows = [
                build_window(
                    external_calendar,
                    w["start_on_or_after"],
                    30,
                    w["label"],
                    "previously_unviewed_backward_time_stress",
                )
                for w in config["external_windows"]
            ]
            recent_days = [d for d in calendar if d >= "2026-09-25"]
            assert recent_days and max(recent_days) <= "2026-10-15"
            recent = build_window(
                calendar,
                "2026-09-25",
                len(recent_days),
                "recent_forward_partial",
                "previously_unviewed_forward_partial_under_30_days",
            )
            viewed = {d for w in config["windows"] for d in w["days"]}
            assert not viewed.intersection({d for w in windows + [recent] for d in w["days"]})
            locked = {
                "plan_sha256": sha256(OUT / "plan_locked.json"),
                "calendar": calendar,
                "external_calendar": external_calendar,
                "external_windows": windows,
                "recent_window": recent,
                "returns_inspected_before_lock": False,
            }
            path.write_text(json.dumps(locked, indent=2), encoding="utf-8")
        frames = []
        signals = sorted(
            {
                s
                for w in locked["external_windows"] + [locked["recent_window"]]
                for s in w["signal_dates"]
            }
        )
        for i, signal in enumerate(signals):
            for pool in config["pools"]:
                frame = dated_members(bs, pool, signal)
                frame["day"], frame["pool"] = signal, pool
                frame["symbol"] = frame.code.str.split(".").str[-1]
                frames.append(frame[["day", "pool", "symbol", "updateDate"]])
            if i % 10 == 0:
                print("Validation membership", i + 1, "/", len(signals), flush=True)
        pd.concat(frames, ignore_index=True).to_parquet(
            OUT / "validation_membership.parquet", index=False
        )
        print(
            "Locked 6 historical windows and",
            len(locked["recent_window"]["days"]),
            "recent sessions",
            flush=True,
        )
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            bs.logout()


if __name__ == "__main__":
    main()
