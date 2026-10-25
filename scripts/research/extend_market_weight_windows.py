"""Lock calendar-selected new windows and rebuild price-only factors from immutable bars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "logs/historical-weight-validation-20261014/inputs"


def main():
    import duckdb
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["lock", "features", "merge-status"])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/market-only-extended-validation-20261014"
    )
    parser.add_argument("--volume-trend-min", type=float, default=1.1)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    market = SOURCE / "market.parquet"
    excluded_file = ROOT / "logs/local-optimization-test-20261014/windows_locked.json"
    if args.mode == "lock":
        if (out / "windows_locked.json").exists():
            raise FileExistsError("Windows are already locked")
        con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
        calendar = con.execute(
            "select cast(trade_date as date) as day,count(*) n from read_parquet(?) "
            "where trade_date>=date '2023-01-01' group by 1 having n>=4000 order by 1",
            [market.as_posix()],
        ).df()
        days = calendar.day.astype(str).tolist()
        previous = json.loads(excluded_file.read_text(encoding="utf-8"))["windows"]
        old_days = {day for window in previous for day in window["days"]}
        selected, occupied = [], set(old_days)

        def choose(label, anchor, name):
            for i, day in enumerate(days):
                if day < anchor or i == 0:
                    continue
                held = days[i : i + 30]
                if len(held) != 30 or set(held) & occupied:
                    continue
                window = {
                    "label": label,
                    "display_name": name,
                    "anchor": anchor,
                    "start": held[0],
                    "end": held[-1],
                    "days": held,
                    "signal_dates": [days[i + j - 1] for j in range(0, 30, 5)],
                    "entry_dates": [days[i + j] for j in range(0, 30, 5)],
                    "exit_dates": [days[i + j + 4] for j in range(0, 30, 5)],
                    "role": "new_time_validation",
                }
                selected.append(window)
                occupied.update(held)
                return
            raise RuntimeError(f"No disjoint 30-day block after {anchor}")

        # Reserve the event-calendar block before assigning quarterly windows.
        choose("calendar_2024_sep", "2024-09-02", "2024年9月前后")
        anchors = [
            ("calendar_2024_q1", "2024-02-01", "2024年一季度"),
            ("calendar_2024_q2", "2024-04-01", "2024年二季度"),
            ("calendar_2024_q3", "2024-07-01", "2024年三季度"),
            ("calendar_2024_q4", "2024-10-01", "2024年四季度后段"),
            ("calendar_2025_q1", "2025-01-01", "2025年一季度"),
            ("calendar_2025_q2", "2025-04-01", "2025年二季度"),
            ("calendar_2025_q3", "2025-07-01", "2025年三季度"),
        ]
        for label, anchor, name in anchors:
            choose(label, anchor, name)
        selected.sort(key=lambda window: window["start"])
        assert len(selected) == 8 and len(occupied) - len(old_days) == 240
        assert all(window["end"] < "2025-09-01" for window in selected)
        protocol = {
            "selection_rule": "Reserve first 30 trading days on/after 2024-09-02. "
            "For seven quarterly anchors, take the earliest disjoint 30 trading days. "
            "2024Q1 starts on/after Feb1 to supply >=250 broad-universe warm-up bars. "
            "No strategy returns or rankings used to select windows.",
            "selected_before_strategy_returns": True,
            "excluded_windows_file": str(excluded_file),
            "excluded_windows_sha256": sha256(excluded_file),
            "market_file": str(market),
            "market_sha256": sha256(market),
            "suite_sha256": sha256(ROOT / "config/research/market_only_weights.json"),
            "previously_selected_candidates": {
                "original_low_position": "P7_risk_only",
                "broader_liquid": "P4_equal_categories",
            },
            "validation_scope": "New historical time windows for previously frozen weights; "
            "not a forward chronological walk-forward or certified unbiased universe.",
            "windows": selected,
        }
        (out / "windows_locked.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        calendar.to_csv(out / "calendar_coverage.csv", index=False)
        wanted = sorted(
            {day for window in selected for day in window["days"] + window["signal_dates"]}
        )
        con.register("wanted", pd.DataFrame({"day": wanted}))
        symbols = (
            con.execute(
                "select distinct b.symbol from read_parquet(?) b join wanted w "
                "on cast(b.trade_date as date)=cast(w.day as date) "
                "where substr(b.symbol,1,1) in ('0','3','6') order by 1",
                [market.as_posix()],
            )
            .df()
            .symbol.astype(str)
            .str.zfill(6)
            .tolist()
        )
        con.close()
        (out / "status_request.json").write_text(
            json.dumps(
                {
                    "source": "BaoStock query_history_k_data_plus",
                    "start": min(wanted),
                    "end": max(wanted),
                    "wanted_days": wanted,
                    "symbols": symbols,
                    "fields": "date,code,isST,tradestatus",
                    "windows_sha256": sha256(out / "windows_locked.json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                [{key: window[key] for key in ("label", "start", "end")} for window in selected],
                indent=2,
            ),
            flush=True,
        )
        print("Status symbols:", len(symbols), flush=True)
        return

    locked = json.loads((out / "windows_locked.json").read_text(encoding="utf-8"))
    assert sha256(market) == locked["market_sha256"]
    signals = sorted({day for window in locked["windows"] for day in window["signal_dates"]})
    if args.mode == "merge-status":
        factors = pd.read_parquet(out / "price_factors_pre_status.parquet")
        status = pd.read_parquet(out / "status-data/historical_status.parquet")
        status["day"] = status.date.astype(str).str[:10]
        status["symbol"] = status.symbol.astype(str).str.zfill(6)
        status = status.loc[status.day.isin(signals), ["day", "symbol", "isST", "tradestatus"]]
        assert not status.duplicated(["day", "symbol"]).any()
        status = status.rename(
            columns={"isST": "historical_isST", "tradestatus": "historical_trade_status"}
        )
        factors = factors.merge(status, on=["day", "symbol"], how="left", validate="one_to_one")
        factors.to_parquet(out / "candidate_factors_final.parquet", index=False)
        counts = factors.groupby("day").agg(
            pre_status_candidates=("symbol", "size"),
            known_status=("historical_isST", "count"),
        )
        counts.to_csv(out / "status_merge_coverage.csv")
        print(counts.to_string(), flush=True)
        return

    con = duckdb.connect(
        config={"threads": 1, "memory_limit": "1GB", "temp_directory": str(out / "duckdb-temp")}
    )
    con.execute(
        "create temp table bars as select * from read_parquet(?) "
        "where trade_date>=date '2023-01-01'",
        [market.as_posix()],
    )
    con.execute(
        "create temp table dates as select trade_date,row_number() over(order by trade_date)-1 "
        "day_index from (select distinct trade_date from bars)"
    )
    con.register("signals", pd.DataFrame({"day": signals}))
    print(f"Computing {len(signals)} new price-only signal dates...", flush=True)
    features = con.execute("""
        with returns as (
          select *,close/nullif(lag(close) over ws,0)-1 daily_ret from bars
          window ws as (partition by symbol order by trade_date)
        ), f as (
          select symbol,trade_date,
            close/nullif(lag(close,20) over ws,0)-1 ret_20,
            close/nullif(lag(close,60) over ws,0)-1 ret_60,
            (close-min(close) over w250)/
              nullif(max(close) over w250-min(close) over w250,0) close_position_250,
            close/nullif(max(close) over w250,0)-1 dist_to_52w_high,
            avg(volume) over w5/nullif(avg(volume) over w20,0) volume_trend_20,
            avg(amount) over w20 turnover_value_20,
            case when count(daily_ret) over w20=20
              then stddev_pop(daily_ret) over w20 end realized_vol_20,
            case when count(daily_ret) over w20=20 then
              coalesce(stddev_pop(case when daily_ret<0 then daily_ret end) over w20,0)
              end downside_vol_20,
            count(close) over w250 observation_count250,
            lag(trade_date,249) over ws observation_250_date
          from returns
          window ws as (partition by symbol order by trade_date),
            w5 as (partition by symbol order by trade_date
              rows between 4 preceding and current row),
            w20 as (partition by symbol order by trade_date
              rows between 19 preceding and current row),
            w250 as (partition by symbol order by trade_date
              rows between 249 preceding and current row)
        )
        select f.*,current_day.day_index-old_day.day_index span250
          from f join signals s on cast(f.trade_date as date)=cast(s.day as date)
          left join dates current_day on current_day.trade_date=f.trade_date
          left join dates old_day on old_day.trade_date=f.observation_250_date
    """).df()
    features["symbol"] = features.symbol.astype(str).str.zfill(6)
    features["day"] = features.trade_date.astype(str).str[:10]
    features["history_days"] = np.where(
        (features.observation_count250 >= 250) & (features.span250 <= 299), 250, np.nan
    )
    features["volatility_value"] = -features.realized_vol_20
    features["downside_volatility_value"] = -features.downside_vol_20
    meta = con.execute(
        "select symbol,cast(listed_date at time zone 'Asia/Shanghai' as date) listed_date "
        "from read_parquet(?)",
        [(SOURCE / "instrument_snapshot.parquet").as_posix()],
    ).df()
    meta["symbol"] = meta.symbol.astype(str).str.zfill(6)
    features = features.merge(meta, on="symbol", how="left", validate="many_to_one")
    features["listed_days"] = (
        pd.to_datetime(features.day) - pd.to_datetime(features.listed_date)
    ).dt.days
    frames, coverage = [], []
    for day in signals:
        frame = features.loc[
            (features.day == day)
            & (features.history_days >= 250)
            & (features.listed_days >= 365)
            & (features.volume_trend_20 >= args.volume_trend_min)
            & (features.turnover_value_20 >= 50_000_000)
        ].copy()
        con.register("pool_symbols", frame[["symbol"]])
        raw = con.execute(
            "select symbol,trade_date,close from bars join pool_symbols using(symbol) "
            "where trade_date<=? qualify row_number() "
            "over(partition by symbol order by trade_date desc)<=60 "
            "order by symbol,trade_date",
            [day],
        ).df()
        risk = {}
        for symbol, group in raw.groupby("symbol", sort=False):
            values = group.close.to_numpy()
            risk[symbol] = (
                float(np.min(values / np.maximum.accumulate(values) - 1))
                if len(values) == 60
                else np.nan
            )
        frame["drawdown_value"] = frame.symbol.map(risk)
        frames.append(frame)
        coverage.append(
            {
                "day": day,
                "broad_pre_status": len(frame),
                "low_pre_status": int(
                    ((frame.close_position_250 <= 0.35) & (frame.dist_to_52w_high <= -0.20)).sum()
                ),
            }
        )
        con.unregister("pool_symbols")
        print(day, "pre-status candidates", len(frame), flush=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out / "price_factors_pre_status.parquet", index=False)
    pd.DataFrame(coverage).to_csv(out / "price_factor_coverage.csv", index=False)
    # Compare optimized calculations with production functions at a new historical date.
    sys.path.insert(0, str(ROOT / "src"))
    from recommend.factors.momentum import add_momentum_factors
    from recommend.factors.price import add_price_factors
    from recommend.factors.risk import add_risk_factors

    latest = signals[-1]
    sample = combined.loc[combined.day == latest].head(12).set_index("symbol")
    dates = con.execute(
        "select trade_date from dates where trade_date<=? order by 1 desc limit 300", [latest]
    ).fetchall()
    con.register("qa_symbols", sample.reset_index()[["symbol"]])
    bars = con.execute(
        "select b.* from bars b join qa_symbols using(symbol) "
        "where trade_date between ? and ? order by symbol,trade_date",
        [dates[-1][0], latest],
    ).df()
    checks = {}
    for calculated, columns in [
        (add_price_factors(bars), ["close_position_250", "dist_to_52w_high"]),
        (add_momentum_factors(bars), ["ret_20", "ret_60", "volume_trend_20", "turnover_value_20"]),
        (
            add_risk_factors(bars),
            ["volatility_value", "downside_volatility_value", "drawdown_value"],
        ),
    ]:
        reference = calculated.set_index("symbol")
        for column in columns:
            difference = float((sample[column] - reference[column]).abs().max())
            assert difference < (1e-4 if column == "turnover_value_20" else 1e-10)
            checks[column] = difference
    con.close()
    (out / "factor_qa.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "production_parity": checks,
                "financial_or_industry_inputs": [],
                "warm_up_start": "2023-01-01",
                "volume_trend_min": args.volume_trend_min,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Price factor parity PASS", flush=True)


if __name__ == "__main__":
    main()
