"""Reconstruct market-only factors and the frozen Qlib feature port on new data."""

from __future__ import annotations

import json

from design_index_improvement import OUT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def p4_features(raw, ipo_day, calendar):
    import numpy as np
    import pandas as pd

    raw = raw.sort_index()
    c, v, amount = raw.close, raw.volume, raw.amount
    ret = c / c.shift(1) - 1
    valid_ret = ret.rolling(20, min_periods=1).count().eq(20)
    day_position = pd.Series(range(len(calendar)), index=calendar)
    span = day_position.reindex(raw.index).to_numpy() - pd.Series(
        day_position.reindex(raw.index).to_numpy(), index=raw.index
    ).shift(249)
    observed = c.rolling(250, min_periods=1).count()
    result = pd.DataFrame(
        {
            "ret_20": c / c.shift(20) - 1,
            "ret_60": c / c.shift(60) - 1,
            "volume_trend_20": v.rolling(5, min_periods=1).mean()
            / v.rolling(20, min_periods=1).mean(),
            "turnover_value_20": amount.rolling(20, min_periods=1).mean(),
            "volatility_value": -ret.rolling(20, min_periods=1).std(ddof=0).where(valid_ret),
            "downside_volatility_value": -ret.where(ret < 0)
            .rolling(20, min_periods=1)
            .std(ddof=0)
            .fillna(0)
            .where(valid_ret),
            "drawdown_value": c.rolling(60, min_periods=60).apply(
                lambda x: np.min(x / np.maximum.accumulate(x) - 1),
                raw=True,
            ),
            "close_position_250": (c - c.rolling(250, min_periods=1).min())
            / (c.rolling(250, min_periods=1).max() - c.rolling(250, min_periods=1).min()),
            "dist_to_52w_high": c / c.rolling(250, min_periods=1).max() - 1,
            "history_days": np.where(observed.ge(250) & span.le(299), 250, np.nan),
            "historical_isST": raw.isST,
            "historical_trade_status": raw.tradestatus,
        },
        index=raw.index,
    )
    result["listed_days"] = (pd.to_datetime(result.index) - pd.Timestamp(ipo_day)).days
    return result.replace([np.inf, -np.inf], np.nan)


def main():
    import argparse

    import duckdb
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import sha256
    from qlib_alpha_features import alpha157

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["external", "recent", "both"], default="both")
    args = parser.parse_args()
    request = json.loads((OUT / "validation_download_request.json").read_text(encoding="utf-8"))
    scope = json.loads((OUT / "validation_windows_locked.json").read_text(encoding="utf-8"))
    finalists = json.loads((OUT / "finalists_locked.json").read_text(encoding="utf-8"))["finalists"]
    assert request["finalists_sha256"] == sha256(OUT / "finalists_locked.json")
    members = pd.read_parquet(OUT / "validation_all_membership.parquet")
    basic = pd.read_csv(OUT / "validation_ipo_dates.csv", dtype=str)
    basic["symbol"] = basic.code.str.split(".").str[-1]
    ipo = basic.drop_duplicates("symbol").set_index("symbol").ipoDate.to_dict()
    selected_ml_pools = {c["pool"] for c in finalists if c["scorer"] != "P4"}
    for cohort in (["external", "recent"] if args.cohort == "both" else [args.cohort]):
        region = request["jobs"][cohort]
        calendar = [d for d in scope["calendar"] if region["start"] <= d <= region["end"]]
        windows = scope["external_windows"] if cohort == "external" else [scope["recent_window"]]
        signals = sorted({s for w in windows for s in w["signal_dates"]})
        local = members.loc[members.day.between(region["start"], region["end"])].copy()
        eval_members = local.loc[local.day.isin(signals)]
        learn_members = local.loc[local.pool.isin(selected_ml_pools)]
        eval_days = eval_members.groupby("symbol").day.apply(lambda x: sorted(set(x))).to_dict()
        learn_days = learn_members.groupby("symbol").day.apply(lambda x: sorted(set(x))).to_dict()
        destination = OUT / "validation_features" / cohort
        destination.mkdir(parents=True, exist_ok=True)
        p4_dir, ml_dir = destination / "p4", destination / "alpha157"
        p4_dir.mkdir(exist_ok=True)
        ml_dir.mkdir(exist_ok=True)
        symbols = sorted(local.symbol.unique())
        missing, fallback, chunks = [], [], 0
        for start in range(0, len(symbols), 25):
            p4_path, ml_path = p4_dir / f"{start:04d}.parquet", ml_dir / f"{start:04d}.parquet"
            if p4_path.exists() and ml_path.exists():
                continue
            p4_parts, ml_parts = [], []
            for symbol in symbols[start : start + 25]:
                source = OUT / "validation_quotes" / cohort / (symbol + ".parquet")
                if not source.exists():
                    missing.append(symbol)
                    continue
                raw = pd.read_parquet(source).set_index("day").sort_index()
                day = ipo.get(symbol)
                if not isinstance(day, str) or len(day) != 10:
                    day = raw.index.min()
                    fallback.append(symbol)
                if symbol in eval_days:
                    p4 = p4_features(raw, day, calendar).reindex(eval_days[symbol])
                    p4["symbol"] = symbol
                    p4_parts.append(p4.reset_index())
                if symbol in learn_days:
                    aligned = raw.reindex(pd.Index(calendar, name="day"))
                    features = alpha157(aligned)
                    features["label_raw"] = aligned.open.shift(-6) / aligned.open.shift(-1) - 1
                    features["label_end"] = pd.Series(calendar, index=calendar).shift(-6)
                    features["fit_eligible"] = (
                        aligned.close.notna().rolling(65, min_periods=1).sum().ge(60)
                        & aligned.volume.gt(0)
                        & aligned.amount.rolling(20, min_periods=20).mean().ge(50_000_000)
                        & aligned.open.shift(-1).gt(0)
                        & aligned.open.shift(-6).gt(0)
                        & aligned.volume.shift(-1).gt(0)
                        & aligned.volume.shift(-6).gt(0)
                        & np.isfinite(features.label_raw)
                    )
                    features["symbol"] = symbol
                    selected = features.reindex(learn_days[symbol]).reset_index()
                    floats = selected.select_dtypes(include=["float64"]).columns
                    selected[floats] = selected[floats].astype("float32")
                    ml_parts.append(selected)
            if p4_parts:
                pd.concat(p4_parts, ignore_index=True).to_parquet(p4_path, index=False)
            if ml_parts:
                pd.concat(ml_parts, ignore_index=True).to_parquet(ml_path, index=False)
            chunks += 1
            if chunks % 4 == 0 or start + 25 >= len(symbols):
                print(
                    cohort, "features", min(start + 25, len(symbols)), "/", len(symbols), flush=True
                )
        con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
        for label, folder in [("p4", p4_dir), ("alpha157", ml_dir)]:
            con.from_parquet((folder / "*.parquet").as_posix()).write_parquet(
                (OUT / (cohort + "_" + label + ".parquet")).as_posix(), compression="zstd"
            )
        con.close()
        (destination / "receipt.json").write_text(
            json.dumps(
                {
                    "missing_symbols": missing,
                    "ipo_fallback_symbols": fallback,
                    "resource": RESOURCE_CONTEXT,
                    "finalists_sha256": sha256(OUT / "finalists_locked.json"),
                    "p4_sha256": sha256(OUT / (cohort + "_p4.parquet")),
                    "alpha157_sha256": sha256(OUT / (cohort + "_alpha157.parquet")),
                    "financial_industry_inputs": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
