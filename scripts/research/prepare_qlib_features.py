"""Compute price-only features sequentially, retaining only dated member observations."""

from __future__ import annotations

import json
import time

from qlib_external_sources import OUT, ROOT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import duckdb
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import sha256
    from qlib_alpha_features import alpha157

    started = time.monotonic()
    config = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    receipt = json.loads((OUT / "plan_receipt.json").read_text(encoding="utf-8"))
    assert sha256(OUT / "plan_locked.json") == receipt["plan_sha256"]
    market = ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet"
    assert sha256(market) == config["market_sha256"]
    assert sha256(OUT / "membership.parquet") == receipt["membership_sha256"]
    members = pd.read_parquet(OUT / "membership.parquet")
    symbols = sorted(members.symbol.unique())
    calendar = pd.Index(config["calendar"], name="day")
    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    con.register("wanted_symbols", pd.DataFrame({"symbol": symbols}))
    # Read small symbol chunks so the full OHLCV panel never lives in pandas.
    parts = OUT / "feature_parts"
    parts.mkdir(exist_ok=True)
    missing, total = [], 0
    member_days = members.groupby("symbol").day.apply(lambda x: sorted(set(x))).to_dict()
    for start in range(0, len(symbols), 25):
        chunk = symbols[start : start + 25]
        destination = parts / f"{start:04d}.parquet"
        if destination.exists():
            continue
        con.register("chunk_symbols", pd.DataFrame({"symbol": chunk}))
        prices = con.execute(
            "select b.symbol,cast(b.trade_date as date)::varchar as day,"
            "b.open,b.high,b.low,b.close,b.volume,b.amount from read_parquet(?) b "
            "join chunk_symbols s using(symbol) where b.trade_date>=date '2023-01-01' "
            "order by b.symbol,b.trade_date",
            [market.as_posix()],
        ).df()
        outputs = []
        for symbol in chunk:
            raw = prices.loc[prices.symbol == symbol].set_index("day")
            if raw.empty:
                missing.append(symbol)
                continue
            assert not raw.index.duplicated().any()
            raw = raw.reindex(calendar)
            features = alpha157(raw)
            features["label_raw"] = raw.open.shift(-6) / raw.open.shift(-1) - 1
            label_day = pd.Series(calendar, index=calendar).shift(-6)
            features["label_end"] = label_day
            features["fit_eligible"] = (
                raw.close.notna().rolling(65, min_periods=1).sum().ge(60)
                & raw.volume.gt(0)
                & raw.amount.rolling(20, min_periods=20).mean().ge(50_000_000)
                & raw.volume.shift(-1).gt(0)
                & raw.volume.shift(-6).gt(0)
                & raw.open.shift(-1).gt(0)
                & raw.open.shift(-6).gt(0)
                & np.isfinite(features.label_raw)
            )
            features["symbol"] = symbol
            features["amount_volume_to_close"] = raw.amount / raw.volume / raw.close
            selected = features.reindex(member_days[symbol]).reset_index()
            float_cols = selected.select_dtypes(include=["float64"]).columns
            selected[float_cols] = selected[float_cols].astype("float32")
            outputs.append(selected)
        if outputs:
            pd.concat(outputs, ignore_index=True).to_parquet(destination, index=False)
        total += len(prices)
        print(f"Features {min(start + 25, len(symbols))}/{len(symbols)} symbols", flush=True)
    output = OUT / "alpha157.parquet"
    con.from_parquet((parts / "*.parquet").as_posix()).write_parquet(
        output.as_posix(), compression="zstd"
    )
    stats = con.execute(
        "select count(*),count(distinct symbol),min(day),max(day) from read_parquet(?)",
        [output.as_posix()],
    ).fetchone()
    con.close()
    columns = pd.read_parquet(next(parts.glob("*.parquet"))).columns
    features = [
        x
        for x in columns
        if x
        not in {"day", "symbol", "label_raw", "label_end", "fit_eligible", "amount_volume_to_close"}
    ]
    assert len(features) == 157
    (OUT / "feature_receipt.json").write_text(
        json.dumps(
            {
                "rows": stats[0],
                "symbols": stats[1],
                "first": stats[2],
                "last": stats[3],
                "feature_names": features,
                "feature_count": 157,
                "omitted": ["VWAP0"],
                "missing_symbols_this_execution": missing,
                "sha256": sha256(output),
                "financial_industry_inputs": [],
                "resource": RESOURCE_CONTEXT,
                "elapsed_seconds": time.monotonic() - started,
                "valid_label_filter": (
                    "positive entry/exit volume and prices, finite return; only used in fitting"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Feature cache complete", stats, flush=True)


if __name__ == "__main__":
    main()
