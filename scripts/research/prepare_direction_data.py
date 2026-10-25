"""Reuse frozen bars/status; add lag-only short returns and skip-recent-week momentum."""

from __future__ import annotations

import json
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/market-direction-study-20261015"
PREVIOUS = ROOT / "logs/market-rule-optimization-20261014"


def lag_features(connection, table, signals):
    import pandas as pd

    connection.register("direction_signal_days", pd.DataFrame({"day": signals}))
    return connection.execute(f"""
        with lags as (
          select symbol,cast(trade_date as date) as day,close,
            lag(close,5) over ws c5,lag(close,10) over ws c10,
            lag(close,20) over ws c20,lag(close,60) over ws c60
          from {table}
          window ws as (partition by symbol order by trade_date)
        )
        select l.*,
          close/nullif(c5,0)-1 ret_5,close/nullif(c10,0)-1 ret_10,
          c5/nullif(c60,0)-1 ret_60_skip5,
          close/nullif(c20,0)-1 ret_20_check,close/nullif(c60,0)-1 ret_60_check
        from lags l join direction_signal_days s on l.day=cast(s.day as date)
    """).df()


def main():
    import duckdb
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import FACTORS, READ_COLUMNS, sha256

    lock = json.loads((OUT / "windows_locked.json").read_text(encoding="utf-8"))
    market = ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet"
    assert sha256(market) == lock["market_sha256"]
    frames, statuses, sources = [], [], []
    for cohort in ["development", "holdout"]:
        factor = PREVIOUS / (cohort + "_factors.parquet")
        status = PREVIOUS / ("execution_status_" + cohort + ".parquet")
        frames.append(pd.read_parquet(factor, columns=list(READ_COLUMNS)))
        statuses.append(pd.read_parquet(status, columns=["day", "symbol", "isST", "tradestatus"]))
        sources.extend([factor, status])
    features = pd.concat(frames, ignore_index=True)
    assert not features.duplicated(["day", "symbol"]).any()
    signals = sorted({s for w in lock["windows"] for s in w["signal_dates"]})
    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    con.execute(
        "create temp table price_source as select symbol,trade_date,close "
        "from read_parquet(?) where trade_date>=date '2023-01-01'",
        [market.as_posix()],
    )
    new = lag_features(con, "price_source", signals)
    new["day"] = new.day.astype(str)
    new["symbol"] = new.symbol.astype(str).str.zfill(6)
    merged = features.merge(new, on=["day", "symbol"], how="left", validate="one_to_one")
    assert len(merged) == len(features)
    parity = {}
    for column in ["ret_20", "ret_60"]:
        difference = float((merged[column] - merged[column + "_check"]).abs().max())
        assert difference < 1e-10
        parity[column] = difference
    common_old = np.isfinite(merged[list(FACTORS)].to_numpy(dtype=float)).all(axis=1)
    common_new = np.isfinite(merged[["ret_5", "ret_10", "ret_60_skip5"]].to_numpy(dtype=float)).all(
        axis=1
    )
    assert not (common_old & ~common_new).any(), (
        "New fields changed the old common-complete universe"
    )
    # Independent arithmetic on published lag prices, not on future returns.
    assert np.allclose(merged.ret_5, merged.close / merged.c5 - 1, equal_nan=True)
    assert np.allclose(
        merged.ret_60_skip5, (1 + merged.ret_60) / (1 + merged.ret_5) - 1, equal_nan=True
    )
    merged[[*READ_COLUMNS, "ret_5", "ret_10", "ret_60_skip5"]].to_parquet(
        OUT / "research_factors.parquet", index=False
    )
    status = pd.concat(statuses, ignore_index=True)
    duplicate = status[status.duplicated(["day", "symbol"], keep=False)]
    if len(duplicate):
        assert (
            duplicate.groupby(["day", "symbol"])[["isST", "tradestatus"]]
            .nunique()
            .le(1)
            .all()
            .all()
        )
    status = status.drop_duplicates(["day", "symbol"])
    status.to_parquet(OUT / "execution_status_research.parquet", index=False)
    con.close()
    receipt = {
        "technical_status": "PASS",
        "resource_context": RESOURCE_CONTEXT,
        "rows": len(merged),
        "signals": len(signals),
        "existing_factor_parity": parity,
        "new_fields_do_not_change_common_universe": True,
        "financial_industry_inputs": [],
        "fresh_validation_included": False,
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in [market, *sources]],
        "feature_sha256": sha256(OUT / "research_factors.parquet"),
        "status_sha256": sha256(OUT / "execution_status_research.parquet"),
    }
    (OUT / "data_qa.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ["technical_status", "rows", "signals", "existing_factor_parity"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
