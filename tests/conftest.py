"""测试共用的构造器。

原则：测试**不依赖真实归档**。真实数据用来做端到端验收（手动跑），
单元测试用合成数据 —— 否则归档一变，测试就开始说谎。
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest


def make_bars(
    n_days: int = 300,
    symbols: tuple[str, ...] = ("000001", "000002"),
    start: str = "2025-01-01",
    close_for=None,
) -> pd.DataFrame:
    """合成一段日线（连续工作日，不模拟真实日历 —— 本模块不需要日历）。

    `close_for(symbol_index, day_index)` 定制收盘价；缺省是一条缓慢上行的序列。
    """

    dates = pd.bdate_range(start, periods=n_days)
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        for day_index, day in enumerate(dates):
            close = (
                close_for(symbol_index, day_index)
                if close_for is not None
                else 10.0 + day_index * 0.01
            )
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "pre_close": close,
                    "volume": 1000.0,
                    "amount": 10_000.0,
                }
            )
    return pd.DataFrame(rows)


def write_parquet(frame: pd.DataFrame, path) -> str:
    """把 DataFrame 落成 parquet（走 duckdb，避免依赖 pyarrow）。"""

    con = duckdb.connect()
    try:
        con.register("df", frame)
        con.execute(f"copy (select * from df) to '{str(path)}' (format parquet)")
    finally:
        con.close()
    return str(path)


@pytest.fixture
def bars_factory():
    return make_bars


@pytest.fixture
def parquet_writer():
    return write_parquet
