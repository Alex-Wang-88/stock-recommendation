"""行情层测试 —— 重点是「基准收盘」与「日期校验」。"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from recommend.market import MarketStore, default_base_dates, is_current_eod_ready


@pytest.fixture
def store(tmp_path, bars_factory, parquet_writer):
    frame = bars_factory(n_days=30, symbols=("000001", "000002"))
    path = parquet_writer(frame, tmp_path / "daily.parquet")
    market = MarketStore(path)
    yield market
    market.close()


def test_trade_dates_are_exactly_what_is_in_the_archive(store):
    dates = store.trade_dates()
    assert len(dates) == 30
    assert dates == sorted(dates)
    assert store.latest_trade_date() == dates[-1]


def test_trade_dates_limit_takes_the_tail(store):
    assert store.trade_dates(limit=3) == store.trade_dates()[-3:]


def test_latest_complete_trade_date_defers_intraday_date(tmp_path, bars_factory, parquet_writer):
    """盘中已有当天行情时，推荐必须继续使用上一根完整日线；15:30 后才切当天。"""

    path = parquet_writer(
        bars_factory(n_days=2, symbols=("000001",), start="2026-09-18"),
        tmp_path / "intraday.parquet",
    )
    market = MarketStore(path)
    try:
        before_close = datetime(2026, 9, 21, 15, 29)
        after_close = datetime(2026, 9, 21, 15, 30)
        assert market.latest_trade_date() == "2026-09-21"
        assert market.latest_complete_trade_date(before_close) == "2026-09-18"
        assert market.latest_complete_trade_date(after_close) == "2026-09-21"
        assert not is_current_eod_ready("2026-09-21", before_close)
        assert is_current_eod_ready("2026-09-21", after_close)
        assert not is_current_eod_ready("2026-09-18", after_close)
    finally:
        market.close()


def test_closes_on_takes_last_trading_day_not_exact_match(store):
    """基准日不是交易日（或当天停牌）时，应退到**不晚于**该日的最后一个交易日。

    这是「924 以来涨幅」在长假/停牌下仍然算得出来的前提。
    """

    dates = store.trade_dates()
    requested = dates[10]
    # 往后挪一天，使请求日不在日期集合里（除非恰好又是交易日）。
    day_after = (pd.Timestamp(requested) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    closes = store.closes_on({"base": day_after})
    expected = store.conn.execute(
        f"select close from '{store.path.as_posix()}' "
        f"where symbol = '000001' and trade_date <= date '{day_after}' "
        "order by trade_date desc limit 1"
    ).fetchone()[0]
    actual = closes.loc[closes["symbol"] == "000001", "base"].iloc[0]
    assert actual == pytest.approx(expected)


def test_symbol_listed_after_the_base_is_absent_not_zero(store, tmp_path, parquet_writer):
    """上市晚于基准日 ⇒ 该 symbol **不出现在结果里**（调用方 left join 后为 NaN）。

    关键是它**不能**以 0 或任何默认值出现 —— 那会把「没有基准价」读成「基准价是 0」，
    于是「924 以来涨幅」变成 inf 或被当成 0%。

    ⚠️ 注意这里断言的是「缺席」而不是「NaN」：`closes_on` 的结果是
    「有基准价的行」的并集，根本没有这一行。上游必须先 left join 到自己的 universe
    才会看到 NaN。
    """

    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": ["000001"] * 5,
                    "trade_date": pd.bdate_range("2025-03-01", periods=5),
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "pre_close": 1.0,
                    "volume": 1.0,
                    "amount": 1.0,
                }
            ),
            pd.DataFrame(
                {
                    "symbol": ["000003"] * 5,
                    "trade_date": pd.bdate_range("2025-06-01", periods=5),
                    "open": 2.0,
                    "high": 2.0,
                    "low": 2.0,
                    "close": 2.0,
                    "pre_close": 2.0,
                    "volume": 1.0,
                    "amount": 1.0,
                }
            ),
        ]
    )
    market = MarketStore(parquet_writer(frame, tmp_path / "later.parquet"))
    try:
        closes = market.closes_on({"base": "2025-04-01"})
        # 调用方的正确用法：left join 到自己的 universe，缺席者变 NaN。
        universe = pd.DataFrame({"symbol": ["000001", "000003"]})
        joined = universe.merge(closes, on="symbol", how="left")
    finally:
        market.close()
    by_symbol = joined.set_index("symbol")["base"]
    assert by_symbol["000001"] == pytest.approx(1.0)
    assert pd.isna(by_symbol["000003"])
    assert "000003" not in set(closes["symbol"])


def test_closes_on_supports_multiple_labels(store):
    dates = store.trade_dates()
    closes = store.closes_on({"early": dates[0], "late": dates[-1]})
    assert set(closes.columns) == {"symbol", "early", "late"}
    assert len(closes) == 2


def test_date_is_validated_before_being_put_into_sql(store):
    """日期来自自然语言翻译层，必须在拼进 SQL 前校验 —— 否则就是注入口。"""

    with pytest.raises(ValueError):
        store.closes_on({"base": "2024-09-24' or '1'='1"})
    with pytest.raises(ValueError):
        store.trade_dates(end="not-a-date")
    with pytest.raises(ValueError):
        # 前 10 个字符合法、后面带脏 —— 截断式校验会放它过去。
        store.closes_on({"base": "2024-09-24xyz"})


def test_bars_are_normalized(store):
    frame = store.bars()
    assert pd.api.types.is_string_dtype(frame["symbol"])
    assert set(frame["symbol"]) == {"000001", "000002"}
    assert pd.api.types.is_datetime64_any_dtype(frame["trade_date"])


def test_default_base_dates_uses_previous_year_end():
    assert default_base_dates("2026-09-16") == {
        "base_924": "2024-09-24",
        "base_ytd": "2025-12-31",
    }


def test_missing_archive_raises_a_helpful_error(tmp_path):
    store = MarketStore(tmp_path / "nope.parquet")
    try:
        with pytest.raises(FileNotFoundError, match="sync-bars"):
            store.trade_dates()
    finally:
        store.close()


def test_industry_meta_is_optional_and_normalized(store, tmp_path):
    path = tmp_path / "industry.csv"
    pd.DataFrame(
        {
            "股票代码": ["000001", "000002.SZ", "000001.SH"],
            "行业": ["银行", "地产", "银行"],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")
    out = store.industry_meta(path).sort_values("symbol").reset_index(drop=True)
    assert out.to_dict("records") == [
        {"symbol": "000001", "industry": "银行"},
        {"symbol": "000002", "industry": "地产"},
    ]
