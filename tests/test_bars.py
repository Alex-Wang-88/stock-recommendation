"""日线增量更新测试。

这里全部用**假客户端**，不碰网络 —— 真实抓取由 `scripts/run_advance.py` 手动验收。
重点覆盖三个容易出错的机械点：

1. **日期取自行情源，而不是系统时间**（回退快照陷阱）；
2. **单位换算**（手→股、万元→元）—— 差 100 倍或 10000 倍都"看起来像数据"；
3. **公司行动的重叠比对** —— 漏掉它会在除权日造出一个虚假大跌。
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from recommend import bars as bars_module
from recommend.bars import (
    AdvanceResult,
    advance_bars,
    fetch_kline,
    resolve_quote_date,
    tencent_code,
)
from recommend.market import MarketStore

# -- 假客户端 --------------------------------------------------------------- #


def kline_node(rows: list) -> dict:
    """腾讯 K 线载荷的节点：键名以 `day` 结尾（`qfqday`），值是行列表。"""

    return {"qfqday": rows}


class StubClient:
    """按 URL 分发的假腾讯客户端。`klines` 形如 `{"sh600519": {"qfqday": [行, ...]}}`。"""

    def __init__(
        self,
        klines: dict[str, dict] | None = None,
        quote: str = "",
        fail: set | None = None,
    ):
        self.klines = klines or {}
        self.quote = quote
        self.fail = fail or set()
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict | None = None):
        self.calls.append((url, params))
        if url.startswith(bars_module.TENCENT_QUOTE_URL):
            return SimpleNamespace(text=self.quote, encoding=None)
        code = str((params or {}).get("param", "")).split(",")[0]
        if code in self.fail:
            raise RuntimeError("模拟网络故障")
        node = self.klines.get(code)
        payload = {"data": {code: node} if node else {}}
        return SimpleNamespace(json=lambda p=payload: p)


def kline_row(day: str, close: float, volume_lots: float = 1000.0, amount_wan: float = 5000.0):
    """腾讯 `newfqkline` 的一行：`[日期, 开, 收, 高, 低, 量(手), {}, 换手, 额(万元), ""]`。"""

    price = str(close)
    return [day, price, price, price, price, str(volume_lots), {}, "1.0", str(amount_wan), ""]


def quote_line(code: str, stamp: str) -> str:
    """腾讯报价串（`~` 分隔，下标 30 是时间戳）。"""

    parts = [""] * 31
    parts[1] = code
    parts[30] = stamp
    return f"v_{code}=" + "~".join(parts) + ";"


# -- 代码映射 --------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("600519", "sh600519"), ("000001", "sz000001"), ("300750", "sz300750"),
     ("688111", "sh688111"), ("430047", "sz430047"), ("1", "sz000001")],
)
def test_tencent_code_prefix(symbol, expected):
    assert tencent_code(symbol) == expected


# -- 交易日解析 ------------------------------------------------------------- #


def test_quote_date_comes_from_the_source_not_the_clock():
    """休市日调用报价接口拿到的是上一个交易日的数据 —— 必须按**源里的时间戳**盖章。

    若用系统时间，周五的行情会被写成周六，之后所有按交易日对齐的逻辑全部错位。
    """

    client = StubClient(quote=quote_line("sh600519", "20260918161436"))
    assert resolve_quote_date(client) == "2026-09-18"


def test_quote_date_takes_the_latest_stamp_among_references():
    text = "\n".join(
        [quote_line("sh600519", "20260918161436"), quote_line("sh000001", "20260918150003")]
    )
    assert resolve_quote_date(StubClient(quote=text)) == "2026-09-18"


def test_quote_date_raises_when_the_stamp_is_missing():
    from recommend.http import FetchError

    with pytest.raises(FetchError, match="时间戳"):
        resolve_quote_date(StubClient(quote="v_sh600519=没有任何波浪号;" ))


# -- 抓取与单位 ------------------------------------------------------------- #


def test_fetch_kline_converts_lots_to_shares_and_wan_to_yuan():
    """归档口径是**股**与**元**；腾讯给的是**手**与**万元**。

    这两处差 100 与 10000 倍，出错了数据**依然像正常数字**，
    只有对着归档逐股比对才看得出来。
    """

    rows = [kline_row("2026-09-18", 1257.12, 2489100, 313584.91)]
    client = StubClient(klines={"sh600519": kline_node(rows)})
    frame = fetch_kline(client, "600519", 5)
    row = frame.iloc[0]
    assert row["volume"] == 248_910_000  # 手 → 股 是 ×100
    assert row["amount"] == pytest.approx(3_135_849_100.0)  # 万元 → 元 是 ×10000


def test_fetch_kline_derives_pre_close_from_the_series_itself():
    """首根 pre_close 必为 NaN —— 正因如此，抓取必须带**重叠窗口**。"""

    rows = [kline_row("2026-09-17", 1266.98), kline_row("2026-09-18", 1257.12)]
    frame = fetch_kline(StubClient(klines={"sh600519": kline_node(rows)}), "600519", 5)
    assert pd.isna(frame.iloc[0]["pre_close"])
    assert frame.iloc[1]["pre_close"] == pytest.approx(1266.98)


def test_fetch_kline_returns_empty_frame_for_unknown_symbol():
    frame = fetch_kline(StubClient(klines={}), "600519", 5)
    assert frame.empty


# -- 增量推进 --------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path, bars_factory, parquet_writer):
    """到 2026-09-16 为止的两只股票（09-10 起 5 个工作日）。"""

    frame = bars_factory(
        n_days=5,
        symbols=("600519", "000001"),
        start="2026-09-10",
        close_for=lambda _s, _d: 100.0,
    )
    market = MarketStore(parquet_writer(frame, tmp_path / "daily.parquet"))
    assert market.latest_trade_date() == "2026-09-16"
    yield market
    market.close()


def test_advance_is_a_noop_when_already_current(store, monkeypatch):
    """归档已到行情源的最新交易日 ⇒ 直接返回，**不做任何抓取**。"""

    stub = StubClient(quote=quote_line("sh600519", "20260910150000"))
    monkeypatch.setattr(bars_module, "TencentSession", lambda: stub)
    result = advance_bars(store)
    assert result.up_to_date is True
    assert result.rows == 0
    assert "已是最新" in result.summary()


def test_advance_refreshes_same_day_after_close(store, monkeypatch):
    """盘中快照已经写入后，收盘后同一天必须重新抓取最终日线。"""

    # 历史重叠日保持一致；当天从盘中 100 刷新成收盘 101，不能被误判为公司行动。
    rows = [kline_row("2026-09-15", 100.0), kline_row("2026-09-16", 101.0)]
    stub = StubClient(
        klines={
            "sh600519": kline_node(rows),
            "sz000001": kline_node(rows),
        },
        quote=quote_line("sh600519", "20260916150000"),
    )
    monkeypatch.setattr(bars_module, "TencentSession", lambda: stub)
    monkeypatch.setattr(bars_module, "is_current_eod_ready", lambda _date: True)

    result = advance_bars(store, workers=1)

    assert result.up_to_date is False
    assert result.previous_date == result.quote_date == "2026-09-16"
    assert result.repaired == []
    assert result.rows > 0
    assert "刷新当天日线" in result.summary()


def test_advance_appends_new_dates(store, monkeypatch):
    klines = {
        "sh600519": kline_node([kline_row("2026-09-16", 100.0), kline_row("2026-09-17", 101.0)]),
        "sz000001": kline_node([kline_row("2026-09-16", 100.0), kline_row("2026-09-17", 102.0)]),
    }
    stub = StubClient(klines=klines, quote=quote_line("sh600519", "20260917150000"))
    monkeypatch.setattr(bars_module, "TencentSession", lambda: stub)

    result = advance_bars(store)
    assert result.previous_date == "2026-09-16"
    assert result.quote_date == "2026-09-17"
    assert result.symbols_ok == 2
    assert result.repaired == []
    assert store.latest_trade_date() == "2026-09-17"


def test_advance_detects_corporate_action_and_repulls_the_whole_window(store, monkeypatch):
    """前复权序列整体平移 ⇒ 重叠日的收盘价对不上 ⇒ 必须整窗重拉。

    漏掉这一步的后果是**在除权日造出一个虚假大跌** —— 而「低位」筛选恰好专挑这种票，
    等于把"刚除权"当成"跌到低位"买进去。
    """

    # 第一遍：600519 在重叠日 2026-09-16 的收盘价被平移成 50（库内是 100）⇒ 触发修复。
    drifted_first = [kline_row("2026-09-16", 50.0), kline_row("2026-09-17", 51.0)]
    # 第二遍（整窗重拉）：整条前复权序列都平移了，这才是"权威版本"。
    repaired = [
        kline_row("2026-09-15", 49.0),
        kline_row("2026-09-16", 50.0),
        kline_row("2026-09-17", 51.0),
    ]
    # 000001 没有公司行动：重叠日与库内逐分一致（100），不该被判为漂移。
    steady = [kline_row("2026-09-16", 100.0), kline_row("2026-09-17", 102.0)]

    calls: list[tuple[str, int]] = []

    class DriftStub(StubClient):
        def get(self, url, params=None):
            if url.startswith(bars_module.TENCENT_QUOTE_URL):
                return SimpleNamespace(text=self.quote, encoding=None)
            code = str((params or {}).get("param", "")).split(",")[0]
            count = int(str(params["param"]).split(",")[4])
            calls.append((code, count))
            if code == "sz000001":
                rows = steady
            else:
                # 修复档（整窗）拿到平移后的全序列；增量档拿到会触发漂移判断的片段。
                rows = repaired if count > bars_module.INCREMENTAL_BARS else drifted_first
            return SimpleNamespace(json=lambda r=rows, c=code: {"data": {c: {"qfqday": r}}})

    monkeypatch.setattr(
        bars_module,
        "TencentSession",
        lambda: DriftStub(quote=quote_line("sh600519", "20260917150000")),
    )

    result = advance_bars(store)
    assert result.repaired == ["600519"]  # 只有它漂移
    assert any(count == bars_module.REPAIR_BARS for _, count in calls)
    # 没漂移的标的**不该**被整窗重拉 —— 否则每次都要多抓 700 根。
    assert ("sz000001", bars_module.REPAIR_BARS) not in calls
    # 修复后的值覆盖了旧值：该标的 2026-09-16 的收盘价从 100 变成 50。
    close = store.conn.execute(
        f"select close from '{store.path.as_posix()}' "
        "where symbol = '600519' and trade_date = date '2026-09-16'"
    ).fetchone()[0]
    assert close == pytest.approx(50.0)


def test_advance_records_failures_without_losing_the_rest(store, monkeypatch):
    """单标的失败不能拖垮整次推进 —— 失败的留给下次自动重试。"""

    # 重试会 sleep，测试里去掉等待（只替换 sleep，不动 time.time）。
    monkeypatch.setattr(bars_module.time, "sleep", lambda _seconds: None)
    klines = {"sz000001": kline_node([kline_row("2026-09-17", 102.0)])}
    stub = StubClient(
        klines=klines, quote=quote_line("sh600519", "20260917150000"), fail={"sh600519"}
    )
    monkeypatch.setattr(bars_module, "TencentSession", lambda: stub)

    result = advance_bars(store, workers=1)
    assert result.failed == ["600519"]
    assert "000001" not in result.failed
    assert store.latest_trade_date() == "2026-09-17"


def test_advance_reports_when_the_date_cannot_be_resolved(store, monkeypatch):
    monkeypatch.setattr(bars_module, "TencentSession", lambda: StubClient(quote="坏的报价串;"))
    result = advance_bars(store)
    assert result.failed == ["<quote-date>"]
    assert result.rows == 0
    assert any("无法确定交易日" in note for note in result.notes)


def test_advance_result_summary_mentions_repairs_and_failures():
    result = AdvanceResult(
        quote_date="2026-09-18",
        previous_date="2026-09-16",
        rows=10,
        symbols_ok=2,
        repaired=["600519"],
        failed=["000002"],
    )
    text = result.summary()
    assert "2026-09-16 → 2026-09-18" in text
    assert "公司行动" in text
    assert "抓取失败" in text


def test_upsert_bars_is_idempotent(store):
    """重跑同一天是幂等的：不会留下半新半旧的行。"""

    frame = store.bars()
    again = frame.copy()
    again["close"] = 123.0
    store.upsert_bars(again)
    assert store.row_count() == len(frame)
    closes = store.closes_on_date("2026-09-16")
    assert set(closes["close"]) == {123.0}


def test_upsert_bars_rejects_missing_columns(store):
    with pytest.raises(ValueError, match="缺少列"):
        store.upsert_bars(pd.DataFrame({"symbol": ["000001"]}))
