"""数据源适配层的规范化测试（全部离线，用假客户端喂固定返回）。"""

from __future__ import annotations

import pandas as pd
import pytest

from recommend.archive.sources import (
    DragonTigerSource,
    DragonTigerWithFallbackSource,
    FallbackResponseError,
    MarginSource,
    OfficialMarginSource,
    SectorSnapshotSource,
    SinaDragonTigerSource,
    ThemeSource,
    _num,
    candidate_dates,
)
from recommend.http import ThrottledClient


class FakeResponse:
    def __init__(self, payload: object, encoding: str = "utf-8") -> None:
        self._payload = payload
        self.encoding = encoding

    def json(self) -> object:
        if isinstance(self._payload, str):
            raise ValueError("不是 JSON")
        return self._payload


class HtmlResponse:
    def __init__(self, html: str) -> None:
        self.text = html
        self.content = html.encode("utf-8")


class FakeClient:
    """按 URL 片段路由的假客户端，并记录每次调用的参数。"""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict | None = None, **_kwargs: object) -> FakeResponse:
        self.calls.append((url, params))
        for fragment, payload in self.routes.items():
            if fragment in url:
                return FakeResponse(payload)
        raise AssertionError(f"未预置的 URL：{url}")

    def get_json(self, url: str, params: dict | None = None, **kwargs: object) -> object:
        return self.get(url, params=params, **kwargs).json()


class OfficialFakeClient(FakeClient):
    def get(self, url: str, params: dict | None = None, **_kwargs: object):
        self.calls.append((url, params))
        for fragment, payload in self.routes.items():
            if fragment in url:
                if isinstance(payload, HtmlResponse):
                    return payload
                return FakeResponse(payload)
        raise AssertionError(f"未预置的 URL：{url}")


# --------------------------------------------------------------------------- #
# 数值清洗
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", [None, "", "-"])
def test_placeholders_become_none_not_zero(raw) -> None:
    """`-` 和空串是**缺失**，不是 0。当成 0 会让"没有成交额"变成"成交额为零"。"""

    assert _num(raw) is None


def test_numeric_strings_are_parsed() -> None:
    assert _num("3.5") == 3.5
    assert _num(0) == 0.0


def test_unparseable_becomes_none() -> None:
    assert _num("--") is None


# --------------------------------------------------------------------------- #
# 候选日期
# --------------------------------------------------------------------------- #


def test_candidate_dates_skips_weekends() -> None:
    # 2026-09-18 是周五，19/20 是周末。
    assert candidate_dates("2026-09-17", "2026-09-21") == [
        "2026-09-17",
        "2026-09-18",
        "2026-09-21",
    ]


def test_candidate_dates_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        candidate_dates("2026-09-18", "2026-09-01")


# --------------------------------------------------------------------------- #
# 题材归因（同花顺）
# --------------------------------------------------------------------------- #


def _theme_payload() -> dict:
    return {
        "errocode": 0,
        "data": [
            {
                "code": "1",
                "name": "平安银行",
                "reason": "银行+金融科技",
                "date": "2026-09-18",
                "close": "12.30",
                "zhangfu": "1.50",
                "huanshou": "0.80",
                "chengjiaoe": "2000000000",
                "chengjiaoliang": "160000000",
                "ddejingliang": "-1200",
                "market": "33",
            }
        ],
    }


def test_theme_symbols_are_zero_padded_to_six_digits() -> None:
    """上游返回的是 `1` 这种裸码，必须补齐成 `000001` 才能和行情对齐。"""

    source = ThemeSource(FakeClient({"getharden": _theme_payload()}))  # type: ignore[arg-type]
    frame = source.fetch("2026-09-18")
    assert frame.loc[0, "symbol"] == "000001"
    assert frame.loc[0, "reason"] == "银行+金融科技"
    assert frame.loc[0, "change_pct"] == 1.5


def test_theme_date_is_placed_in_the_url_path() -> None:
    """日期是**路径参数** —— 这正是历史可以回补的原因。"""

    payload = _theme_payload()
    payload["data"][0]["date"] = "2025-06-16"  # 回显日期必须与请求一致，否则被拒
    client = FakeClient({"getharden": payload})
    ThemeSource(client).fetch("2025-06-16")  # type: ignore[arg-type]
    assert "date/2025-06-16/" in client.calls[0][0]


def test_theme_empty_payload_returns_empty_frame_with_schema() -> None:
    source = ThemeSource(FakeClient({"getharden": {"errocode": 0, "data": []}}))  # type: ignore[arg-type]
    frame = source.fetch("2026-09-19")
    assert frame.empty
    assert "reason" in frame.columns


def test_theme_error_code_raises_instead_of_pretending_to_be_empty() -> None:
    """接口报错必须抛异常。静默当成"今天没有强势股"会让归档悄悄留洞。"""

    payload = {"errocode": 5, "errormsg": "参数错误", "data": []}
    source = ThemeSource(FakeClient({"getharden": payload}))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="参数错误"):
        source.fetch("2026-09-18")


def test_source_date_is_recorded_from_the_echo() -> None:
    source = ThemeSource(FakeClient({"getharden": _theme_payload()}))  # type: ignore[arg-type]
    assert source.fetch("2026-09-18").loc[0, "source_date"] == "2026-09-18"


# --------------------------------------------------------------------------- #
# 回退快照：这是实测污染过 38 个休市日的坑
# --------------------------------------------------------------------------- #


def test_fallback_snapshot_is_rejected_by_the_date_echo() -> None:
    """请求没有数据的日期时，接口**静默返回最新快照**，并在行里回显那个日期。

    实测：请求 2024-10-01（休市）时回显的是接口当天的最新数据日。
    不回显就发现不了 —— 而不知不觉写进去，就是 38 天的污染。
    """

    payload = {
        "errocode": 0,
        "data": [
            {
                "code": "1",
                "name": "平安银行",
                "reason": "银行+金融科技",
                "date": "2026-09-18",
                "close": "12.30",
                "zhangfu": "1.50",
            }
        ],
    }
    source = ThemeSource(FakeClient({"getharden": payload}))  # type: ignore[arg-type]
    with pytest.raises(FallbackResponseError, match="回退返回了 2026-09-18"):
        source.fetch("2024-10-01")


def test_fallback_snapshot_is_also_rejected_when_prices_are_absent() -> None:
    """第二道闸：真实交易日 100% 带行情，回退响应 0% 带行情（实测 42218 vs 3081 行）。

    即便日期回显恰好对得上，没有行情也不该当成交易日写进归档。
    """

    payload = {
        "errocode": 0,
        "data": [
            {
                "code": "603375",
                "name": "盛景微",
                "reason": "模拟芯片+电子雷管",
                "date": "2026-09-18",
            }
        ],
    }
    source = ThemeSource(FakeClient({"getharden": payload}))  # type: ignore[arg-type]
    with pytest.raises(FallbackResponseError, match="不含任何行情字段"):
        source.fetch("2026-09-18")


def test_a_genuine_trading_day_passes_both_checks() -> None:
    source = ThemeSource(FakeClient({"getharden": _theme_payload()}))  # type: ignore[arg-type]
    frame = source.fetch("2026-09-18")
    assert len(frame) == 1
    assert frame.loc[0, "change_pct"] == 1.5
    assert frame.loc[0, "source_date"] == "2026-09-18"


# --------------------------------------------------------------------------- #
# 龙虎榜
# --------------------------------------------------------------------------- #


def test_dragon_tiger_normalises_amounts_in_yuan() -> None:
    payload = {
        "result": {
            "data": [
                {
                    "TRADE_DATE": "2026-09-18 00:00:00",
                    "SECURITY_CODE": "600519",
                    "SECURITY_NAME_ABBR": "贵州茅台",
                    "EXPLANATION": "日涨幅偏离值达7%",
                    "CLOSE_PRICE": 1500.0,
                    "CHANGE_RATE": 7.1,
                    "BILLBOARD_NET_AMT": 123456789.0,
                    "BILLBOARD_BUY_AMT": 200000000.0,
                    "BILLBOARD_SELL_AMT": 76543211.0,
                    "TURNOVERRATE": 2.4,
                }
            ]
        }
    }
    source = DragonTigerSource(FakeClient({"datacenter": payload}))  # type: ignore[arg-type]
    frame = source.fetch("2026-09-18")
    assert frame.loc[0, "trade_date"] == "2026-09-18"
    # 存**元**，不做万元换算 —— 换算是展示层的事。
    assert frame.loc[0, "net_buy"] == 123456789.0
    assert frame.loc[0, "reason" if "reason" in frame.columns else "explanation"] == (
        "日涨幅偏离值达7%"
    )


def test_dragon_tiger_missing_result_block_is_empty_not_error() -> None:
    source = DragonTigerSource(FakeClient({"datacenter": {"result": None}}))  # type: ignore[arg-type]
    assert source.fetch("2026-09-19").empty


def test_datacenter_paginates_when_result_has_multiple_pages() -> None:
    payload = {
        "result": {
            "pages": 2,
            "data": [{"SECURITY_CODE": "000001"}],
        }
    }
    second = {"result": {"pages": 2, "data": [{"SECURITY_CODE": "000002"}]}}

    class PagedClient(FakeClient):
        def get_json(self, url: str, params: dict | None = None, **kwargs: object):
            self.calls.append((url, params))
            return payload if len(self.calls) == 1 else second

    client = PagedClient({"datacenter": payload})
    frame = DragonTigerSource(client).fetch("2026-09-18")  # type: ignore[arg-type]
    assert frame["symbol"].tolist() == ["000001", "000002"]
    assert client.calls[1][1]["pageNumber"] == "2"  # type: ignore[index]


def test_sina_fallback_keeps_money_fields_missing() -> None:
    pytest.importorskip("lxml")
    html = """
    <table>
      <tr><th>股票代码</th><th>股票名称</th><th>收盘价</th><th>指标</th></tr>
      <tr><td>1</td><td>平安银行</td><td>12.30</td><td>日涨幅偏离值达7%</td></tr>
    </table>
    """
    source = SinaDragonTigerSource(
        OfficialFakeClient({"vip.stock": HtmlResponse(html)})  # type: ignore[arg-type]
    )
    frame = source.fetch("2026-09-18")
    assert frame.loc[0, "symbol"] == "000001"
    assert pd.isna(frame.loc[0, "net_buy"])
    assert frame.loc[0, "explanation"] == "日涨幅偏离值达7%"


def test_official_sse_margin_maps_exchange_rows() -> None:
    sse_row = [
        "unused",
        "2026-09-18",
        "9000",
        "10000",
        "30000000",
        "unused",
        "unused",
        "90000000",
        "120000000",
        "unused",
        "5000000000",
        "贵州茅台",
        "600519",
    ]
    client = OfficialFakeClient({"query.sse": {"result": [sse_row]}, "szse.cn": {}})
    frame = OfficialMarginSource(client).fetch_sse("2026-09-18")  # type: ignore[arg-type]
    assert frame.loc[0, "symbol"] == "600519"
    assert frame.loc[0, "rzye"] == 5.0e9
    assert frame.loc[0, "rqmcl"] == 10000.0


def test_fallback_wrapper_uses_sina_only_after_primary_failure() -> None:
    pytest.importorskip("lxml")
    html = """
    <table>
      <tr><th>股票代码</th><th>股票名称</th><th>收盘价</th><th>指标</th></tr>
      <tr><td>1</td><td>平安银行</td><td>12.30</td><td>换手率达20%</td></tr>
    </table>
    """

    class FailingPrimaryClient(OfficialFakeClient):
        def get(self, url: str, params: dict | None = None, **kwargs: object):
            self.calls.append((url, params))
            if "datacenter" in url:
                raise RuntimeError("主源不可用")
            return HtmlResponse(html)

    frame = DragonTigerWithFallbackSource(FailingPrimaryClient({})).fetch("2026-09-18")
    assert frame.loc[0, "symbol"] == "000001"
    assert pd.isna(frame.loc[0, "net_buy"])


# --------------------------------------------------------------------------- #
# 两融
# --------------------------------------------------------------------------- #


def test_margin_normalises_daily_rows() -> None:
    payload = {
        "result": {
            "data": [
                {
                    "DATE": "2026-09-18 00:00:00",
                    "SCODE": "600519",
                    "RZYE": 5.0e9,
                    "RZMRE": 1.2e8,
                    "RZCHE": 9.0e7,
                    "RQYE": 3.0e7,
                    "RQMCL": 1.0e4,
                    "RQCHL": 9.0e3,
                    "RZRQYE": 5.03e9,
                }
            ]
        }
    }
    source = MarginSource(FakeClient({"datacenter": payload}))  # type: ignore[arg-type]
    frame = source.fetch("600519")
    assert frame.loc[0, "trade_date"] == "2026-09-18"
    assert frame.loc[0, "rzye"] == 5.0e9


# --------------------------------------------------------------------------- #
# 板块快照（T2）
# --------------------------------------------------------------------------- #


def test_sector_filter_keeps_the_literal_plus() -> None:
    """`fs=m:90+t:2` 里的 `+` 必须是**字面加号**（由 requests 编码成 %2B）。
    预编码成 %2B 会被双重编码，传成空格则过滤条件失效。"""

    payload = {
        "data": {
            "diff": [
                {"f12": "BK0475", "f14": "银行", "f3": 1.2, "f62": 1.5e8, "f104": 40, "f105": 2}
            ]
        }
    }
    client = FakeClient({"clist": payload})
    frame = SectorSnapshotSource(client).fetch()  # type: ignore[arg-type]
    params = client.calls[0][1] or {}
    assert params["fs"] == "m:90+t:2"
    assert frame.loc[0, "sector_name"] == "银行"
    assert frame.loc[0, "up_count"] == 40


def test_sector_empty_diff_is_empty_frame() -> None:
    source = SectorSnapshotSource(FakeClient({"clist": {"data": None}}))  # type: ignore[arg-type]
    assert source.fetch().empty


# --------------------------------------------------------------------------- #
# 通路诊断：不能把端点问题误诊成被封
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _FakeSession:
    """最小 requests.Session 替身：只实现 ThrottledClient 用到的那几处。"""

    def __init__(self, payload: object = None, boom: bool = False) -> None:
        self.headers: dict[str, str] = {}
        self._payload = payload
        self._boom = boom

    def get(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        if self._boom:
            raise ConnectionError("Remote end closed connection without response")
        return _FakeResponse(self._payload)


def _probe_client(payload: object = None, boom: bool = False) -> ThrottledClient:
    return ThrottledClient(
        min_interval=0.0,
        jitter=0.0,
        retries=1,
        backoff=0.0,
        session=_FakeSession(payload, boom=boom),
    )


def test_probe_reports_true_when_the_control_endpoint_answers() -> None:
    assert _probe_client({"data": {"klines": ["2026-09-18,1,2,3"]}}).probe() is True


def test_probe_reports_true_only_when_klines_are_present() -> None:
    """对照端点返回 200 但没有 klines，仍应判为不可用。"""

    assert _probe_client({"data": {"klines": []}}).probe() is False
    assert _probe_client({"data": None}).probe() is False


def test_probe_reports_false_when_even_the_control_endpoint_fails() -> None:
    """只有**对照端点**也失败，才能推断"网络/IP 层面不可用"。"""

    assert _probe_client(None, boom=True).probe() is False
