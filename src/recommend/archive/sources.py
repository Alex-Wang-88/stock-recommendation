"""数据源适配：把各家接口的原始返回规范化成归档表所需的列。

分层约定（见 docs/stock-recommendation-system-plan.md）
-------------------------------------------------------
* **T1 可回补**：题材归因（同花顺）、龙虎榜（东财主源 + 新浪备用）、两融（上交所 / 深交所官方）。
  都按日期或交易日参数回补历史。
* **T2 仅当日快照**：板块排名与资金流。没有历史参数 —— 每天不落盘就永远缺这一天。

**单位约定**：金额一律存**元**（原始单位，不做万/亿换算），避免在各层转换中丢精度。
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from io import BytesIO, StringIO
from typing import Any

import pandas as pd

from ..http import FetchError, ThrottledClient

DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"

SSE_MARGIN_URL = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
SZSE_MARGIN_URL = "https://www.szse.cn/api/report/ShowReport"

class FallbackResponseError(RuntimeError):
    """接口没有该日的数据，**回退**返回了另一个日期的快照。

    这是实测踩到的坑：同花顺 `getharden` 对"没有数据的日期"不报错、不返空，
    而是**静默返回它自己最新的那份快照**。于是 38 个法定休市日全部被写进了
    同一个快照（同一套 79 只股票），把归档污染了。

    两道校验（都在本数据集上**零误报**）：
    1. **行级 `date` 回显**必须等于请求日期。节假日请求 2024-10-01 时回显的是
       接口当天的最新日（实测 2026-09-18），直接露馅。
    2. **必须带行情字段**。实测 496 个真实交易日 **100%** 有 `change_pct`
       （42218/42218 行），而回退响应 **0%** 有（3081 行全 NULL）。
    """


THEME_COLUMNS = (
    "symbol",
    "name",
    "reason",
    "close",
    "change_pct",
    "turnover_pct",
    "amount",
    "volume",
    "big_order_net",
    "market",
    "source_date",
)
DRAGON_TIGER_COLUMNS = (
    "trade_date",
    "symbol",
    "name",
    "explanation",
    "close",
    "change_pct",
    "net_buy",
    "buy_amt",
    "sell_amt",
    "turnover_pct",
)
MARGIN_COLUMNS = (
    "trade_date",
    "symbol",
    "rzye",
    "rzmre",
    "rzche",
    "rqye",
    "rqmcl",
    "rqchl",
    "rzrqye",
)
SECTOR_COLUMNS = (
    "sector_code",
    "sector_name",
    "change_pct",
    "main_net_inflow",
    "up_count",
    "down_count",
)


def _num(value: Any) -> float | None:
    """稳健转数：缺失/占位符统一成 None，而不是悄悄变 0。"""

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value in {"", "-", "--", "—"}:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date10(value: Any) -> str:
    """把官方接口可能返回的日期/时间值统一成 YYYY-MM-DD。"""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if len(text) >= 8 and text[:8].isdigit() and "-" not in text[:8]:
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text[:10]


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


# --------------------------------------------------------------------------- #
# T1 · 题材归因（同花顺）
# --------------------------------------------------------------------------- #


class ThemeSource:
    """同花顺当日强势股 + 题材归因。

    `date` 是**路径参数**，所以历史可以逐日回补 —— 本轮实测：
    2025-06-16 → 73 条、2024-09-30 → 693 条，均正常返回。
    """

    URL = (
        "http://zx.10jqka.com.cn/event/api/getharden/"
        "date/{date}/orderby/date/orderway/desc/charset/GBK/"
    )

    def __init__(self, client: ThrottledClient) -> None:
        self.client = client

    def fetch(self, trade_date: str) -> pd.DataFrame:
        url = self.URL.format(date=trade_date)
        response = self.client.get(url)
        try:
            payload = response.json()
        except ValueError:
            # 接口标注 GBK；实测为 UTF-8。两条路都试，避免编码差异导致静默空表。
            response.encoding = "gbk"
            payload = response.json()

        error_code = payload.get("errocode", 0)
        if error_code not in (0, None):
            raise RuntimeError(f"同花顺接口报错：{payload.get('errormsg', '')}")

        rows = payload.get("data") or []
        if not rows:
            return pd.DataFrame(columns=THEME_COLUMNS)

        frame = pd.DataFrame(rows)

        # ── 校验 1：行级日期回显必须等于请求日期 ──────────────────────────
        # 接口对"没有数据的日期"会回退返回它最新的一份快照，并在行里回显**那个**
        # 日期。不回显就对不上 —— 这是识别回退最直接的证据。
        echoed = frame.get("date")
        if echoed is not None and echoed.notna().any():
            actual = str(echoed.dropna().iloc[0])[:10]
            if actual != trade_date:
                raise FallbackResponseError(
                    f"请求 {trade_date}，接口回退返回了 {actual} 的快照"
                    f"（该日无数据，通常是休市日）"
                )

        # ── 校验 2：真实交易日必须带行情字段 ─────────────────────────────
        normalized = pd.DataFrame(
            {
                "symbol": frame.get("code"),
                "name": frame.get("name"),
                "reason": frame.get("reason"),
                "close": frame.get("close", pd.Series(dtype=float)).map(_num),
                "change_pct": frame.get("zhangfu", pd.Series(dtype=float)).map(_num),
                "turnover_pct": frame.get("huanshou", pd.Series(dtype=float)).map(_num),
                "amount": frame.get("chengjiaoe", pd.Series(dtype=float)).map(_num),
                "volume": frame.get("chengjiaoliang", pd.Series(dtype=float)).map(_num),
                "big_order_net": frame.get("ddejingliang", pd.Series(dtype=float)).map(_num),
                "market": frame.get("market"),
                "source_date": trade_date,
            }
        )
        if normalized["change_pct"].isna().all():
            raise FallbackResponseError(
                f"{trade_date} 的响应不含任何行情字段（实测回退响应 100% 无行情、"
                "真实交易日 100% 有）"
            )

        normalized = normalized.dropna(subset=["symbol"])
        normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
        return normalized.loc[:, list(THEME_COLUMNS)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# T1 · 龙虎榜 / 两融（东财 datacenter）
# --------------------------------------------------------------------------- #


def datacenter_query(
    client: ThrottledClient,
    report_name: str,
    filter_str: str,
    page_size: int = 500,
    sort_columns: str = "",
    sort_types: str = "-1",
    columns: str = "ALL",
) -> list[dict[str, Any]]:
    """东财数据中心统一查询（龙虎榜 / 两融 / 解禁 / 大宗 / 股东户数 / 分红 共用）。"""

    params = {
        "reportName": report_name,
        "columns": columns,
        "filter": filter_str,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "sortColumns": sort_columns,
        "sortTypes": sort_types,
        "source": "WEB",
        "client": "WEB",
    }
    payload = client.get_json(DATACENTER_URL, params=params)
    if not isinstance(payload, dict):
        raise RuntimeError("东财数据中心返回了无法解析的响应")
    result = payload.get("result") or {}
    if not isinstance(result, dict):
        return []
    rows = list(result.get("data") or [])

    # 龙虎榜单日可能超过默认页容量。东财返回 `pages` 时逐页补齐；
    # 没有该字段的旧/假响应仍按单页处理，保持兼容。
    try:
        pages = max(1, int(result.get("pages") or 1))
    except (TypeError, ValueError):
        pages = 1
    for page in range(2, pages + 1):
        params["pageNumber"] = str(page)
        next_payload = client.get_json(DATACENTER_URL, params=params)
        if not isinstance(next_payload, dict):
            raise RuntimeError(f"东财数据中心第 {page} 页返回了无法解析的响应")
        next_result = next_payload.get("result") or {}
        if not isinstance(next_result, dict):
            continue
        rows.extend(next_result.get("data") or [])
    return rows


class DragonTigerSource:
    """全市场龙虎榜（T1，按 `TRADE_DATE` 可查历史）。"""

    REPORT = "RPT_DAILYBILLBOARD_DETAILSNEW"

    def __init__(self, client: ThrottledClient) -> None:
        self.client = client

    def fetch(self, trade_date: str) -> pd.DataFrame:
        rows = datacenter_query(
            self.client,
            self.REPORT,
            filter_str=f"(TRADE_DATE>='{trade_date}')(TRADE_DATE<='{trade_date}')",
            page_size=5000,
            sort_columns="SECURITY_CODE,TRADE_DATE",
            sort_types="1,-1",
        )
        if not rows:
            return pd.DataFrame(columns=DRAGON_TIGER_COLUMNS)
        frame = pd.DataFrame(rows)
        normalized = pd.DataFrame(
            {
                "trade_date": frame.get("TRADE_DATE", pd.Series(dtype=object)).astype(str).str[:10],
                "symbol": frame.get("SECURITY_CODE"),
                "name": frame.get("SECURITY_NAME_ABBR"),
                "explanation": frame.get("EXPLANATION"),
                "close": frame.get("CLOSE_PRICE", pd.Series(dtype=float)).map(_num),
                "change_pct": frame.get("CHANGE_RATE", pd.Series(dtype=float)).map(_num),
                "net_buy": frame.get("BILLBOARD_NET_AMT", pd.Series(dtype=float)).map(_num),
                "buy_amt": frame.get("BILLBOARD_BUY_AMT", pd.Series(dtype=float)).map(_num),
                "sell_amt": frame.get("BILLBOARD_SELL_AMT", pd.Series(dtype=float)).map(_num),
                "turnover_pct": frame.get("TURNOVERRATE", pd.Series(dtype=float)).map(_num),
            }
        )
        normalized = normalized.dropna(subset=["symbol"])
        normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
        return normalized.loc[:, list(DRAGON_TIGER_COLUMNS)].reset_index(drop=True)


class SinaDragonTigerSource:
    """新浪龙虎榜备用源。

    新浪页面能提供「哪些股票上榜」和上榜指标，但没有可与东财严格对齐的买卖席位
    净额。因此备用源**只填上榜事件、股票名、收盘价和指标说明**，不伪造
    `net_buy` / `buy_amt` / `sell_amt`，避免把缺失误当成零资金流。
    """

    URL = "https://vip.stock.finance.sina.com.cn/q/go.php/vInvestConsult/kind/lhb/index.phtml"

    def __init__(self, client: ThrottledClient) -> None:
        self.client = client

    @staticmethod
    def _flatten_columns(columns: object) -> list[str]:
        flattened: list[str] = []
        for column in columns:  # type: ignore[union-attr]
            if isinstance(column, tuple):
                parts = [str(item) for item in column if str(item).lower() != "nan"]
                flattened.append("".join(parts).strip())
            else:
                flattened.append(str(column).strip())
        return flattened

    @staticmethod
    def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
        for candidate in candidates:
            for column in columns:
                if candidate in column:
                    return column
        return None

    @staticmethod
    def _empty_value(frame: pd.DataFrame) -> pd.Series:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")

    @staticmethod
    def _title(table: pd.DataFrame) -> str | None:
        if table.empty:
            return None
        for value in table.iloc[0].tolist():
            if value is not None and not pd.isna(value):
                text = str(value).strip()
                if text and text not in {"序号", "股票代码", "股票名称"}:
                    return text
        return None

    def fetch(self, trade_date: str) -> pd.DataFrame:
        response = self.client.get(self.URL, params={"tradedate": trade_date})
        html = getattr(response, "text", "")
        if not html:
            content = getattr(response, "content", b"")
            html = content.decode("utf-8", errors="replace") if content else ""
        if not html:
            raise RuntimeError("新浪龙虎榜响应为空")

        try:
            tables = pd.read_html(StringIO(html), displayed_only=False)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "新浪龙虎榜备用源需要 HTML 解析依赖，请安装 `pip install -e '.[free-data]'`"
            ) from exc
        except ValueError as exc:
            if "暂无" in html or "没有数据" in html:
                return _empty(DRAGON_TIGER_COLUMNS)
            raise RuntimeError(f"新浪龙虎榜页面没有可解析的表格：{exc}") from exc

        chunks: list[pd.DataFrame] = []
        for table in tables:
            table = table.copy()
            table.columns = self._flatten_columns(table.columns)
            code_column = self._find_column(table.columns, ("股票代码", "证券代码", "代码"))
            name_column = self._find_column(table.columns, ("股票名称", "证券简称", "名称"))
            close_column = self._find_column(table.columns, ("收盘价", "收盘"))
            indicator_column = self._find_column(table.columns, ("指标", "上榜原因"))
            data = table
            title = self._title(table)
            if code_column is None or name_column is None:
                # 新浪真实页面把表头放在第二行，第一行是合并单元格的上榜原因，
                # read_html 因此会把列名读成 0..7。按表头行定位，避免依赖脆弱的 CSS。
                header_row: int | None = None
                code_position: int | None = None
                for row_index in range(min(3, len(table))):
                    for position, value in enumerate(table.iloc[row_index].tolist()):
                        if "股票代码" in str(value):
                            header_row = row_index
                            code_position = position
                            break
                    if header_row is not None:
                        break
                if (
                    header_row is None
                    or code_position is None
                    or table.shape[1] <= code_position + 2
                ):
                    continue
                data = table.iloc[header_row + 1 :].copy()
                code_values = data.iloc[:, code_position]
                name_values = data.iloc[:, code_position + 1]
                close_values = data.iloc[:, code_position + 2]
                indicator_values = pd.Series(
                    [title] * len(data), index=data.index, dtype="object"
                )
            else:
                code_values = table[code_column]
                name_values = table[name_column]
                close_values = table[close_column] if close_column is not None else None
                indicator_values = (
                    table[indicator_column]
                    if indicator_column is not None
                    else self._empty_value(table)
                )

            symbols = code_values.astype(str).str.extract(r"(\d+)", expand=False)
            symbols = symbols.dropna().str.zfill(6)
            if symbols.empty:
                continue
            chunk = pd.DataFrame(index=data.index)
            chunk["trade_date"] = trade_date
            chunk["symbol"] = symbols
            chunk["name"] = name_values
            chunk["explanation"] = indicator_values
            chunk["close"] = (
                close_values.map(_num)
                if close_values is not None
                else self._empty_value(data)
            )
            # 新浪页面的「对应值」不是统一定义的涨跌幅，且没有席位买卖金额。
            # 这些字段保持缺失，绝不填零。
            chunk["change_pct"] = self._empty_value(table)
            chunk["net_buy"] = self._empty_value(table)
            chunk["buy_amt"] = self._empty_value(table)
            chunk["sell_amt"] = self._empty_value(table)
            chunk["turnover_pct"] = self._empty_value(table)
            # 真实页的每只股票后面还跟着一个「查看详情/营业部」行，里面可能
            # 夹带别的数字。只有同时有 6 位代码和收盘价的行才是股票主表记录。
            chunk = chunk.loc[chunk["symbol"].notna() & chunk["close"].notna()].copy()
            if chunk.empty:
                continue
            chunks.append(chunk.loc[:, list(DRAGON_TIGER_COLUMNS)])

        if not chunks:
            if "暂无" in html or "没有数据" in html:
                return _empty(DRAGON_TIGER_COLUMNS)
            raise RuntimeError("新浪龙虎榜页面结构发生变化，未识别到股票代码表")

        combined = pd.concat(chunks, ignore_index=True)
        rows: list[pd.Series] = []
        for _symbol, group in combined.groupby("symbol", sort=False):
            row = group.iloc[0].copy()
            explanations = [
                str(value).strip()
                for value in group["explanation"]
                if value is not None and not pd.isna(value) and str(value).strip()
            ]
            row["explanation"] = "|".join(dict.fromkeys(explanations)) or None
            rows.append(row)
        return pd.DataFrame(rows, columns=list(DRAGON_TIGER_COLUMNS)).reset_index(drop=True)


class DragonTigerWithFallbackSource:
    """东财主源 + 新浪事件备用源。只在主源失败时切换，不在主源空返回时猜测。"""

    def __init__(self, client: ThrottledClient) -> None:
        self.primary = DragonTigerSource(client)
        self.fallback = SinaDragonTigerSource(client)

    def fetch(self, trade_date: str) -> pd.DataFrame:
        try:
            return self.primary.fetch(trade_date)
        except (FetchError, RuntimeError, ValueError) as primary_error:
            try:
                return self.fallback.fetch(trade_date)
            except (FetchError, RuntimeError, ValueError) as fallback_error:
                raise RuntimeError(
                    f"龙虎榜主源失败且新浪备用源也失败；主源：{primary_error}；"
                    f"备用源：{fallback_error}"
                ) from fallback_error


class MarginSource:
    """融资融券明细（T1，按 `SCODE` 可查历史）。"""

    REPORT = "RPTA_WEB_RZRQ_GGMX"

    def __init__(self, client: ThrottledClient) -> None:
        self.client = client

    def fetch(self, symbol: str, page_size: int = 60) -> pd.DataFrame:
        rows = datacenter_query(
            self.client,
            self.REPORT,
            filter_str=f'(SCODE="{symbol}")',
            page_size=page_size,
            sort_columns="DATE",
        )
        if not rows:
            return pd.DataFrame(columns=MARGIN_COLUMNS)
        frame = pd.DataFrame(rows)
        normalized = pd.DataFrame(
            {
                "trade_date": frame.get("DATE", pd.Series(dtype=object)).astype(str).str[:10],
                "symbol": frame.get("SCODE"),
                "rzye": frame.get("RZYE", pd.Series(dtype=float)).map(_num),
                "rzmre": frame.get("RZMRE", pd.Series(dtype=float)).map(_num),
                "rzche": frame.get("RZCHE", pd.Series(dtype=float)).map(_num),
                "rqye": frame.get("RQYE", pd.Series(dtype=float)).map(_num),
                "rqmcl": frame.get("RQMCL", pd.Series(dtype=float)).map(_num),
                "rqchl": frame.get("RQCHL", pd.Series(dtype=float)).map(_num),
                "rzrqye": frame.get("RZRQYE", pd.Series(dtype=float)).map(_num),
            }
        )
        normalized = normalized.dropna(subset=["symbol"])
        normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
        return normalized.loc[:, list(MARGIN_COLUMNS)].reset_index(drop=True)


class OfficialMarginSource:
    """上交所 + 深交所官方融资融券明细（按交易日、全市场）。

    这条链路不依赖东财端点：上交所返回 JSON，深交所返回 XLSX。两边的字段能
    覆盖现有 `margin_trading` 表所需的核心余额/买入/偿还量；官方接口没有提供的
    字段保持缺失，不用零值冒充真实观测。
    """

    SSE_HEADERS = {"Referer": "https://www.sse.com.cn/"}
    SZSE_HEADERS = {"Referer": "https://www.szse.cn/disclosure/margin/margin/index.html"}

    def __init__(self, client: ThrottledClient) -> None:
        self.client = client

    def fetch_sse(self, trade_date: str) -> pd.DataFrame:
        params = {
            "isPagination": "true",
            "tabType": "mxtype",
            "detailsDate": trade_date.replace("-", ""),
            "stockCode": "",
            "beginDate": "",
            "endDate": "",
            "pageHelp.pageSize": "5000",
            "pageHelp.pageCount": "50",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.cacheSize": "1",
            "pageHelp.endPage": "21",
        }
        payload = self.client.get_json(SSE_MARGIN_URL, params=params, headers=self.SSE_HEADERS)
        if not isinstance(payload, dict):
            raise RuntimeError("上交所两融接口返回了无法解析的响应")
        rows = payload.get("result") or []
        if not isinstance(rows, list):
            raise RuntimeError("上交所两融接口的 result 不是列表")
        if not rows:
            return _empty(MARGIN_COLUMNS)

        raw = pd.DataFrame(rows)
        if raw.shape[1] < 13:
            raise RuntimeError(f"上交所两融字段数异常：收到 {raw.shape[1]} 列，预期至少 13 列")
        raw_date = raw.iloc[:, 1].map(_date10)
        echoed = raw_date[raw_date.notna()]
        if not echoed.empty and (echoed != trade_date).any():
            actual = str(echoed.iloc[0])
            raise FallbackResponseError(f"请求 {trade_date}，上交所回退返回了 {actual} 的快照")

        normalized = pd.DataFrame(
            {
                "trade_date": raw_date,
                "symbol": raw.iloc[:, 12],
                "rzye": raw.iloc[:, 10].map(_num),
                "rzmre": raw.iloc[:, 8].map(_num),
                "rzche": raw.iloc[:, 7].map(_num),
                "rqye": raw.iloc[:, 4].map(_num),
                "rqmcl": raw.iloc[:, 3].map(_num),
                "rqchl": raw.iloc[:, 2].map(_num),
                "rzrqye": pd.Series([None] * len(raw), index=raw.index, dtype="object"),
            }
        )
        return self._clean(normalized)

    def fetch_szse(self, trade_date: str) -> pd.DataFrame:
        params = {
            "SHOWTYPE": "xlsx",
            "CATALOGID": "1837_xxpl",
            "txtDate": trade_date,
            "tab2PAGENO": "1",
            "random": "0.5",
            "TABKEY": "tab2",
        }
        response = self.client.get(SZSE_MARGIN_URL, params=params, headers=self.SZSE_HEADERS)
        content = getattr(response, "content", b"")
        if not content:
            raise RuntimeError("深交所两融接口返回空文件")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Workbook contains no default style", category=UserWarning
                )
                raw = pd.read_excel(
                    BytesIO(content), engine="openpyxl", dtype={"证券代码": str}
                )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "深交所两融 XLSX 解析需要 openpyxl，请安装 `pip install -e '.[free-data]'`"
            ) from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"深交所两融 XLSX 无法解析：{exc}") from exc

        if raw.empty:
            return _empty(MARGIN_COLUMNS)
        raw.columns = [str(column).strip() for column in raw.columns]
        expected = [
            "证券代码",
            "证券简称",
            "融资买入额",
            "融资余额",
            "融券卖出量",
            "融券余量",
            "融券余额",
            "融资融券余额",
        ]
        if not set(expected).issubset(raw.columns):
            if raw.shape[1] < len(expected):
                raise RuntimeError(f"深交所两融字段异常：{list(raw.columns)}")
            raw = raw.iloc[:, : len(expected)].copy()
            raw.columns = expected

        normalized = pd.DataFrame(
            {
                "trade_date": trade_date,
                "symbol": raw["证券代码"],
                "rzye": raw["融资余额"].map(_num),
                "rzmre": raw["融资买入额"].map(_num),
                "rzche": pd.Series([None] * len(raw), index=raw.index, dtype="object"),
                "rqye": raw["融券余量"].map(_num),
                "rqmcl": raw["融券卖出量"].map(_num),
                "rqchl": pd.Series([None] * len(raw), index=raw.index, dtype="object"),
                "rzrqye": raw["融资融券余额"].map(_num),
            }
        )
        return self._clean(normalized)

    @staticmethod
    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.dropna(subset=["symbol"]).copy()
        normalized["symbol"] = (
            normalized["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
        )
        normalized = normalized.dropna(subset=["symbol"])
        normalized["symbol"] = normalized["symbol"].str.zfill(6)
        normalized = normalized.drop_duplicates(subset=["trade_date", "symbol"], keep="first")
        return normalized.loc[:, list(MARGIN_COLUMNS)].reset_index(drop=True)

    def fetch(self, trade_date: str) -> pd.DataFrame:
        sse = self.fetch_sse(trade_date)
        szse = self.fetch_szse(trade_date)
        if sse.empty and szse.empty:
            return _empty(MARGIN_COLUMNS)
        if sse.empty != szse.empty:
            raise RuntimeError(
                f"{trade_date} 两融官方数据不完整：上交所 {len(sse)} 行、"
                f"深交所 {len(szse)} 行；拒绝写入半日快照"
            )
        combined = pd.concat([sse, szse], ignore_index=True)
        return self._clean(combined)


# --------------------------------------------------------------------------- #
# T2 · 板块快照（仅当日，必须自建归档）
# --------------------------------------------------------------------------- #


class SectorSnapshotSource:
    """行业板块排名 + 板块主力资金流（**只有当日**，没有历史参数）。

    ⇒ 这是 T2：**每天不落盘就永远缺这一天**，所以归档任务必须先于模型上线。

    注意 `fs=m:90+t:2` 里的 `+` 在查询串里会被解码成空格，必须传编码后的值。
    """

    URL = "https://push2.eastmoney.com/api/qt/clist/get"
    FIELDS = "f3,f12,f14,f62,f104,f105"

    def __init__(self, client: ThrottledClient) -> None:
        self.client = client

    def fetch(self) -> pd.DataFrame:
        params = {
            "pn": "1",
            "pz": "500",
            "po": "1",
            "np": "1",
            "fltt": "2",
            "invt": "2",
            "fs": "m:90+t:2",  # requests 会编码成 m%3A90%2Bt%3A2
            "fields": self.FIELDS,
        }
        payload = self.client.get_json(
            self.URL, params=params, headers={"Referer": "https://quote.eastmoney.com/"}
        )
        items = (payload.get("data") or {}).get("diff") or []
        if not items:
            return pd.DataFrame(columns=SECTOR_COLUMNS)
        frame = pd.DataFrame(items)
        normalized = pd.DataFrame(
            {
                "sector_code": frame.get("f12"),
                "sector_name": frame.get("f14"),
                "change_pct": frame.get("f3", pd.Series(dtype=float)).map(_num),
                "main_net_inflow": frame.get("f62", pd.Series(dtype=float)).map(_num),
                "up_count": frame.get("f104", pd.Series(dtype=float)).map(_num),
                "down_count": frame.get("f105", pd.Series(dtype=float)).map(_num),
            }
        )
        normalized = normalized.dropna(subset=["sector_code"])
        for column in ("up_count", "down_count"):
            normalized[column] = normalized[column].astype("Int64")
        return normalized.loc[:, list(SECTOR_COLUMNS)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 交易日历
# --------------------------------------------------------------------------- #


def candidate_dates(start: str, end: str) -> list[str]:
    """生成待尝试的日期（仅工作日）。

    只做**工作日**粗筛，不内置节假日表 —— 真正的交易日由"接口是否返回数据"决定，
    并记进 `archive_runs`。这样节假日表和真实交易日的漂移不会污染归档。
    """

    begin = date.fromisoformat(start)
    finish = date.fromisoformat(end)
    if begin > finish:
        raise ValueError(f"起始日期晚于结束日期：{start} > {end}")
    days: list[str] = []
    cursor = begin
    while cursor <= finish:
        if cursor.weekday() < 5:
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days
