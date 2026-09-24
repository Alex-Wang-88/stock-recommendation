"""日线增量更新 —— 让本项目自己的行情归档每天前进一天。

数据源选择的实测依据（2026-09-19 复测）
------------------------------------
* **腾讯 `newfqkline`：8 并发 120/120 成功，全市场约 2.6 分钟**，且返回 10 个字段
  含**成交额**与换手率。
* **东财 `kline`/`clist`：本轮经本机代理 100% `ProxyError`**（同一会话内早些时候还能通）。
  ⇒ 不把日更建立在东财上。

因此日更**只用腾讯**。腾讯 `qfq` 与此前归档的 `ADJUST_PREV` **逐分一致**
（600519 在 2026-09-16 = 1258.000），成交额也**逐元一致**
（09-18 = 3,135,849,100 元，与东财 3,135,849,108 只差 8 元，是"万元"舍入）。

⚠️ 快照类接口不带日期字段
-----------------------
`qt.gtimg.cn` 的报价给的是"它此刻认为的最新价"——休市日调用它，拿到的是上一个交易日的
数据。若按系统时间盖章，周五的行情就会被写成周六，而周六在归档里"看起来是个交易日"，
之后所有按交易日对齐的逻辑都会错位。这正是题材归档踩过的**回退快照陷阱**。
⇒ 日期**只能从行情源自身取**：`resolve_quote_date()` 读报价里的时间戳。**绝不用系统时间。**

⚠️ 前复权序列会整体平移
---------------------
前复权价随新的除权除息**整体重算**。所以增量追加时必须做**重叠比对**：
把新拉的、与已存**历史日**重叠的那几根收盘价对一遍，不一致就说明该标的发生了公司行动，
必须整窗重拉。当天的日线会在盘中和收盘后被刷新，不能拿它来判断前复权漂移；否则会把
正常的当日涨跌误判成公司行动，并在除权日造出一个**虚假大跌** —— 而"低位"筛选恰好专挑这种票。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from .http import FetchError
from .market import BAR_COLUMNS, MarketStore, is_current_eod_ready

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
)

# 腾讯报价里时间戳所在下标（形如 20260918161436）。
QUOTE_TIME_INDEX = 30

# 日常增量抓多少根：需要覆盖「重叠比对」的窗口，多几根不吃亏。
INCREMENTAL_BARS = 15
# 首次见到某标的时抓多少根：够算 250 日分位 + 924 基准（约 480 个交易日）。
BOOTSTRAP_BARS = 700
# 公司行动整窗重拉时的根数。
REPAIR_BARS = 700

THREADS = 8
RETRIES = 3

# 昨收与已存收盘的允许偏差。无公司行动的两者应**完全相等**；
# 留余量只为容忍交易单位舍入，不是用来容纳真实波动的。
CORPORATE_ACTION_TOLERANCE = 0.002


@dataclass
class AdvanceResult:
    quote_date: str | None = None
    previous_date: str | None = None
    rows: int = 0
    new_rows: int = 0
    symbols_ok: int = 0
    symbols_new: int = 0
    repaired: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    up_to_date: bool = False
    elapsed: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.up_to_date:
            return f"已是最新：行情源与归档都止于 {self.previous_date}"
        lines = [
            (
                f"行情归档刷新 {self.quote_date}"
                if self.previous_date == self.quote_date
                else f"行情归档推进 {self.previous_date} → {self.quote_date}"
            )
            + f"（新增 {self.new_rows} 行 / 写入 {self.rows} 行含重叠窗口，"
            f"{self.symbols_ok} 只，其中新标的 {self.symbols_new} 只，"
            f"耗时 {self.elapsed:.0f}s）"
        ]
        if self.repaired:
            lines.append(
                f"  ⚠️ {len(self.repaired)} 只发生公司行动，已整窗重拉："
                f"{'、'.join(self.repaired[:10])}" + ("…" if len(self.repaired) > 10 else "")
            )
        if self.failed:
            lines.append(
                f"  ⚠️ {len(self.failed)} 只抓取失败（下次跑会自动重试）："
                f"{'、'.join(self.failed[:10])}" + ("…" if len(self.failed) > 10 else "")
            )
        lines.extend(f"  ! {note}" for note in self.notes)
        return "\n".join(lines)


class TencentSession:
    """线程局部的 requests 会话（每线程一个，避免共享连接被并发打断）。"""

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self, url: str, params: dict | None = None) -> requests.Response:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT, "Referer": "https://gu.qq.com/"})
            self._local.session = session
        return session.get(url, params=params, timeout=20)


def tencent_code(symbol: str) -> str:
    """裸 6 位码 → 腾讯代码（`sh` 沪市 / `sz` 深市与北交所）。"""

    code = str(symbol).strip().zfill(6)
    return f"sh{code}" if code[0] in {"5", "6", "9"} else f"sz{code}"


def resolve_quote_date(
    client: TencentSession, references: tuple[str, ...] = ("sh600519", "sh000001")
) -> str:
    """从**行情源自身**读取最新交易日，而不是用系统时间。"""

    response = client.get(TENCENT_QUOTE_URL + ",".join(references))
    response.encoding = "gbk"
    stamps: list[str] = []
    for line in response.text.strip().splitlines():
        if "~" not in line:
            continue
        parts = line.split("~")
        if len(parts) > QUOTE_TIME_INDEX:
            raw = parts[QUOTE_TIME_INDEX].strip()
            if len(raw) >= 8 and raw[:8].isdigit():
                stamps.append(raw[:8])
    if not stamps:
        raise FetchError("无法从行情源解析出交易日期（报价时间戳缺失）")
    latest = max(stamps)
    return f"{latest[:4]}-{latest[4:6]}-{latest[6:8]}"


def fetch_kline(client: TencentSession, symbol: str, count: int = INCREMENTAL_BARS) -> pd.DataFrame:
    """单标的近 `count` 根前复权日线。

    腾讯 `newfqkline` 行格式：
    `[日期, 开, 收, 高, 低, 成交量(手), {}, 换手%, 成交额(万元), ""]`
    """

    code = tencent_code(symbol)
    payload = client.get(
        TENCENT_KLINE_URL, params={"param": f"{code},day,,,{int(count)},qfq"}
    ).json()
    node = (payload.get("data") or {}).get(code) or {}
    key = next((name for name in node if str(name).endswith("day")), None)
    if key is None:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    records = []
    for row in node[key]:
        if len(row) < 6:
            continue
        records.append(
            {
                "symbol": str(symbol).zfill(6),
                "trade_date": pd.Timestamp(row[0]),
                "open": _float(row[1]),
                "close": _float(row[2]),
                "high": _float(row[3]),
                "low": _float(row[4]),
                "volume": _lots_to_shares(row[5]),
                # 腾讯给的是**万元**，归档统一存**元**。
                "amount": _wan_to_yuan(row[8]) if len(row) > 8 else None,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    # 昨收由序列自身推得（前复权序列内部自洽）；首根为 NaN，所以抓取要带重叠窗口。
    frame["pre_close"] = frame["close"].shift(1)
    return frame.loc[:, list(BAR_COLUMNS)]


def _float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _lots_to_shares(value: object) -> int | None:
    """成交量「手」→「股」。归档口径是股（实测 ×100 后逐股吻合）。"""

    number = _float(value)
    return None if number is None else int(round(number * 100))


def _wan_to_yuan(value: object) -> float | None:
    number = _float(value)
    return None if number is None else number * 10_000.0


def advance_bars(
    store: MarketStore,
    *,
    symbols: list[str] | None = None,
    count: int = INCREMENTAL_BARS,
    workers: int = THREADS,
    progress_every: int = 1000,
    progress: Callable[[int, int, str], None] | None = None,
) -> AdvanceResult:
    """把行情归档推进到行情源的最新交易日。

    两次抓取：第一遍按增量窗口抓所有标的；第二遍只对**重叠价不一致**
    （= 发生公司行动）的标的整窗重拉。最后**一次性写盘** ——
    `upsert_bars` 会重写整个 parquet，逐标的调用会把 150MB 写 5000 遍。
    """

    started = time.time()
    client = TencentSession()
    result = AdvanceResult()
    result.previous_date = store.latest_trade_date() if store.ready else None

    try:
        result.quote_date = resolve_quote_date(client)
    except FetchError as error:
        result.notes.append(f"无法确定交易日：{error}")
        result.failed.append("<quote-date>")
        result.elapsed = time.time() - started
        return result

    if result.previous_date and result.quote_date < result.previous_date:
        result.up_to_date = True
        result.elapsed = time.time() - started
        return result
    if (
        result.previous_date
        and result.quote_date == result.previous_date
        and not is_current_eod_ready(result.quote_date)
    ):
        result.up_to_date = True
        result.elapsed = time.time() - started
        return result
    if result.previous_date and result.quote_date == result.previous_date:
        result.notes.append("已过 15:30，刷新当天日线以覆盖盘中快照。")

    if symbols is None:
        symbols = _known_symbols(store)
    codes = [str(symbol).zfill(6) for symbol in symbols]

    frames: list[pd.DataFrame] = []

    def pull(code: str, bars: int) -> tuple[str, pd.DataFrame | None]:
        for attempt in range(1, RETRIES + 1):
            try:
                return code, fetch_kline(client, code, bars)
            except Exception:  # noqa: BLE001 - 逐标的失败不影响整体
                if attempt == RETRIES:
                    return code, None
                time.sleep(1.5 * attempt)
        return code, None  # pragma: no cover

    stored_spans = _stored_spans(store, codes)
    expected_dates = store.trade_dates()
    if result.quote_date not in expected_dates:
        expected_dates.append(result.quote_date)

    batch: list[tuple[str, int]] = []
    for code in codes:
        span = stored_spans.get(code)
        if span is None:
            request_count = BOOTSTRAP_BARS
        else:
            stored_count, first_date, last_date = span
            historical_dates = [
                day for day in expected_dates if first_date <= day <= last_date
            ]
            has_interior_gap = stored_count < len(historical_dates)
            new_dates = [day for day in expected_dates if day > last_date]
            if has_interior_gap:
                # Pull from the first locally stored date to the current source
                # date so an interior hole cannot become invisible after the
                # latest date is appended.
                request_count = max(
                    count,
                    len([day for day in expected_dates if day >= first_date]) + 2,
                )
            elif new_dates:
                # Downtime is handled by requesting the whole missing tail plus
                # overlap, not by assuming that 15 bars always covers it.
                request_count = max(count, len(new_dates) + 2)
            else:
                request_count = count
        batch.append((code, request_count))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (code, frame) in enumerate(
            pool.map(lambda item: pull(*item), batch), start=1
        ):
            if frame is None or frame.empty:
                result.failed.append(code)
            else:
                if code not in stored_spans:
                    result.symbols_new += 1
                frames.append(frame)
            if progress:
                progress(index, len(batch), "initial")
            if progress_every and index % progress_every == 0:
                print(f"  … 第一遍 {index}/{len(batch)}", flush=True)

    # 第二遍：公司行动整窗重拉
    # 当天日线可能已经在盘中写入过一次，收盘后刷新时价格变化是正常行情，
    # 不能把它当成前复权序列漂移。公司行动检测只比较 quote_date 之前的完整历史日。
    drifted = [code for code in _find_drifted(store, frames, before_date=result.quote_date)]
    if drifted:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (code, frame) in enumerate(
                pool.map(lambda c: pull(c, REPAIR_BARS), drifted), start=1
            ):
                if frame is None or frame.empty:
                    result.failed.append(code)
                else:
                    frames = [f for f in frames if str(f["symbol"].iloc[0]) != code]
                    frames.append(frame)
                    result.repaired.append(code)
                if progress:
                    progress(index, len(drifted), "repair")

    if not frames:
        result.notes.append("没有任何标的抓到数据 —— 归档未改动。")
        result.elapsed = time.time() - started
        return result

    combined = pd.concat(frames, ignore_index=True)
    before = store.row_count()
    result.rows = store.upsert_bars(combined)
    # `rows` 是**写入**行数（含与库内重叠的窗口），不能读成"新增了这么多天"。
    # 两者差得很远：全市场 15 根重叠 × 5300 只 ≈ 8 万行，而真正的新增只有 1 万行。
    result.new_rows = max(store.row_count() - before, 0)
    result.symbols_ok = int(combined["symbol"].nunique())
    latest = str(combined["trade_date"].max())[:10]
    if latest < result.quote_date:
        result.notes.append(
            f"抓到的最后一天是 {latest}，但行情源的最新交易日是 {result.quote_date}"
            "（可能全部标的在该日均无成交）。"
        )
    result.elapsed = time.time() - started
    return result


def _known_symbols(store: MarketStore) -> list[str]:
    if not store.ready:
        raise FileNotFoundError(
            f"行情归档不存在：{store.path}。先跑 `recommend sync-bars` 引导一次。"
        )
    return sorted(
        store.conn.execute(f"select distinct symbol from '{store.path.as_posix()}'")
        .df()["symbol"]
        .astype(str)
        .str.zfill(6)
        .tolist()
    )


def symbols_for_refresh(
    store: MarketStore, reference_source: str | Path | None = None
) -> list[str]:
    """Union local symbols with the project's local instrument reference.

    The old implementation only refreshed symbols already present in the
    Parquet file, which made every new listing invisible forever.  Keeping the
    existing symbols in the union also preserves historical/delisted rows.
    """

    symbols = set(_known_symbols(store))
    if reference_source is not None:
        meta = store.symbol_meta(reference_source)
        if not meta.empty and "symbol" in meta.columns:
            symbols.update(meta["symbol"].astype(str).str.zfill(6).tolist())
    return sorted(symbols)


def _stored_spans(store: MarketStore, codes: list[str]) -> dict[str, tuple[int, str, str]]:
    """Return row count and date span for each symbol in the local archive."""

    if not store.ready or not codes:
        return {}
    frame = store.conn.execute(
        f"""
        select cast(symbol as varchar) as symbol,
               count(distinct trade_date) as day_count,
               min(trade_date) as first_date,
               max(trade_date) as last_date
        from '{store.path.as_posix()}'
        group by symbol
        """
    ).df()
    wanted = set(codes)
    return {
        str(row["symbol"]).zfill(6): (
            int(row["day_count"]),
            str(row["first_date"])[:10],
            str(row["last_date"])[:10],
        )
        for _, row in frame.iterrows()
        if str(row["symbol"]).zfill(6) in wanted
    }


def _stored_last_dates(store: MarketStore, codes: list[str]) -> dict[str, str]:
    if not store.ready:
        return {}
    frame = store.conn.execute(
        f"select symbol, max(trade_date) as last from '{store.path.as_posix()}' group by symbol"
    ).df()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    wanted = set(codes)
    return {
        row["symbol"]: str(row["last"])[:10]
        for _, row in frame.iterrows()
        if row["symbol"] in wanted
    }


def _find_drifted(
    store: MarketStore,
    frames: list[pd.DataFrame],
    *,
    before_date: str | None = None,
) -> list[str]:
    """找出历史重叠收盘价不一致的标的（前复权被平移过）。

    ``before_date`` 是行情源当前日期。当天日线可能先以盘中快照写入，
    收盘后再次抓取时必然会变化；因此公司行动检测只允许使用它之前的历史日。
    """

    if not store.ready or not frames:
        return []
    incoming = pd.concat(frames, ignore_index=True)
    stored = store.conn.execute(
        f"select symbol, trade_date, close from '{store.path.as_posix()}'"
    ).df()
    stored["symbol"] = stored["symbol"].astype(str).str.zfill(6)
    stored["trade_date"] = pd.to_datetime(stored["trade_date"])
    merged = incoming.merge(
        stored, on=["symbol", "trade_date"], how="inner", suffixes=("_new", "_old")
    )
    if merged.empty:
        return []
    if before_date:
        merged = merged[merged["trade_date"] < pd.Timestamp(before_date)]
        if merged.empty:
            return []
    merged = merged[merged["close_old"].notna() & (merged["close_old"] != 0)]
    ratio = merged["close_new"] / merged["close_old"]
    drifted = merged.loc[(ratio - 1.0).abs() > CORPORATE_ACTION_TOLERANCE, "symbol"]
    return sorted(set(drifted.astype(str)))


def local_now() -> str:
    """仅用于日志/文件名，**不用于给行情盖章**。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


__all__ = [
    "BOOTSTRAP_BARS",
    "CORPORATE_ACTION_TOLERANCE",
    "INCREMENTAL_BARS",
    "AdvanceResult",
    "TencentSession",
    "advance_bars",
    "fetch_kline",
    "local_now",
    "resolve_quote_date",
    "symbols_for_refresh",
    "tencent_code",
]
