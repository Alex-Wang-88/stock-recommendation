"""行情数据层：读取本项目自己的日线，并提供「基准收盘价」查询。

为什么要有「基准收盘价」这个专用接口
------------------------------------
「924 以来涨幅」「年内涨幅」这类指标需要**某个历史日期的收盘价**作为分母。
朴素做法是把整段历史载进来再挑，但 5,300 只 × 730 天 ≈ 450 万行，
每次筛选都全量载入是浪费（滚动窗口只需要最近 ~300 天）。

所以拆成两件事：
* `bars()` —— 只取滚动窗口需要的那段。
* `closes_on()` —— **按需**取基准日的收盘价（取**不晚于**该日的最后一个交易日，
  因为基准日可能停牌，或标的当时还没上市）。

本模块所有拼进 SQL 的日期都过 `_iso()` 校验 —— 这些值将来会来自自然语言翻译层，
不能当可信输入。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

_ISO_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

BAR_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
)

# 默认滚动窗口（交易日数）。250 日位置分位需要 250 根，留 50 根缓冲。
DEFAULT_WINDOW = 300

# A 股收盘后的数据源（题材、龙虎榜、两融等）通常要到收盘后才会更新。
# 使用固定 UTC+8，不依赖 Windows 是否安装 IANA 时区数据库，保证项目搬到其它
# Windows 电脑后仍能按中国市场时间判断。
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")
A_SHARE_EOD_READY = time(15, 30)
A_SHARE_SYNC_BLOCK_START = time(9, 30)


def _shanghai_now(value: datetime | None = None) -> datetime:
    """返回带中国时区的当前时间；测试可传入固定时间。"""

    current = value or datetime.now(SHANGHAI_TIMEZONE)
    if current.tzinfo is None:
        return current.replace(tzinfo=SHANGHAI_TIMEZONE)
    return current.astimezone(SHANGHAI_TIMEZONE)


def is_current_eod_ready(trade_date: str, now: datetime | None = None) -> bool:
    """判断 ``trade_date`` 是否是今天且已经过收盘后数据安全线。"""

    current = _shanghai_now(now)
    return _iso(trade_date) == current.date().isoformat() and current.time() >= A_SHARE_EOD_READY


def intraday_sync_block_reason(now: datetime | None = None) -> str | None:
    """盘中暂停日常同步，避免把当日未完成行情写入本地推荐数据。"""

    current = _shanghai_now(now)
    if (
        current.weekday() < 5
        and A_SHARE_SYNC_BLOCK_START <= current.time() < A_SHARE_EOD_READY
    ):
        return (
            "工作日 09:30–15:30 暂停日常数据同步；页面继续读取本地最近一个完整交易日。"
            "请在 16:00 收盘后任务运行，或 15:30 后再手动同步。"
        )
    return None


def _iso(value: str) -> str:
    """校验并规范化 `YYYY-MM-DD`。

    这些日期可能来自自然语言翻译层，**必须**在进 SQL 之前校验 ——
    不然 `2024-09-24' or '1'='1` 这类输入会直接拼进查询。

    ⚠️ 用 `fullmatch` 匹配**整个**字符串，不要先 `[:10]` 再校验：
    截断会把 `2024-09-24' or '1'='1` 的前 10 个字符变成合法的 `2024-09-24`，
    校验通过而注入部分被悄悄保留到别处 —— 等于没校验。
    """

    text = str(value).strip()
    if not _ISO_PATTERN.fullmatch(text):
        raise ValueError(f"日期格式必须是 YYYY-MM-DD，收到：{value!r}")
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"不是合法日期：{value!r}") from error
    return text


@dataclass(frozen=True)
class SymbolMeta:
    symbol: str
    name: str
    listed_date: str | None = None


class MarketStore:
    """本项目自己的日线归档（parquet）。"""

    def __init__(self, path: str | Path = "data/market/daily.parquet") -> None:
        self.path = Path(path)
        self.conn = duckdb.connect()

    def __enter__(self) -> MarketStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.conn.close()

    def close(self) -> None:
        self.conn.close()

    # -- 引导 / 同步 --------------------------------------------------------- #

    def sync_from(self, source: str | Path, columns: tuple[str, ...] = BAR_COLUMNS) -> int:
        """从既有 parquet 归档复制日线（一次性引导）。

        只保留需要的列 —— 上游带 `bob`/`eob`/`frequency`/`vendor_symbol` 等本项目用不到的字段。
        这是 `推荐系统只复用归档层、不共享模型` 边界内的动作：行情是上游原始数据。
        """

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"引导源不存在：{source}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        column_list = ", ".join(columns)
        self.conn.execute(
            f"copy (select {column_list} from read_parquet('{source.as_posix()}')) "
            f"to '{self.path.as_posix()}' (format parquet, compression zstd)"
        )
        return self.row_count()

    # -- 读 ------------------------------------------------------------------ #

    @property
    def ready(self) -> bool:
        return self.path.exists()

    def _require(self) -> None:
        if not self.ready:
            raise FileNotFoundError(
                f"行情归档不存在：{self.path}。先运行 `recommend sync-bars` 引导一次。"
            )

    def row_count(self) -> int:
        self._require()
        path = self.path.as_posix()
        return int(self.conn.execute(f"select count(*) from '{path}'").fetchone()[0])

    def trade_dates(self, end: str | None = None, limit: int | None = None) -> list[str]:
        """交易日列表（升序）。

        行情里实际出现过的日期**就是**交易日，不需要额外日历 —— 也不会把法定假日
        误当成交易日（这正是题材归档踩过的坑）。
        """

        self._require()
        clause = f"where trade_date <= date '{_iso(end)}'" if end else ""
        rows = self.conn.execute(
            f"select distinct trade_date from '{self.path.as_posix()}' {clause} "
            "order by trade_date"
        ).fetchall()
        dates = [str(row[0])[:10] for row in rows]
        return dates[-int(limit) :] if limit else dates

    def latest_trade_date(self) -> str:
        dates = self.trade_dates(limit=1)
        if not dates:
            raise ValueError("行情归档为空")
        return dates[-1]

    def symbols(self) -> list[str]:
        """Return every symbol already present in the local archive."""

        self._require()
        frame = self.conn.execute(
            f"select distinct symbol from '{self.path.as_posix()}'"
        ).df()
        if frame.empty:
            return []
        return sorted(frame["symbol"].astype(str).str.zfill(6).unique().tolist())

    def latest_complete_trade_date(self, now: datetime | None = None) -> str:
        """返回最近一个适合生成推荐的**完整交易日**。

        行情源在交易时段可能已经出现当天的盘中日线，但题材、龙虎榜、两融等
        收盘后数据尚未形成。此时不能把当天盘中日期传给推荐和回补任务，否则
        收盘后数据会被显示成缺失。若本地最新日期是今天且尚未过 15:30，就退回
        到前一个交易日；如果没有前一个日期，只能暂时使用唯一已有日期。
        """

        dates = self.trade_dates()
        if not dates:
            raise ValueError("行情归档为空")
        latest = dates[-1]
        current = _shanghai_now(now)
        if latest == current.date().isoformat() and current.time() < A_SHARE_EOD_READY:
            return dates[-2] if len(dates) >= 2 else latest
        return latest

    def bars(
        self, start: str | None = None, end: str | None = None, window: int = DEFAULT_WINDOW
    ) -> pd.DataFrame:
        """取日线。默认取**最近 `window` 个交易日**。

        `start` 为 None 时按交易日起算（而不是按自然日回推），这样"最近 300 个交易日"
        是精确的，不会因为长假而少给历史。
        """

        self._require()
        if start is None:
            tail = self.trade_dates(end=end, limit=window)
            start = tail[0] if tail else None
        clauses = []
        if start:
            clauses.append(f"trade_date >= date '{_iso(start)}'")
        if end:
            clauses.append(f"trade_date <= date '{_iso(end)}'")
        where = f"where {' and '.join(clauses)}" if clauses else ""
        frame = self.conn.execute(
            f"select {', '.join(BAR_COLUMNS)} from '{self.path.as_posix()}' {where} "
            "order by symbol, trade_date"
        ).df()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        return frame

    def closes_on(self, dates: dict[str, str]) -> pd.DataFrame:
        """取若干基准日的收盘价，**取不晚于该日的最后一个交易日**。

        返回宽表：`symbol | <label1> | <label2> | ...`。

        为什么要"不晚于"：基准日当天可能停牌、或标的尚未上市。停牌时应取停牌前最后一个
        交易日的收盘（价格没变，语义正确）；尚未上市时该 symbol **不出现在结果里**，
        由调用方在 outer join 后表现为 NaN —— 这样"上市太晚、算不了 924 以来涨幅"
        是**显式缺失**，而不是被某个默认值悄悄填成 0%。
        """

        self._require()
        if not dates:
            return pd.DataFrame(columns=["symbol"])
        frames: list[pd.DataFrame] = []
        for label, day in dates.items():
            frames.append(
                self.conn.execute(
                    f"""
                    select symbol, close as "{label}"
                    from (
                        select symbol, close,
                               row_number() over (
                                   partition by symbol order by trade_date desc
                               ) as rn
                        from '{self.path.as_posix()}'
                        where trade_date <= date '{_iso(day)}'
                    )
                    where rn = 1
                    """
                ).df()
            )
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="symbol", how="outer")
        merged["symbol"] = merged["symbol"].astype(str).str.zfill(6)
        return merged

    def closes_on_date(self, trade_date: str) -> pd.DataFrame:
        """某个交易日的收盘价（`symbol / close`）。用于比对「昨收」以发现公司行动。"""

        self._require()
        frame = self.conn.execute(
            f"select symbol, close from '{self.path.as_posix()}' "
            f"where trade_date = date '{_iso(trade_date)}'"
        ).df()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        return frame

    def upsert_bars(self, frame: pd.DataFrame) -> int:
        """增量写入日线（同 `(symbol, trade_date)` 覆盖）。

        **不做部分覆盖**：先把新行涉及的键从旧文件里去掉，再整体 union 重写。
        这样"重跑同一天"是幂等的，而不会留下半新半旧的行。

        写临时文件再 `os.replace` —— duckdb 不能在读同一文件的同时写它，
        而且原子替换保证任务中途被杀不会把归档写坏。
        """

        if frame is None or frame.empty:
            return 0
        missing = [column for column in BAR_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"日线缺少列：{missing}")

        incoming = frame.loc[:, list(BAR_COLUMNS)].copy()
        incoming["symbol"] = incoming["symbol"].astype(str).str.zfill(6)
        incoming["trade_date"] = pd.to_datetime(incoming["trade_date"]).dt.normalize()
        incoming = incoming.drop_duplicates(subset=["symbol", "trade_date"], keep="last")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        projections = (
            "symbol, trade_date, open, high, low, close, pre_close, volume, amount"
        )
        self.conn.register("_incoming", incoming)
        try:
            if self.ready:
                self.conn.execute(
                    f"""
                    copy (
                        select cast(o.symbol as varchar) as symbol,
                               cast(o.trade_date as date) as trade_date,
                               cast(o.open as double) as open,
                               cast(o.high as double) as high,
                               cast(o.low as double) as low,
                               cast(o.close as double) as close,
                               cast(o.pre_close as double) as pre_close,
                               cast(o.volume as bigint) as volume,
                               cast(o.amount as double) as amount
                        from '{self.path.as_posix()}' as o
                        anti join _incoming as n
                            on cast(o.symbol as varchar) = n.symbol
                           and cast(o.trade_date as date) = n.trade_date
                        union all
                        select {projections} from _incoming
                    ) to '{temporary.as_posix()}' (format parquet, compression zstd)
                    """
                )
            else:
                self.conn.execute(
                    f"copy (select {projections} from _incoming) "
                    f"to '{temporary.as_posix()}' (format parquet, compression zstd)"
                )
        finally:
            self.conn.unregister("_incoming")
        os.replace(temporary, self.path)
        return len(incoming)

    def symbol_meta(self, source: str | Path) -> pd.DataFrame:
        """从 instrument_master 读取名称与上市日期（用于次新筛选、ST 剔除与结果可读性）。

        上市日期归一到 `YYYY-MM-DD`（上游是带时区的 timestamp）。`listed_days` 由调用方
        按 as-of 日期算 —— 不要在这里用"今天"，否则回看历史时会含未来信息。
        """

        source = Path(source)
        if not source.exists():
            return pd.DataFrame(columns=["symbol", "name", "listed_date"])
        frame = self.conn.execute(
            f"""
            select
                symbol,
                sec_name as name,
                strftime(cast(listed_date as timestamp), '%Y-%m-%d') as listed_date
            from read_parquet('{source.as_posix()}')
            """
        ).df()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        return frame.drop_duplicates(subset=["symbol"])

    def industry_meta(self, source: str | Path) -> pd.DataFrame:
        """读取项目内的个股行业快照。

        行业是可选的展示元数据，不应因为文件缺失或字段不完整而阻断推荐。
        同时兼容 CSV / Parquet，以及常见的中英文列名，方便把项目目录整体搬到
        另一台 Windows 电脑后直接使用已经落盘的快照。
        """

        source = Path(source)
        empty = pd.DataFrame(columns=["symbol", "industry"])
        if not source.exists():
            return empty
        try:
            if source.suffix.lower() == ".parquet":
                frame = self.conn.execute(
                    f"select * from read_parquet('{source.as_posix()}')"
                ).df()
            else:
                frame = pd.read_csv(source, dtype=str)
        except Exception:  # noqa: BLE001 - 可选展示数据，不能阻断推荐
            return empty
        if frame.empty:
            return empty

        columns = {str(column).strip().lower(): column for column in frame.columns}
        symbol_column = next(
            (
                columns[name]
                for name in ("symbol", "code", "股票代码", "证券代码")
                if name in columns
            ),
            None,
        )
        industry_column = next(
            (
                columns[name]
                for name in ("industry", "sector", "行业", "所属行业")
                if name in columns
            ),
            None,
        )
        if symbol_column is None or industry_column is None:
            return empty

        out = frame.loc[:, [symbol_column, industry_column]].copy()
        out.columns = ["symbol", "industry"]
        out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
        out["industry"] = out["industry"].astype("string").str.strip()
        out = out.loc[out["symbol"].notna() & out["industry"].notna()]
        out = out.loc[out["industry"] != ""]
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
        return out.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)

    def listed_days(self, meta: pd.DataFrame, as_of: str) -> pd.DataFrame:
        """按 as-of 日期算上市天数（自然日）。上市日缺失或晚于 as_of ⇒ NaN。"""

        frame = meta.copy()
        if frame.empty:
            frame["listed_days"] = pd.Series(dtype="float64")
            return frame
        listed = pd.to_datetime(frame["listed_date"], errors="coerce")
        frame["listed_days"] = (pd.Timestamp(_iso(as_of)) - listed).dt.days
        frame.loc[frame["listed_days"] < 0, "listed_days"] = pd.NA
        return frame


def default_base_dates(as_of: str, since_924: str = "2024-09-24") -> dict[str, str]:
    """构造基准日字典：`924` 与 `ytd`（上年最后一天）。

    `ytd` 取**上年 12-31**；`closes_on` 会退到不晚于该日的最后交易日 ⇒ 正好是上年最后一个
    交易日，与"年内涨幅"的常规口径一致。
    """

    day = datetime.strptime(_iso(as_of), "%Y-%m-%d").date()
    return {"base_924": since_924, "base_ytd": date(day.year - 1, 12, 31).isoformat()}
