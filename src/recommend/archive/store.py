"""归档存储层（duckdb）。

设计原则
--------
1. **只用 `insert or replace`，天然幂等**：回补可以随时重跑、可以断点续跑，
   同一天重复抓取不会产生重复行。
2. **每张表都带 `fetched_at`**：数据的"抓取时点"和"数据时点"必须分开记，
   否则无法判断某条记录是不是用后来的数据回填的（前视偏差的常见来源）。
3. **`archive_runs` 记录每次运行**：包括"当天没有数据"这种事也要落库。
   非交易日与接口失效**必须可区分**，否则归档覆盖率的判断会失真。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# 题材标签的分隔符：同花顺 reason 形如 "光通信测试+CPO+伽蓝特订单+中报增长"。
TAG_SEPARATORS = "+、,，/|"

SCHEMA: tuple[str, ...] = (
    """
    create table if not exists theme_attribution (
        trade_date    date      not null,
        symbol        varchar   not null,
        name          varchar,
        reason        varchar,
        close         double,
        change_pct    double,
        turnover_pct  double,
        amount        double,
        volume        double,
        big_order_net double,
        market        varchar,
        -- 上游返回的行级日期。必须等于 trade_date；不等就说明接口回退给了别的日子。
        source_date   varchar,
        fetched_at    timestamp,
        primary key (trade_date, symbol)
    )
    """,
    """
    create table if not exists theme_tag_daily (
        trade_date  date    not null,
        tag         varchar not null,
        stock_count integer,
        primary key (trade_date, tag)
    )
    """,
    """
    create table if not exists dragon_tiger (
        trade_date   date    not null,
        symbol       varchar not null,
        name         varchar,
        explanation  varchar,
        close        double,
        change_pct   double,
        net_buy      double,
        buy_amt      double,
        sell_amt     double,
        turnover_pct double,
        fetched_at   timestamp,
        primary key (trade_date, symbol)
    )
    """,
    """
    create table if not exists margin_trading (
        trade_date date    not null,
        symbol     varchar not null,
        rzye       double,
        rzmre      double,
        rzche      double,
        rqye       double,
        rqmcl      double,
        rqchl      double,
        rzrqye     double,
        fetched_at timestamp,
        primary key (trade_date, symbol)
    )
    """,
    """
    create table if not exists sector_snapshot (
        trade_date      date    not null,
        sector_code     varchar not null,
        sector_name     varchar,
        change_pct      double,
        main_net_inflow double,
        up_count        integer,
        down_count      integer,
        fetched_at      timestamp,
        primary key (trade_date, sector_code)
    )
    """,
    """
    create table if not exists valuation_daily (
        trade_date    date      not null,
        symbol        varchar   not null,
        close         double,
        pe_ttm        double,
        pb_mrq        double,
        ps_ttm        double,
        pcf_ncf_ttm   double,
        turnover_rate double,
        trade_status  varchar,
        source        varchar   not null,
        fetched_at    timestamp,
        primary key (trade_date, symbol)
    )
    """,
    """
    create table if not exists financial_quarterly (
        symbol                varchar   not null,
        pub_date              date      not null,
        stat_date             date      not null,
        roe_avg               double,
        net_profit_margin     double,
        gross_profit_margin   double,
        net_profit            double,
        eps_ttm               double,
        revenue               double,
        total_share           double,
        liqa_share            double,
        yoy_equity            double,
        yoy_asset             double,
        yoy_net_income        double,
        yoy_eps_basic         double,
        yoy_profit_net_income double,
        current_ratio         double,
        quick_ratio           double,
        cash_ratio            double,
        yoy_liability         double,
        liability_to_asset    double,
        asset_to_equity       double,
        ebit_to_interest      double,
        cfo_to_or             double,
        cfo_to_np             double,
        cfo_to_gr             double,
        source                varchar   not null,
        fetched_at            timestamp,
        primary key (symbol, stat_date)
    )
    """,
    """
    create table if not exists archive_runs (
        run_id       varchar primary key,
        sync_run_id  varchar,
        task         varchar,
        target       varchar,
        started_at   timestamp,
        finished_at  timestamp,
        rows_written integer,
        status       varchar,
        detail       varchar
    )
    """,
    """
    create table if not exists company_announcements (
        exchange               varchar not null,
        announcement_id        varchar not null,
        symbol                 varchar,
        company_name           varchar,
        announcement_date      date not null,
        title                  varchar not null,
        url                    varchar,
        category               varchar,
        signal                 varchar,
        summary                varchar,
        classification_status  varchar not null,
        classification_source  varchar,
        confidence             double,
        fetched_at             timestamp,
        classified_at          timestamp,
        sync_run_id            varchar,
        primary key (exchange, announcement_id)
    )
    """,
    """
    create table if not exists announcement_sync_state (
        exchange          varchar primary key,
        last_success_date date not null,
        updated_at        timestamp,
        sync_run_id       varchar
    )
    """,
)

# 旧版项目已经可能存在 financial_quarterly；`create table if not exists` 不会给旧表
# 补列，所以启动时做幂等迁移。所有新增列都允许为空，旧归档可以继续参与推荐。
FINANCIAL_MIGRATIONS: tuple[str, ...] = (
    "alter table financial_quarterly add column if not exists current_ratio double",
    "alter table financial_quarterly add column if not exists quick_ratio double",
    "alter table financial_quarterly add column if not exists cash_ratio double",
    "alter table financial_quarterly add column if not exists yoy_liability double",
    "alter table financial_quarterly add column if not exists liability_to_asset double",
    "alter table financial_quarterly add column if not exists asset_to_equity double",
    "alter table financial_quarterly add column if not exists ebit_to_interest double",
    "alter table financial_quarterly add column if not exists cfo_to_or double",
    "alter table financial_quarterly add column if not exists cfo_to_np double",
    "alter table financial_quarterly add column if not exists cfo_to_gr double",
    "alter table archive_runs add column if not exists sync_run_id varchar",
)

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "theme_attribution": ("trade_date", "symbol"),
    "theme_tag_daily": ("trade_date", "tag"),
    "dragon_tiger": ("trade_date", "symbol"),
    "margin_trading": ("trade_date", "symbol"),
    "sector_snapshot": ("trade_date", "sector_code"),
    "valuation_daily": ("trade_date", "symbol"),
    "financial_quarterly": ("symbol", "stat_date"),
    "company_announcements": ("exchange", "announcement_id"),
    "announcement_sync_state": ("exchange",),
}


def split_tags(reason: str | None) -> list[str]:
    """把 `reason` 拆成题材标签。

    保留全部标签（包括"中报增长"这类基本面标签）——**过滤是下游的策略决定**，
    归档层不做主观筛选，否则将来想改口径就得重新抓一遍历史。
    """

    if not reason:
        return []
    cleaned = str(reason)
    for separator in TAG_SEPARATORS:
        cleaned = cleaned.replace(separator, "\x00")
    tags = [part.strip() for part in cleaned.split("\x00")]
    seen: dict[str, None] = {}
    for tag in tags:
        if tag:
            seen.setdefault(tag, None)
    return list(seen)


class ArchiveStore:
    """归档库读写。所有写操作幂等。"""

    def __init__(self, path: str | Path = "data/archive.duckdb") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self.ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> ArchiveStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        for statement in SCHEMA:
            self.conn.execute(statement)
        for statement in FINANCIAL_MIGRATIONS:
            self.conn.execute(statement)

    # -- 写 ------------------------------------------------------------------ #

    def upsert(
        self, table: str, frame: pd.DataFrame, columns: tuple[str, ...]
    ) -> int:
        """幂等写入。`frame` 缺列会报错而不是静默补空。"""

        if frame is None or frame.empty:
            return 0
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{table} 缺少列：{missing}")
        payload = frame.loc[:, list(columns)].copy()
        keys = PRIMARY_KEYS.get(table)
        if keys:
            missing_keys = [key for key in keys if key not in payload.columns]
            if missing_keys:
                raise ValueError(f"{table} 缺少主键列：{missing_keys}")
            payload = payload.drop_duplicates(subset=list(keys), keep="last")
        self.conn.register("_incoming", payload)
        try:
            column_list = ", ".join(columns)
            self.conn.execute(
                f"insert or replace into {table} ({column_list}) "
                f"select {column_list} from _incoming"
            )
        finally:
            self.conn.unregister("_incoming")
        return len(payload)

    def insert_new_announcements(self, frame: pd.DataFrame) -> int:
        """只插入新公告，避免重叠同步窗口覆盖已有的分类结果。"""

        columns = (
            "exchange",
            "announcement_id",
            "symbol",
            "company_name",
            "announcement_date",
            "title",
            "url",
            "category",
            "signal",
            "summary",
            "classification_status",
            "classification_source",
            "confidence",
            "fetched_at",
            "classified_at",
            "sync_run_id",
        )
        if frame is None or frame.empty:
            return 0
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"company_announcements 缺少列：{missing}")
        payload = frame.loc[:, list(columns)].drop_duplicates(
            subset=["exchange", "announcement_id"], keep="last"
        )
        self.conn.register("_incoming_announcements", payload)
        try:
            column_list = ", ".join(columns)
            self.conn.execute(
                f"insert or ignore into company_announcements ({column_list}) "
                f"select {column_list} from _incoming_announcements"
            )
        finally:
            self.conn.unregister("_incoming_announcements")
        return len(payload)

    def write_themes(self, frame: pd.DataFrame, trade_date: str) -> int:
        """写入题材归因，并重建该日的标签聚合。

        ⚠️ 这里用**整日替换**（先删该日再插），而不是逐键 upsert。原因是幂等语义：

        * 题材归因是「**当日强势股名单**」——一天的数据是一个**整体快照**。
          上游若修正了名单（变短），逐键 upsert 会让旧行残留，库里于是存在
          "上游已经不认了的股票"，而标签计数会被这些残留行污染，
          直接影响题材生命周期（启动/发酵/高潮/退潮）的阶段判定。
        * **整日替换为什么是安全的**：同花顺 `getharden` 是**一次返回全量**的接口，
          没有分页 ⇒ "抓取成功"就等于"拿到了这一天的完整名单"，
          不存在"抓到一半"的中间态。
        * 这只在**调用方确认抓取非空**之后发生（见 `backfill.run_single`），
          所以空返回不会清空已有数据。

        ⚠️ 反例（不要照抄到别的表）：龙虎榜有 `page_size` 分页上限，
        **部分抓取是可能的**，所以那边只能用逐键 upsert —— **宁可残留，不可误删**。
        """

        fetched_at = datetime.now(UTC).replace(tzinfo=None)
        payload = frame.copy()
        payload["trade_date"] = pd.to_datetime(trade_date).date()
        payload["fetched_at"] = fetched_at
        try:
            self.conn.execute("begin transaction")
            self.conn.execute(
                "delete from theme_attribution where trade_date = ?",
                [pd.to_datetime(trade_date).date()],
            )
            written = self.upsert(
                "theme_attribution",
                payload,
                (
                    "trade_date",
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
                    "fetched_at",
                ),
            )
            self.rebuild_theme_tags(trade_date)
            self.conn.execute("commit")
            return written
        except Exception:
            self.conn.execute("rollback")
            raise

    def purge_date(self, table: str, trade_date: str) -> int:
        """删除某张表某一天的全部记录（用于清除已确认被污染的日子）。"""

        if table not in {"theme_attribution", "dragon_tiger", "margin_trading", "sector_snapshot"}:
            raise ValueError(f"不支持按日清除的表：{table}")
        before = self.conn.execute(
            f"select count(*) from {table} where trade_date = ?",
            [pd.to_datetime(trade_date).date()],
        ).fetchone()[0]
        self.conn.execute(
            f"delete from {table} where trade_date = ?", [pd.to_datetime(trade_date).date()]
        )
        if table == "theme_attribution":
            self.conn.execute(
                "delete from theme_tag_daily where trade_date = ?",
                [pd.to_datetime(trade_date).date()],
            )
        return int(before)

    def theme_quality(self) -> pd.DataFrame:
        """按日检查题材归档的质量信号（用来自查有没有混进"回退快照"）。

        一个**真实交易日**的响应必须带行情字段。实测 496 个真实交易日
        **100% 有 `change_pct`**（42218/42218 行），而回退响应**一行都没有**
        （3081 行全为 NULL）。⇒「无行情则判为回退」这条规则在本数据集上**零误报**。
        """

        return self.conn.execute(
            """
            select
                trade_date,
                count(*)             as rows,
                count(change_pct)    as rows_with_price,
                min(source_date)     as source_date
            from theme_attribution
            group by trade_date
            having count(change_pct) = 0
                or min(source_date) is null
                or min(source_date) <> strftime(min(trade_date), '%Y-%m-%d')
            order by trade_date
            """
        ).df()

    def rebuild_theme_tags(self, trade_date: str) -> int:
        """按日重建题材标签计数（状态机的输入）。"""

        count = self.conn.execute(
            "select count(*) from theme_attribution where trade_date = ?",
            [pd.to_datetime(trade_date).date()],
        ).fetchone()[0]
        self.conn.execute("delete from theme_tag_daily where trade_date = ?", [
            pd.to_datetime(trade_date).date()
        ])
        if not count:
            return 0
        reasons = self.conn.execute(
            "select symbol, reason from theme_attribution where trade_date = ?",
            [pd.to_datetime(trade_date).date()],
        ).fetchall()
        tally: dict[str, int] = {}
        for _symbol, reason in reasons:
            for tag in split_tags(reason):
                tally[tag] = tally.get(tag, 0) + 1
        if tally:
            self.conn.executemany(
                "insert or replace into theme_tag_daily (trade_date, tag, stock_count) "
                "values (?, ?, ?)",
                [
                    (pd.to_datetime(trade_date).date(), tag, count_)
                    for tag, count_ in tally.items()
                ],
            )
        return len(tally)

    def record_run(
        self,
        task: str,
        target: str,
        started_at: datetime,
        rows_written: int,
        status: str,
        detail: str = "",
        sync_run_id: str | None = None,
    ) -> str:
        """记录一次运行。`status` 取 OK / EMPTY / REJECTED / FAILED。

        `EMPTY` 必须与 `FAILED` 分开记：非交易日返回空是**正常**的，
        而接口失效返回空是**故障**。混在一起会让覆盖率统计失去意义。
        """

        run_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "insert into archive_runs "
            "(run_id, sync_run_id, task, target, started_at, finished_at, "
            "rows_written, status, detail) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                sync_run_id,
                task,
                target,
                started_at.replace(tzinfo=None),
                datetime.now(UTC).replace(tzinfo=None),
                int(rows_written),
                status,
                detail[:500],
            ],
        )
        return run_id

    # -- 读 ------------------------------------------------------------------ #

    def coverage(self) -> pd.DataFrame:
        """各表的实际覆盖区间与行数（判断归档进度用）。"""

        queries = {
            "theme_attribution": "select count(*) n, count(distinct trade_date) d, "
            "min(trade_date) d0, max(trade_date) d1 from theme_attribution",
            "theme_tag_daily": "select count(*) n, count(distinct trade_date) d, "
            "min(trade_date) d0, max(trade_date) d1 from theme_tag_daily",
            "dragon_tiger": "select count(*) n, count(distinct trade_date) d, "
            "min(trade_date) d0, max(trade_date) d1 from dragon_tiger",
            "margin_trading": "select count(*) n, count(distinct trade_date) d, "
            "min(trade_date) d0, max(trade_date) d1 from margin_trading",
            "sector_snapshot": "select count(*) n, count(distinct trade_date) d, "
            "min(trade_date) d0, max(trade_date) d1 from sector_snapshot",
            "valuation_daily": "select count(*) n, count(distinct trade_date) d, "
            "min(trade_date) d0, max(trade_date) d1 from valuation_daily",
            "financial_quarterly": "select count(*) n, count(distinct stat_date) d, "
            "min(stat_date) d0, max(stat_date) d1 from financial_quarterly",
        }
        rows: list[dict[str, Any]] = []
        for table, query in queries.items():
            record = self.conn.execute(query).fetchone()
            rows.append(
                {
                    "table": table,
                    "rows": int(record[0] or 0),
                    "days": int(record[1] or 0),
                    "first": record[2],
                    "last": record[3],
                }
            )
        return pd.DataFrame(rows)

    def archived_dates(self, table: str) -> set[str]:
        """已经成功归档（且非空）的交易日集合 —— 断点续跑的依据。"""

        rows = self.conn.execute(
            f"select distinct trade_date from {table} order by trade_date"
        ).fetchall()
        return {str(row[0]) for row in rows}

    # -- 基本面 ------------------------------------------------------------ #

    def valuation_dates(self) -> set[str]:
        """已经写入过的估值快照日期。"""

        rows = self.conn.execute(
            "select distinct trade_date from valuation_daily order by trade_date"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def valuation_asof(self, as_of: str) -> pd.DataFrame:
        """取不晚于 ``as_of`` 的每只股票最近估值快照。"""

        return self.conn.execute(
            """
            select * exclude (rn)
            from (
                select *, row_number() over (
                    partition by symbol order by trade_date desc, fetched_at desc
                ) as rn
                from valuation_daily
                where trade_date <= ?
            )
            where rn = 1
            """,
            [pd.to_datetime(as_of).date()],
        ).df()

    def financial_period_coverage(self, stat_date: str, as_of: str) -> int:
        """某财报期在公告日已到达 ``as_of`` 的股票数量。"""

        return int(
            self.conn.execute(
                """
                select count(distinct symbol)
                from financial_quarterly
                where stat_date = ? and pub_date <= ?
                """,
                [pd.to_datetime(stat_date).date(), pd.to_datetime(as_of).date()],
            ).fetchone()[0]
        )

    def financial_asof(self, as_of: str) -> pd.DataFrame:
        """按公告日截断，取每只股票最近一份已披露财报，避免前视偏差。"""

        return self.conn.execute(
            """
            select * exclude (rn)
            from (
                select *, row_number() over (
                    partition by symbol order by pub_date desc, stat_date desc, fetched_at desc
                ) as rn
                from financial_quarterly
                where pub_date <= ?
            )
            where rn = 1
            """,
            [pd.to_datetime(as_of).date()],
        ).df()

    def financial_history_asof(self, as_of: str, periods: int = 5) -> pd.DataFrame:
        """按公告日截断，取每只股票最近若干份已披露财报。

        多期数据用于计算营收同比、利润增速加速度和连续增长；排序仍只使用公告日
        不晚于 ``as_of`` 的记录，不能因为财报期末日更早就绕过前视检查。
        """

        periods = max(1, int(periods))
        return self.conn.execute(
            """
            select * exclude (rn)
            from (
                select *, row_number() over (
                    partition by symbol order by stat_date desc, pub_date desc, fetched_at desc
                ) as rn
                from financial_quarterly
                where pub_date <= ?
            )
            where rn <= ?
            order by symbol, stat_date desc
            """,
            [pd.to_datetime(as_of).date(), periods],
        ).df()

    def run_log(self, limit: int = 20) -> pd.DataFrame:
        return self.conn.execute(
            "select sync_run_id, task, target, rows_written, status, detail, "
            "finished_at as finished_at_utc, "
            "finished_at + interval '8 hours' as finished_at_shanghai "
            "from archive_runs order by finished_at desc limit ?",
            [limit],
        ).df()

    def latest_run_status(self, task: str) -> dict[str, tuple[str, int]]:
        """Return the most recent status and row count for each target."""

        rows = self.conn.execute(
            """
            select target, status, rows_written
            from (
                select target, status, rows_written,
                       row_number() over (
                           partition by target order by finished_at desc, run_id desc
                       ) as rn
                from archive_runs
                where task = ?
            )
            where rn = 1
            """,
            [task],
        ).fetchall()
        return {
            str(target): (str(status), int(rows_written or 0))
            for target, status, rows_written in rows
        }

    def retryable_dates(
        self,
        table: str,
        trade_dates: list[str],
        *,
        task: str,
        force: bool = False,
        recent_empty_days: int = 5,
    ) -> list[str]:
        """Find missing dates without repeatedly hammering old empty dates.

        Dates with no successful rows are normally retryable.  An old ``EMPTY``
        response, however, usually means a non-trading day or a source whose
        historical window does not reach that far.  Only the most recent few
        empty dates are retried; FAILED/REJECTED dates remain retryable.
        """

        if force:
            return list(trade_dates)
        done = self.archived_dates(table)
        statuses = self.latest_run_status(task)
        recent = set(trade_dates[-max(1, int(recent_empty_days)) :])
        pending: list[str] = []
        for day in trade_dates:
            if day in done:
                continue
            status, rows = statuses.get(day, ("", 0))
            if not status or status in {"FAILED", "REJECTED"} or (status == "OK" and rows <= 0):
                pending.append(day)
            elif status == "EMPTY" and day in recent:
                pending.append(day)
        return pending

    def top_tags(self, trade_date: str, limit: int = 20) -> pd.DataFrame:
        return self.conn.execute(
            "select tag, stock_count from theme_tag_daily where trade_date = ? "
            "order by stock_count desc limit ?",
            [pd.to_datetime(trade_date).date(), limit],
        ).df()

    def themes_on(self, trade_date: str) -> pd.DataFrame:
        """某日的题材归因全量（当日强势股名单）。"""

        return self.conn.execute(
            """
            select trade_date, symbol, name, reason, close, change_pct,
                   turnover_pct, amount, volume, big_order_net, market, source_date
            from theme_attribution
            where trade_date = ?
            """,
            [pd.to_datetime(trade_date).date()],
        ).df()

    def tag_counts_on(self, trade_date: str) -> pd.DataFrame:
        """某日的题材标签计数（`theme_tag_daily`）。"""

        return self.conn.execute(
            "select tag, stock_count from theme_tag_daily where trade_date = ?",
            [pd.to_datetime(trade_date).date()],
        ).df()

    def theme_history(
        self, start: str, end: str, symbols: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """区间内的题材归因「上榜」记录，只取 symbol / trade_date。

        用于算 `theme_days_20`（近 N 个交易日上榜几天）。`symbols` 给定时只取这些标的
        —— 归档里一天有几十到几百只，全量拉进来算完再筛是浪费。
        """

        clause = ""
        params: list[Any] = [pd.to_datetime(start).date(), pd.to_datetime(end).date()]
        if symbols is not None:
            placeholders = ", ".join("?" for _ in symbols)
            clause = f" and symbol in ({placeholders})"
            params.extend(list(symbols))
        return self.conn.execute(
            "select distinct symbol, trade_date from theme_attribution "
            f"where trade_date between ? and ?{clause}",
            params,
        ).df()

    def dragon_tiger_between(self, start: str, end: str) -> pd.DataFrame:
        return self.conn.execute(
            "select trade_date, symbol, name, net_buy, buy_amt, sell_amt, "
            "change_pct, turnover_pct, explanation "
            "from dragon_tiger where trade_date between ? and ?",
            [pd.to_datetime(start).date(), pd.to_datetime(end).date()],
        ).df()

    def margin_between(self, start: str, end: str) -> pd.DataFrame:
        return self.conn.execute(
            "select trade_date, symbol, rzye, rzmre, rzche, rqye, rzrqye "
            "from margin_trading where trade_date between ? and ?",
            [pd.to_datetime(start).date(), pd.to_datetime(end).date()],
        ).df()

    def company_announcements_between(
        self, start: str, end: str, symbols: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """已分类公告事件；供报告提示使用，不进入任何因子或排序。"""

        clause = ""
        params: list[Any] = [pd.to_datetime(start).date(), pd.to_datetime(end).date()]
        if symbols is not None:
            if not symbols:
                return pd.DataFrame()
            placeholders = ", ".join("?" for _ in symbols)
            clause = f" and symbol in ({placeholders})"
            params.extend(list(symbols))
        return self.conn.execute(
            "select exchange, announcement_id, symbol, company_name, announcement_date, "
            "title, url, category, signal, summary, classification_status "
            "from company_announcements where announcement_date between ? and ? "
            "and classification_status = 'classified' "
            f"{clause} order by announcement_date desc, fetched_at desc",
            params,
        ).df()

    def pending_company_announcements(self, limit: int = 25) -> pd.DataFrame:
        """按日期升序返回待 Codex 分类条目，优先清理最早积压。"""

        return self.conn.execute(
            "select exchange, announcement_id, symbol, company_name, announcement_date, "
            "title, url from company_announcements "
            "where classification_status = 'pending' "
            "order by announcement_date, exchange, announcement_id limit ?",
            [max(1, int(limit))],
        ).df()

    def table_row_count(self, table: str) -> int:
        """某表总行数（用于区分「归档为空」与「该标的没上榜」）。"""

        if table not in {
            "theme_attribution",
            "dragon_tiger",
            "margin_trading",
            "sector_snapshot",
            "valuation_daily",
            "financial_quarterly",
            "company_announcements",
        }:
            raise ValueError(f"未知表：{table}")
        return int(self.conn.execute(f"select count(*) from {table}").fetchone()[0])
