"""免费基本面数据同步（BaoStock）与本地归档适配。

这里不把 BaoStock 的结果直接交给筛选器。同步阶段先把估值快照和财报写入本项目
自己的 DuckDB；筛选阶段再按公告日截断读取，避免运行时依赖外部目录，也避免把未来
才披露的财务数据带进历史口径。

BaoStock 0.9.3 的 ``ResultData.get_data`` 仍调用了 pandas 已移除的 ``DataFrame.append``。
因此本模块故意用 ``next/get_row_data`` 读取结果，兼容当前项目的 pandas 3.x。
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import re
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

import pandas as pd

from .archive.store import ArchiveStore

BAOSTOCK_SOURCE = "baostock"
SYMBOL_PATTERN = re.compile(r"(\d{6})")

VALUATION_COLUMNS = (
    "trade_date",
    "symbol",
    "close",
    "pe_ttm",
    "pb_mrq",
    "ps_ttm",
    "pcf_ncf_ttm",
    "turnover_rate",
    "trade_status",
    "source",
    "fetched_at",
)

FINANCIAL_COLUMNS = (
    "symbol",
    "pub_date",
    "stat_date",
    "roe_avg",
    "net_profit_margin",
    "gross_profit_margin",
    "net_profit",
    "eps_ttm",
    "revenue",
    "total_share",
    "liqa_share",
    "yoy_equity",
    "yoy_asset",
    "yoy_net_income",
    "yoy_eps_basic",
    "yoy_profit_net_income",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "yoy_liability",
    "liability_to_asset",
    "asset_to_equity",
    "ebit_to_interest",
    "cfo_to_or",
    "cfo_to_np",
    "cfo_to_gr",
    "source",
    "fetched_at",
)

Progress = Callable[[int, int, str], None]

_WORKER_BAOSTOCK: Any | None = None
_WORKER_ERROR = ""


@dataclass
class FundamentalSyncResult:
    """一次 BaoStock 同步的可读统计。"""

    task: str = "baostock-fundamentals"
    valuation_attempted: int = 0
    valuation_ok: int = 0
    valuation_empty: int = 0
    valuation_rows: int = 0
    financial_attempted: int = 0
    financial_ok: int = 0
    financial_empty: int = 0
    financial_rows: int = 0
    skipped_valuation: bool = False
    skipped_financial: bool = False
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def attempted(self) -> int:
        return self.valuation_attempted + self.financial_attempted

    @property
    def rows(self) -> int:
        return self.valuation_rows + self.financial_rows

    @property
    def warning(self) -> bool:
        valuation_missing = not self.skipped_valuation and self.valuation_rows == 0
        financial_missing = not self.skipped_financial and self.financial_rows == 0
        return bool(self.failures) or valuation_missing or financial_missing

    def summary(self) -> str:
        valuation = (
            "已覆盖，跳过"
            if self.skipped_valuation
            else f"估值快照 {self.valuation_attempted}，写入 {self.valuation_rows} 行"
        )
        financial = (
            "已覆盖，跳过"
            if self.skipped_financial
            else f"财务股票 {self.financial_attempted}，归档 {self.financial_rows} 行"
        )
        text = f"BaoStock：{valuation}；{financial}"
        if self.failures:
            text += f"；失败 {len(self.failures)}（样例：{self.failures[:2]}）"
        return text


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    """把本地代码、交易所代码统一为六位 A 股代码。"""

    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        match = SYMBOL_PATTERN.search(str(raw))
        if not match:
            continue
        symbol = match.group(1)
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def baostock_code(symbol: str) -> str:
    """BaoStock 代码格式。当前项目的股票池是沪深 A 股，不含北交所。"""

    symbol = normalize_symbols([symbol])[0]
    return f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"


def _load_baostock() -> Any:
    try:
        import baostock as bs
    except ImportError as error:  # pragma: no cover - 依赖缺失时由同步层显示
        raise RuntimeError(
            "未安装 BaoStock。请运行 .venv\\Scripts\\python.exe -m pip install -e "
            "\".[web,free-data]\""
        ) from error
    return bs


def _result_frame(result: Any) -> pd.DataFrame:
    """兼容 pandas 3 的 BaoStock 结果读取。"""

    if str(getattr(result, "error_code", "")) != "0":
        raise RuntimeError(
            f"BaoStock 返回错误 {getattr(result, 'error_code', '')}: "
            f"{getattr(result, 'error_msg', '')}"
        )
    rows: list[list[object]] = []
    while result.next():
        row = result.get_row_data()
        if row:
            rows.append(row)
    return pd.DataFrame(rows, columns=list(getattr(result, "fields", [])))


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _asof_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


def _latest_quarter(as_of: str) -> tuple[int, int, str]:
    """返回按日历推断的最近财报期（年、季度、期末日）。"""

    day = _asof_date(as_of)
    if day.month <= 3:
        return day.year - 1, 4, f"{day.year - 1}-12-31"
    if day.month <= 6:
        return day.year, 1, f"{day.year}-03-31"
    if day.month <= 9:
        return day.year, 2, f"{day.year}-06-30"
    return day.year, 3, f"{day.year}-09-30"


def _previous_quarter(year: int, quarter: int) -> tuple[int, int]:
    return (year - 1, 4) if quarter == 1 else (year, quarter - 1)


def _period_candidates(as_of: str, limit: int = 4) -> list[tuple[int, int]]:
    year, quarter, _ = _latest_quarter(as_of)
    periods: list[tuple[int, int]] = []
    for _ in range(limit):
        periods.append((year, quarter))
        year, quarter = _previous_quarter(year, quarter)
    return periods


def _period_stat_date(year: int, quarter: int) -> str:
    """把 BaoStock 的年/季参数映射为财报期末日。"""

    return {
        1: f"{year}-03-31",
        2: f"{year}-06-30",
        3: f"{year}-09-30",
        4: f"{year}-12-31",
    }[quarter]


def _query_profit(bs: Any, code: str, year: int, quarter: int) -> pd.DataFrame:
    return _result_frame(bs.query_profit_data(code, year=year, quarter=quarter))


def _query_growth(bs: Any, code: str, year: int, quarter: int) -> pd.DataFrame:
    return _result_frame(bs.query_growth_data(code, year=year, quarter=quarter))


def _query_balance(bs: Any, code: str, year: int, quarter: int) -> pd.DataFrame:
    return _result_frame(bs.query_balance_data(code, year=year, quarter=quarter))


def _query_cash_flow(bs: Any, code: str, year: int, quarter: int) -> pd.DataFrame:
    return _result_frame(bs.query_cash_flow_data(code, year=year, quarter=quarter))


def _init_financial_worker() -> None:
    """为一个独立进程建立 BaoStock 连接。

    BaoStock 把连接保存在模块级全局对象里，不能在线程之间安全复用；Windows 下用
    spawn 启动少量独立进程，每个进程只登录一次，既避免线程串包，也避免每只股票
    重复登录。
    """

    global _WORKER_BAOSTOCK, _WORKER_ERROR
    try:
        _WORKER_BAOSTOCK = _load_baostock()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            login = _WORKER_BAOSTOCK.login()
        if str(getattr(login, "error_code", "")) != "0":
            _WORKER_ERROR = (
                f"登录失败 {getattr(login, 'error_code', '')}: "
                f"{getattr(login, 'error_msg', '')}"
            )
            _WORKER_BAOSTOCK = None
    except Exception as error:  # noqa: BLE001 - 由主进程汇总显示
        _WORKER_ERROR = f"{type(error).__name__}: {error}"
        _WORKER_BAOSTOCK = None


def _fetch_financial_worker(
    payload: tuple[str, str, list[tuple[int, int]]],
) -> tuple[str, pd.DataFrame, str]:
    """一个 BaoStock 进程处理一只股票的多期财报和当前财务健康数据。"""

    symbol, as_of, periods = payload
    if _WORKER_BAOSTOCK is None:
        return symbol, pd.DataFrame(columns=list(FINANCIAL_COLUMNS)), _WORKER_ERROR or "连接未建立"
    try:
        frames: list[pd.DataFrame] = []
        latest_found = False
        for candidate_year, candidate_quarter in periods:
            profit = pd.DataFrame()
            growth = pd.DataFrame()
            balance = pd.DataFrame()
            cash_flow = pd.DataFrame()
            try:
                profit = _query_profit(
                    _WORKER_BAOSTOCK, baostock_code(symbol), candidate_year, candidate_quarter
                )
            except Exception:  # noqa: BLE001 - 单期失败不应丢掉其它报告期
                continue
            if profit.empty:
                continue
            try:
                growth = _query_growth(
                    _WORKER_BAOSTOCK, baostock_code(symbol), candidate_year, candidate_quarter
                )
            except Exception:  # noqa: BLE001 - 增长接口失败仍保留利润/质量字段
                growth = pd.DataFrame()
            if not latest_found:
                # 偿债/现金流只取最近一期，避免把一次同步放大成四倍请求；多期
                # 盈利/成长数据足够计算营收同比、增速加速度和连续增长。
                try:
                    balance = _query_balance(
                        _WORKER_BAOSTOCK, baostock_code(symbol), candidate_year, candidate_quarter
                    )
                except Exception:  # noqa: BLE001 - 健康字段缺失不丢掉盈利数据
                    balance = pd.DataFrame()
                try:
                    cash_flow = _query_cash_flow(
                        _WORKER_BAOSTOCK, baostock_code(symbol), candidate_year, candidate_quarter
                    )
                except Exception:  # noqa: BLE001 - 健康字段缺失不丢掉盈利数据
                    cash_flow = pd.DataFrame()
                latest_found = True
            frames.append(
                _prepare_financial(
                    profit,
                    growth,
                    balance,
                    cash_flow,
                    as_of,
                    dt.datetime.now(dt.UTC).replace(tzinfo=None),
                )
            )
        if not frames:
            return symbol, pd.DataFrame(columns=list(FINANCIAL_COLUMNS)), ""
        return symbol, pd.concat(frames, ignore_index=True), ""
    except Exception as error:  # noqa: BLE001 - 主进程继续收集其它股票
        return (
            symbol,
            pd.DataFrame(columns=list(FINANCIAL_COLUMNS)),
            f"{type(error).__name__}: {error}",
        )


def _prepare_valuation(frame: pd.DataFrame, as_of: str, fetched_at: dt.datetime) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(VALUATION_COLUMNS))
    out = frame.copy()
    if "code" not in out.columns:
        return pd.DataFrame(columns=list(VALUATION_COLUMNS))
    out["symbol"] = out["code"].astype(str).str.extract(SYMBOL_PATTERN, expand=False)
    out["trade_date"] = out.get("date", as_of).astype(str).str[:10]
    out = out.loc[out["symbol"].notna() & (out["trade_date"] == str(as_of)[:10])].copy()
    out = _numeric(
        out,
        ("close", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "turn"),
    )
    out = out.rename(
        columns={
            "peTTM": "pe_ttm",
            "pbMRQ": "pb_mrq",
            "psTTM": "ps_ttm",
            "pcfNcfTTM": "pcf_ncf_ttm",
            "turn": "turnover_rate",
            "tradestatus": "trade_status",
        }
    )
    out["source"] = BAOSTOCK_SOURCE
    out["fetched_at"] = fetched_at
    for column in VALUATION_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, list(VALUATION_COLUMNS)].drop_duplicates(
        subset=["trade_date", "symbol"], keep="last"
    )


def _prepare_derived_valuation(
    financial: pd.DataFrame,
    close_prices: Mapping[str, float],
    as_of: str,
    fetched_at: dt.datetime,
) -> pd.DataFrame:
    """用本地收盘价和 BaoStock 的 EPS(TTM) 构造当前估值快照。

    BaoStock 的估值历史接口需要对每只股票单独请求，5000 多只股票会让启动同步
    变得非常慢。PE(TTM) 的定义本身就是 ``收盘价 / EPS(TTM)``，因此收盘价直接
    使用项目已经落盘的行情，EPS(TTM) 使用同一份 BaoStock 财报数据即可，既保持
    口径透明，也不再重复请求一遍逐股历史行情。PB/PS/PCF 等字段暂时留空，避免
    用不兼容的替代值冒充官方口径。
    """

    if financial.empty or not close_prices:
        return pd.DataFrame(columns=list(VALUATION_COLUMNS))
    out = financial.copy()
    out["symbol"] = out["symbol"].astype(str).str.extract(SYMBOL_PATTERN, expand=False)
    out = out.loc[out["symbol"].notna()].copy()
    normalized_prices: dict[str, float] = {}
    for raw_symbol, value in close_prices.items():
        normalized = normalize_symbols([raw_symbol])
        if normalized:
            normalized_prices[normalized[0]] = value
    out["close"] = out["symbol"].map(normalized_prices)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["eps_ttm"] = pd.to_numeric(out.get("eps_ttm"), errors="coerce")
    positive_eps = out["eps_ttm"].where(out["eps_ttm"] > 0)
    out["pe_ttm"] = out["close"] / positive_eps
    out["trade_date"] = str(as_of)[:10]
    out["pb_mrq"] = pd.NA
    out["ps_ttm"] = pd.NA
    out["pcf_ncf_ttm"] = pd.NA
    out["turnover_rate"] = pd.NA
    out["trade_status"] = pd.NA
    out["source"] = f"{BAOSTOCK_SOURCE}:eps_ttm"
    out["fetched_at"] = fetched_at
    out = out.loc[out["close"].notna()].copy()
    for column in VALUATION_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, list(VALUATION_COLUMNS)].drop_duplicates(
        subset=["trade_date", "symbol"], keep="last"
    )


def _prepare_financial(
    profit: pd.DataFrame,
    growth: pd.DataFrame,
    balance: pd.DataFrame,
    cash_flow: pd.DataFrame,
    as_of: str,
    fetched_at: dt.datetime,
) -> pd.DataFrame:
    if profit.empty or "code" not in profit.columns:
        return pd.DataFrame(columns=list(FINANCIAL_COLUMNS))
    out = profit.copy()
    for extra in (growth, balance, cash_flow):
        if not extra.empty and "code" in extra.columns:
            extra = extra.copy()
            extra = extra.drop(columns=["pubDate", "statDate"], errors="ignore")
            out = out.merge(extra, on="code", how="left")
    out["symbol"] = out["code"].astype(str).str.extract(SYMBOL_PATTERN, expand=False)
    out["pub_date"] = pd.to_datetime(out.get("pubDate"), errors="coerce").dt.date
    out["stat_date"] = pd.to_datetime(out.get("statDate"), errors="coerce").dt.date
    asof = _asof_date(as_of)
    out = out.loc[
        out["symbol"].notna()
        & out["pub_date"].notna()
        & out["stat_date"].notna()
        & (out["pub_date"] <= asof)
    ].copy()
    out = _numeric(
        out,
        (
            "roeAvg",
            "npMargin",
            "gpMargin",
            "netProfit",
            "epsTTM",
            "MBRevenue",
            "totalShare",
            "liqaShare",
            "YOYEquity",
            "YOYAsset",
            "YOYNI",
            "YOYEPSBasic",
            "YOYPNI",
            "currentRatio",
            "quickRatio",
            "cashRatio",
            "YOYLiability",
            "liabilityToAsset",
            "assetToEquity",
            "ebitToInterest",
            "CFOToOR",
            "CFOToNP",
            "CFOToGr",
        ),
    )
    out = out.rename(
        columns={
            "roeAvg": "roe_avg",
            "npMargin": "net_profit_margin",
            "gpMargin": "gross_profit_margin",
            "netProfit": "net_profit",
            "epsTTM": "eps_ttm",
            "MBRevenue": "revenue",
            "totalShare": "total_share",
            "liqaShare": "liqa_share",
            "YOYEquity": "yoy_equity",
            "YOYAsset": "yoy_asset",
            "YOYNI": "yoy_net_income",
            "YOYEPSBasic": "yoy_eps_basic",
            "YOYPNI": "yoy_profit_net_income",
            "currentRatio": "current_ratio",
            "quickRatio": "quick_ratio",
            "cashRatio": "cash_ratio",
            "YOYLiability": "yoy_liability",
            "liabilityToAsset": "liability_to_asset",
            "assetToEquity": "asset_to_equity",
            "ebitToInterest": "ebit_to_interest",
            "CFOToOR": "cfo_to_or",
            "CFOToNP": "cfo_to_np",
            "CFOToGr": "cfo_to_gr",
        }
    )
    out["source"] = BAOSTOCK_SOURCE
    out["fetched_at"] = fetched_at
    for column in FINANCIAL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, list(FINANCIAL_COLUMNS)].drop_duplicates(
        subset=["symbol", "stat_date"], keep="last"
    )


def sync_baostock(
    store: ArchiveStore,
    *,
    as_of: str,
    symbols: Iterable[str],
    close_prices: Mapping[str, float] | None = None,
    start_date: str = "2024-09-24",
    financial_refresh_days: int = 45,
    workers: int = 4,
    history_periods: int = 5,
    progress: Progress | None = None,
) -> FundamentalSyncResult:
    """同步当前交易日估值和最近若干期已披露财报。

    首次运行建立**当前全市场快照**，而不是对每只股票回补数百个历史日：BaoStock
    的免费接口按股票逐只返回，盲目全历史回补会让启动任务长时间占用网络。这里固定
    保留至少 `history_periods` 个报告期，保证增长趋势与财务质量因子有足够历史；之后
    每个新交易日只新增一份估值快照，新报告期出现或覆盖不足时才重新检查财务。
    ``start_date`` 保留在接口中，方便以后加入受控的历史基本面回补，不影响当前断点逻辑。
    """

    del start_date  # 当前策略只建立最新快照；参数保留用于未来历史回补。
    result = FundamentalSyncResult()
    started = dt.datetime.now(dt.UTC)

    def finish() -> FundamentalSyncResult:
        store.record_run(
            result.task,
            str(as_of)[:10],
            started,
            result.rows,
            "FAILED" if result.warning else "OK",
            result.summary(),
        )
        return result

    codes = normalize_symbols(symbols)
    if not codes:
        result.failures.append(("symbols", "没有可用的 A 股代码"))
        return finish()

    as_of_date = _asof_date(as_of)
    valuation_needed = str(as_of)[:10] not in store.valuation_dates()
    year, quarter, stat_date = _latest_quarter(as_of)
    history_periods = max(5, int(history_periods))
    periods = _period_candidates(as_of, limit=history_periods)
    stat_dates = [_period_stat_date(year_, quarter_) for year_, quarter_ in periods]
    covered = store.financial_period_coverage(stat_date, as_of)
    threshold = max(1, int(len(codes) * 0.80))
    latest_pub = store.conn.execute(
        "select max(pub_date) from financial_quarterly where pub_date <= ?",
        [as_of_date],
    ).fetchone()[0]
    stale = latest_pub is None or (as_of_date - latest_pub).days > max(1, financial_refresh_days)
    history_missing = any(
        store.financial_period_coverage(period, as_of) < threshold for period in stat_dates
    )
    financial_needed = covered < threshold or history_missing or stale
    result.skipped_valuation = not valuation_needed
    result.skipped_financial = not financial_needed

    if not valuation_needed and not financial_needed:
        if progress:
            progress(1, 1, "估值与财务归档均已覆盖，跳过重复抓取")
        return finish()

    fetched_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    financial_frames: list[pd.DataFrame] = []
    if financial_needed:
        result.financial_attempted = len(codes)
        worker_count = max(1, min(int(workers), len(codes)))
        payloads = [(symbol, str(as_of)[:10], periods) for symbol in codes]
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=get_context("spawn"),
                initializer=_init_financial_worker,
            ) as pool:
                futures = {
                    pool.submit(_fetch_financial_worker, payload): payload[0]
                    for payload in payloads
                }
                for index, future in enumerate(as_completed(futures), start=1):
                    symbol = futures[future]
                    try:
                        _symbol, frame, error_text = future.result()
                    except Exception as error:  # noqa: BLE001 - 进程异常不阻断其它源
                        frame = pd.DataFrame(columns=list(FINANCIAL_COLUMNS))
                        error_text = f"{type(error).__name__}: {error}"
                    if error_text:
                        result.failures.append((symbol, f"财务 {error_text}"))
                    elif frame.empty:
                        result.financial_empty += 1
                    else:
                        financial_frames.append(frame)
                        result.financial_ok += 1
                    if progress and (index == 1 or index % 25 == 0 or index == len(codes)):
                        progress(
                            index,
                            len(codes),
                            f"BaoStock 财务 {index}/{len(codes)}（{worker_count}进程）",
                        )
        except Exception as error:  # noqa: BLE001 - 汇总进程池/依赖错误
            result.failures.append(("pool", f"{type(error).__name__}: {error}"))
        if financial_frames:
            financial = pd.concat(financial_frames, ignore_index=True)
            result.financial_rows = store.upsert(
                "financial_quarterly", financial, FINANCIAL_COLUMNS
            )
    elif progress:
        progress(1, 1, "财务报告期已覆盖，跳过")

    # 财务归档完成后再构造估值快照。这样首次运行不会对 5000 多只股票再发一轮
    # history_k_data 请求；后续若财务已覆盖，也完全可以只用本地归档重建 PE(TTM)。
    if valuation_needed:
        financial_asof = store.financial_asof(str(as_of)[:10])
        result.valuation_attempted = len(codes)
        valuation = _prepare_derived_valuation(
            financial_asof,
            close_prices or {},
            str(as_of)[:10],
            fetched_at,
        )
        result.valuation_ok = int(valuation["symbol"].nunique()) if not valuation.empty else 0
        result.valuation_empty = max(0, result.valuation_attempted - result.valuation_ok)
        if not valuation.empty:
            result.valuation_rows = store.upsert(
                "valuation_daily", valuation, VALUATION_COLUMNS
            )
        if progress:
            progress(
                1,
                1,
                f"BaoStock 估值归档 {result.valuation_ok}/{len(codes)}（EPS(TTM)+本地收盘价）",
            )
    elif progress:
        progress(1, 1, "估值快照已覆盖，跳过")

    return finish()


__all__ = [
    "BAOSTOCK_SOURCE",
    "FINANCIAL_COLUMNS",
    "FundamentalSyncResult",
    "VALUATION_COLUMNS",
    "baostock_code",
    "normalize_symbols",
    "sync_baostock",
]
