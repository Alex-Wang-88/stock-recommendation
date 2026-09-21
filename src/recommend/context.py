"""筛选上下文 —— 把「行情 + 归档 + 名称」组装成一次筛选所需的全部输入。

一次筛选需要四类数据，来自三个地方：
* 行情（parquet，本项目 `MarketStore`）
* 题材 / 龙虎榜 / 两融（duckdb 归档，本项目 `ArchiveStore`）
* 名称与上市日（instrument_master，引导时复制来的上游数据）

`ScreenContext` 负责把它们的**时间口径对齐到同一个 as-of**。这件事很容易做错：
用了今天的名称去回看三个月前的筛选，就会含未来信息（ST 标记是后来才加的）。
所以名称只用于**当前**筛选；阶段 4 的回测必须按历史时点重建名称，不能复用这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .archive import ArchiveStore
from .config import AppConfig
from .factors import FactorTable, build_factor_table
from .market import MarketStore, default_base_dates

# 题材 / 龙虎榜 / 两融的「近 N 个交易日」窗口长度。
#
# ⚠️ 必须按**交易日**取窗口边界，不能用"as_of 往前 N 个自然日"：
# 春节 / 国庆会让自然日窗口只覆盖十几个交易日，于是 `theme_days_20` 的实际口径
# 会随季节漂移。这里从行情里取第 N 个交易日，口径恒定。
LOOKBACK_DAYS = 20


@dataclass
class ScreenContext:
    config: AppConfig
    market: MarketStore
    archive: ArchiveStore
    as_of: str
    window_start: str
    notes: list[str] = field(default_factory=list)

    def close(self) -> None:
        self.market.close()
        self.archive.close()

    def __enter__(self) -> ScreenContext:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 组装 ------------------------------------------------------------- #

    @classmethod
    def open(cls, config: AppConfig, as_of: str | None = None) -> ScreenContext:
        market = MarketStore(config.market.daily_parquet)
        if not market.ready:
            raise FileNotFoundError(
                f"行情归档不存在：{config.market.daily_parquet}。先跑 `recommend sync-bars`。"
            )
        archive = ArchiveStore(config.archive_db)
        notes: list[str] = []

        if as_of:
            resolved = str(as_of)[:10]
            available = set(market.trade_dates(end=resolved))
            if resolved not in available:
                raise ValueError(
                    f"{resolved} 不是归档里的交易日。"
                    f"最近可用：{market.trade_dates(end=resolved, limit=3)}"
                )
        else:
            raw_latest = market.latest_trade_date()
            resolved = market.latest_complete_trade_date()
            if resolved != raw_latest:
                notes.append(
                    f"行情已出现 {raw_latest} 的盘中数据，但尚未过 15:30；"
                    f"推荐暂按最近完整交易日 {resolved} 计算。"
                )

        # 窗口 = 含 as_of 在内的最近 LOOKBACK_DAYS 个交易日。
        dates = market.trade_dates(end=resolved, limit=LOOKBACK_DAYS)
        window_start = dates[0] if dates else resolved

        return cls(
            config=config,
            market=market,
            archive=archive,
            as_of=resolved,
            window_start=window_start,
            notes=notes,
        )

    # -- 数据 ------------------------------------------------------------- #

    def base_closes(self) -> pd.DataFrame:
        return self.market.closes_on(
            default_base_dates(self.as_of, self.config.market.since_924)
        )

    def instrument_meta(self) -> pd.DataFrame:
        meta = self.market.symbol_meta(self.config.market.instrument_parquet)
        if meta.empty:
            self.notes.append(
                f"未找到 instrument_master：{self.config.market.instrument_parquet}"
                " ⇒ 名称 / 上市天数为空，ST 剔除与次新筛选失效。"
            )
            meta = pd.DataFrame(columns=["symbol", "name", "listed_date"])

        meta = self.market.listed_days(meta, self.as_of)
        industry = self.market.industry_meta(self.config.market.industry_csv)
        if industry.empty:
            self.notes.append(
                f"未找到可用行业快照：{self.config.market.industry_csv}"
                " ⇒ 页面行业显示为‘未同步’，行业评分缺失。"
            )
            meta["industry"] = pd.NA
        else:
            meta = meta.merge(industry, on="symbol", how="left")
        return meta

    def build(self) -> FactorTable:
        """组装因子宽表。缺哪一块就只让对应列为 NaN，并记一条 note。"""

        bars = self.market.bars(end=self.as_of, window=self.config.market.window)
        if bars.empty:
            raise ValueError(f"{self.as_of} 没有任何行情数据")

        # universe：以有行情的标的为准（as-of 当天停牌的标的不会出现在这里）。
        last_date = bars["trade_date"].max()
        today = bars[bars["trade_date"] == last_date]
        universe = pd.DataFrame({"symbol": today["symbol"].astype(str).str.zfill(6)})
        meta = self.instrument_meta()
        if not meta.empty:
            universe = universe.merge(meta, on="symbol", how="left")

        attribution = self.archive.themes_on(self.as_of)
        if attribution.empty:
            self.notes.append(
                f"{self.as_of} 的题材归因为空 ⇒ 题材因子全为缺失。"
                "若不是非交易日，请跑 `recommend backfill themes` 补齐。"
            )
        tag_daily = self.archive.tag_counts_on(self.as_of)

        theme_history = None
        if not attribution.empty:
            theme_history = self.archive.theme_history(
                self.window_start, self.as_of, symbols=universe["symbol"].tolist()
            )

        if self.archive.table_row_count("dragon_tiger"):
            dragon_tiger = self.archive.dragon_tiger_between(self.window_start, self.as_of)
        else:
            dragon_tiger = pd.DataFrame()
            self.notes.append("龙虎榜归档为空 ⇒ `lhb_*` 因子为缺失（不是 0）。")

        if self.archive.table_row_count("margin_trading"):
            margin = self.archive.margin_between(self.window_start, self.as_of)
        else:
            margin = pd.DataFrame()
            self.notes.append("两融归档为空 ⇒ `margin_rz_chg_20` 为缺失（不是 0）。")

        valuation = self.archive.valuation_asof(self.as_of)
        if valuation.empty:
            self.notes.append(
                "估值归档为空 ⇒ PE(TTM) 未参与估值评分；请先运行同步并检查 BaoStock。"
            )

        financial = self.archive.financial_asof(self.as_of)
        if financial.empty:
            self.notes.append(
                "财务归档为空 ⇒ 增长质量未参与评分；请先运行同步并检查 BaoStock。"
            )
        financial_history = self.archive.financial_history_asof(
            self.as_of, periods=self.config.fundamentals.history_periods
        )
        if financial_history.empty:
            self.notes.append("多期财务归档为空 ⇒ 营收同比/增速加速度未参与评分。")
        elif "symbol" in financial_history.columns:
            period_counts = financial_history.groupby("symbol", dropna=True).size()
            insufficient = int((period_counts < self.config.fundamentals.history_periods).sum())
            if insufficient:
                periods = self.config.fundamentals.history_periods
                self.notes.append(
                    f"{insufficient} 只股票的财务历史少于 {periods} 期，"
                    "营收同比/增速加速度按可用数据计算。"
                )

        table = build_factor_table(
            universe=universe,
            bars=bars,
            base_closes=self.base_closes(),
            attribution=attribution,
            tag_daily=tag_daily,
            theme_history=theme_history,
            dragon_tiger=dragon_tiger,
            margin=margin,
            valuation=valuation,
            financial=financial,
            financial_history=financial_history,
            as_of=self.as_of,
        )
        table.notes = list(self.notes) + list(table.notes)
        return table


def resolve(ctx: ScreenContext) -> FactorTable:  # pragma: no cover - 便捷别名
    return ctx.build()


__all__ = ["LOOKBACK_DAYS", "ScreenContext", "resolve"]
