from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HttpConfig:
    min_interval_seconds: float = 1.2
    jitter_seconds: float = 0.4
    retries: int = 3
    backoff_seconds: float = 2.0
    timeout_seconds: float = 15.0


@dataclass(frozen=True)
class BackfillConfig:
    themes_start: str = "2024-09-01"
    dragon_tiger_start: str = "2024-09-01"
    # 官方两融是按交易日全市场回补；默认先从近三个月开始，避免首次运行变成
    # 数千次请求。需要更长历史时在 CLI 显式传 --start。
    margin_start: str = "2026-07-01"


@dataclass(frozen=True)
class MarketConfig:
    """行情引导与基准日。

    **本项目与量化项目的唯一接触面是「上游原始行情」，而且只在两个时点**：

    * `instrument_parquet` —— 名称 / 上市日表。**已本地化为本项目自己的副本**：
      它每次筛选都会被读（ST 剔除、次新筛选、展示名称），指去别的项目就等于
      让每日任务依赖那个目录存在，且失效方式是"筛不出任何票"而不是报错。
    * `industry_csv` —— 个股行业快照。用于推荐卡片、报告展示和行业横截面评分。
    * `source_parquet` —— 仅 `sync-bars` 一次性引导用。日更走 `bars.py` 抓腾讯，
      **不读它**。本项目日线一旦建立，这个路径只在重建归档时才需要。

    两者都只复制**行情原始数据**，不含任何模型、参数或结论。
    """

    daily_parquet: Path = Path("data/market/daily.parquet")
    # 本地副本优先；不要改回项目外目录。
    instrument_parquet: Path = Path("data/reference/instrument_master.parquet")
    # 个股行业快照；运行时只读项目内文件，不因行业数据缺失阻断推荐。
    industry_csv: Path = Path("data/reference/industry_master.csv")
    # 仅引导用（sync-bars）。每日任务不读；原始种子也随项目一起搬运。
    source_parquet: Path = Path("data/reference/source_daily.parquet")
    # 「924 行情」起点：2024-09-24 是那轮政策的第一个交易日。
    since_924: str = "2024-09-24"
    window: int = 300


@dataclass(frozen=True)
class FundamentalsConfig:
    """本地基本面归档与 BaoStock 同步策略。

    基本面不是运行时直连数据：同步成功后会写入项目内 DuckDB，推荐计算只读
    本地归档。``enabled`` 默认关闭是为了让纯单元测试和没有安装可选依赖的
    环境仍能安全构造 ``AppConfig``；正式配置文件会显式打开它。
    """

    enabled: bool = False
    start_date: str = "2024-09-24"
    financial_refresh_days: int = 45
    workers: int = 4
    history_periods: int = 5


@dataclass(frozen=True)
class RankWeights:
    """排序权重 —— **政策参数，不是拟合出来的**（方案 §四.3：排序不用 ML）。

    只能包含可解释的方向性因子（行业 / 增长 / 财务质量 / 估值 / 动量 / 风险 /
    题材 / 资金 / 流动性）。
    `POSITION_FACTORS` 里的位置类因子（如距 52 周高点）**禁止**进入此处，
    这是「低位只做筛选、不做排序加分」的机械保证（`rank/score.py` 会 raise）。
    """

    industry: float = 0.20
    growth: float = 0.18
    quality: float = 0.15
    valuation: float = 0.15
    momentum: float = 0.15
    risk: float = 0.08
    theme: float = 0.02
    capital: float = 0.02
    liquidity: float = 0.05

    @property
    def growth_quality(self) -> float:
        """旧版配置的兼容视图；新排序使用 growth + quality 两类。"""

        return self.growth + self.quality

    def as_dict(self) -> dict[str, float]:
        return {
            "industry": self.industry,
            "growth": self.growth,
            "quality": self.quality,
            "valuation": self.valuation,
            "momentum": self.momentum,
            "risk": self.risk,
            "theme": self.theme,
            "capital": self.capital,
            "liquidity": self.liquidity,
        }


@dataclass(frozen=True)
class AppConfig:
    archive_db: Path = Path("data/archive.duckdb")
    logs_dir: Path = Path("logs")
    reports_dir: Path = Path("data/reports")
    http: HttpConfig = HttpConfig()
    backfill: BackfillConfig = BackfillConfig()
    market: MarketConfig = MarketConfig()
    fundamentals: FundamentalsConfig = FundamentalsConfig()
    rank: RankWeights = field(default_factory=RankWeights)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        """Load config from TOML. Missing file falls back to the defaults."""

        if path is None:
            candidate = Path("config/default.toml")
            if not candidate.exists():
                return cls()
            path = candidate
        path = Path(path)
        if not path.exists():
            return cls()
        payload = tomllib.loads(path.read_text(encoding="utf-8"))

        paths = payload.get("paths", {})
        http = payload.get("http", {})
        backfill = payload.get("backfill", {})
        market = payload.get("market", {})
        fundamentals = payload.get("fundamentals", {})
        rank = payload.get("rank", {})
        defaults = cls()
        legacy_growth_quality = rank.get("growth_quality")
        if "growth" in rank:
            growth_weight = float(rank["growth"])
        elif legacy_growth_quality is not None:
            growth_weight = float(legacy_growth_quality) * 0.60
        else:
            growth_weight = defaults.rank.growth
        if "quality" in rank:
            quality_weight = float(rank["quality"])
        elif legacy_growth_quality is not None:
            quality_weight = float(legacy_growth_quality) - growth_weight
        else:
            quality_weight = defaults.rank.quality
        return cls(
            archive_db=Path(paths.get("archive_db", str(defaults.archive_db))),
            logs_dir=Path(paths.get("logs_dir", str(defaults.logs_dir))),
            reports_dir=Path(paths.get("reports_dir", str(defaults.reports_dir))),
            http=HttpConfig(
                min_interval_seconds=float(http.get("min_interval_seconds", 1.2)),
                jitter_seconds=float(http.get("jitter_seconds", 0.4)),
                retries=int(http.get("retries", 3)),
                backoff_seconds=float(http.get("backoff_seconds", 2.0)),
                timeout_seconds=float(http.get("timeout_seconds", 15.0)),
            ),
            backfill=BackfillConfig(
                themes_start=str(backfill.get("themes_start", "2024-09-01")),
                dragon_tiger_start=str(backfill.get("dragon_tiger_start", "2024-09-01")),
                margin_start=str(backfill.get("margin_start", "2026-07-01")),
            ),
            market=MarketConfig(
                daily_parquet=Path(market.get("daily_parquet", str(defaults.market.daily_parquet))),
                source_parquet=Path(
                    market.get("source_parquet", str(defaults.market.source_parquet))
                ),
                instrument_parquet=Path(
                    market.get("instrument_parquet", str(defaults.market.instrument_parquet))
                ),
                industry_csv=Path(
                    market.get("industry_csv", str(defaults.market.industry_csv))
                ),
                since_924=str(market.get("since_924", defaults.market.since_924)),
                window=int(market.get("window", defaults.market.window)),
            ),
            fundamentals=FundamentalsConfig(
                enabled=bool(fundamentals.get("enabled", defaults.fundamentals.enabled)),
                start_date=str(fundamentals.get("start_date", defaults.fundamentals.start_date)),
                financial_refresh_days=int(
                    fundamentals.get(
                        "financial_refresh_days", defaults.fundamentals.financial_refresh_days
                    )
                ),
                workers=max(1, int(fundamentals.get("workers", defaults.fundamentals.workers))),
                history_periods=max(
                    5,
                    int(
                        fundamentals.get(
                            "history_periods", defaults.fundamentals.history_periods
                        )
                    ),
                ),
            ),
            rank=RankWeights(
                industry=float(rank.get("industry", defaults.rank.industry)),
                growth=growth_weight,
                quality=quality_weight,
                valuation=float(rank.get("valuation", defaults.rank.valuation)),
                momentum=float(rank.get("momentum", defaults.rank.momentum)),
                risk=float(rank.get("risk", defaults.rank.risk)),
                theme=float(rank.get("theme", defaults.rank.theme)),
                capital=float(rank.get("capital", defaults.rank.capital)),
                liquidity=float(rank.get("liquidity", defaults.rank.liquidity)),
            ),
        )
