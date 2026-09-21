"""自包含性回归闸门 —— 「本项目能不能在没有量化项目的情况下独立运行」。

为什么值得单独一组测试
--------------------
跨项目依赖的失效方式**不是报错**，而是：

* `instrument_parquet` 指去别的目录 ⇒ 上游一动，`listed_days` 整列缺失，
  `low-position` 预置的 `listed_days >= 365` 变成"条件无法评估" ⇒ **筛出零只票**。
  看起来像"今天没有符合条件的股票"，实际是路径断了。
* `source_parquet` 被日更路径读到 ⇒ 上游一动，**每天的定时任务全挂**。

这类耦合会随着时间被人"顺手加回去"（比如为了图省事把路径指回上游）。
所以用测试钉住，而不是靠注释。
"""

from __future__ import annotations

from pathlib import Path

from recommend.config import AppConfig, MarketConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_files() -> list[Path]:
    return sorted(
        path
        for folder in ("src", "scripts", "config")
        for path in (PROJECT_ROOT / folder).rglob("*")
        if path.is_file() and path.suffix in {".py", ".toml", ".cmd"}
    )


def test_instrument_table_is_a_local_copy():
    """名称 / 上市日表必须在本项目里 —— 它**每次筛选**都会被读。"""

    path = MarketConfig().instrument_parquet
    assert ".." not in path.parts, (
        f"instrument_parquet 指到了项目外（{path}）。"
        "它的失效方式是'筛出零只票'而不是报错，很难发现。"
        "刷新副本用 scripts/refresh_reference.py。"
    )
    assert path == Path("data/reference/instrument_master.parquet")


def test_default_toml_agrees_with_the_dataclass_defaults():
    """`config/default.toml` 与 dataclass 默认值不能漂移。

    TOML 存在时它**优先**，所以两边不一致时，测试里看到的是默认值、
    实际跑的是 TOML 的值 —— 这种错位会让"测试通过但线上不一样"。
    """

    config = AppConfig.load(None)
    defaults = AppConfig()
    assert config.market.daily_parquet == defaults.market.daily_parquet
    assert config.market.instrument_parquet == defaults.market.instrument_parquet
    assert config.market.industry_csv == defaults.market.industry_csv
    assert config.market.source_parquet == defaults.market.source_parquet
    assert ".." not in config.market.instrument_parquet.parts
    assert ".." not in config.market.industry_csv.parts
    assert ".." not in config.market.source_parquet.parts


def test_bootstrap_seed_paths_are_inside_the_project():
    """首次引导与参考数据刷新也不能依赖相邻的 quantitative 目录。"""

    for path in (
        MarketConfig().source_parquet,
        MarketConfig().instrument_parquet,
        MarketConfig().industry_csv,
    ):
        assert ".." not in path.parts


def test_only_the_bootstrap_command_reads_the_upstream_source():
    """`source_parquet`（上游引导源）只能出现在 `sync-bars` 里。

    日更走 `bars.py` 抓腾讯，与上游无关。一旦日更路径读了这个路径，
    每日任务就依赖另一个项目存在 —— 而"另一个项目会不会被挪走"不在本项目的控制范围。
    """

    offenders: list[str] = []
    for path in project_files():
        if path.name == "config.py" or path.name == "default.toml":
            continue  # 定义处
        if "source_parquet" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == ["src\\recommend\\cli.py"] or offenders == ["src/recommend/cli.py"], (
        f"source_parquet 被这些文件引用了：{offenders}。"
        "它只应出现在 cli.py 的 sync-bars（一次性引导）中。"
    )


def test_no_absolute_paths_in_source_or_config():
    """源码/配置里不能出现绝对路径 —— 否则项目无法整体搬移。"""

    hits: list[str] = []
    for path in project_files():
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        ):
            if "C:\\" in line or "C:/" in line:
                hits.append(f"{path.relative_to(PROJECT_ROOT)}:{number}")
    assert hits == [], f"发现硬编码绝对路径：{hits}"
