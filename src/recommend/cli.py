"""命令行入口：归档层回补 / 快照 / 诊断，以及阶段 1 的确定性筛选。

用法示例
--------
    .venv/Scripts/python.exe -m recommend.cli probe
    .venv/Scripts/python.exe -m recommend.cli backfill themes --start 2026-09-01 --end 2026-09-18
    .venv/Scripts/python.exe -m recommend.cli snapshot
    .venv/Scripts/python.exe -m recommend.cli status
    .venv/Scripts/python.exe -m recommend.cli sync-bars
    .venv/Scripts/python.exe -m recommend.cli fields
    .venv/Scripts/python.exe -m recommend.cli screen --preset low-position
    .venv/Scripts/python.exe -m recommend.cli screen --preset theme-watch --theme 算力
    .venv/Scripts/python.exe -m recommend.cli screen --spec '{"close_position_250": [null, 0.3]}'
    .venv/Scripts/python.exe -m recommend.cli daily-advance
    .venv/Scripts/python.exe -m recommend.cli daily-job
    .venv/Scripts/python.exe -m recommend.cli sync-all
    .venv/Scripts/python.exe -m recommend.cli forward-report

⚠️ 运行 CLI 需要 `PYTHONPATH=src`（或先 `pip install -e .`）。`pythonpath` 配置只对 pytest 生效。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from .archive import (
    ArchiveStore,
    backfill_dragon_tiger,
    backfill_margin,
    backfill_margin_exchange,
    backfill_themes,
    check_connectivity,
    snapshot_all,
    symbols_from_themes,
)
from .bars import advance_bars, local_now
from .config import AppConfig
from .context import ScreenContext
from .factors import FACTOR_CATEGORY, ID_COLUMNS
from .forward import append_record, build_record, evaluate_and_persist
from .http import ThrottledClient
from .market import MarketStore, is_current_eod_ready
from .report import render_markdown, to_display
from .screen import FILTER_FIELDS, Spec, SpecError, run_screen
from .screen.presets import PRESETS, load_preset
from .sync import run_sync_all


def _client(config: AppConfig) -> ThrottledClient:
    return ThrottledClient(
        min_interval=config.http.min_interval_seconds,
        jitter=config.http.jitter_seconds,
        retries=config.http.retries,
        backoff=config.http.backoff_seconds,
        timeout=config.http.timeout_seconds,
    )


def _store(config: AppConfig) -> ArchiveStore:
    return ArchiveStore(config.archive_db)


def _split_symbols(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().zfill(6) for item in value.split(",") if item.strip()]


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #


def cmd_probe(config: AppConfig, _args: argparse.Namespace) -> int:
    """通路诊断。区分「网络/IP 不可用」与「某端点/参数不对」。"""

    client = _client(config)
    alive = check_connectivity(client)
    print(f"对照端点（东财个股资金流日线）：{'✅ 可用' if alive else '❌ 不可用'}")
    if alive:
        print("⇒ 通路正常。若某个业务端点仍失败，**是端点或参数问题，不是被封 IP**。")
        print("   不要盲目加冷却时间，先检查 secid / fields 的编码。")
    else:
        print("⇒ 网络或 IP 层面可能确实不可用，稍后再试或降低频率。")
    return 0 if alive else 1


def cmd_backfill(config: AppConfig, args: argparse.Namespace) -> int:
    store = _store(config)
    client = _client(config)
    try:
        if args.source == "themes":
            result = backfill_themes(store, client, args.start, args.end, force=args.force)
        elif args.source == "dragon-tiger":
            result = backfill_dragon_tiger(store, client, args.start, args.end, force=args.force)
        elif args.source == "margin":
            symbols = _split_symbols(args.symbols)
            if args.legacy_per_symbol or symbols or args.from_themes:
                if not symbols and args.from_themes:
                    symbols = symbols_from_themes(store, args.limit)
                if not symbols:
                    print("旧版逐标的两融回补需要 --symbols 或 --from-themes", file=sys.stderr)
                    return 2
                print(f"旧版逐标的两融回补：{len(symbols)} 只")
                result = backfill_margin(store, client, symbols, page_size=args.page_size)
            else:
                if not args.start or not args.end:
                    print("官方两融回补需要 --start 和 --end", file=sys.stderr)
                    return 2
                result = backfill_margin_exchange(
                    store, client, args.start, args.end, force=args.force
                )
        else:  # pragma: no cover - argparse 已限制取值
            raise ValueError(args.source)
    finally:
        store.close()

    print(result.summary())
    return 1 if result.failed and not result.ok else 0


def cmd_snapshot(config: AppConfig, args: argparse.Namespace) -> int:
    """T2 当日快照。

    ⚠️ 日期**必须**来自行情归档的交易日，不能用系统日期：
    T2 接口不带日期字段，周六调用它返回的是周五的数据。按系统日期盖章的话，
    周五的行情会被写成"周六的"，之后按交易日对齐的逻辑全部错位。
    """

    target = args.date or _trade_date_from_archive(config)
    if not is_current_eod_ready(target):
        print(
            f"当前不能安全保存 {target} 的 T2 快照：只有当天收盘后（15:30 以后）"
            "才能把接口返回的当日数据盖章；请收盘后重试。"
        )
        return 0
    store = _store(config)
    client = _client(config)
    try:
        results = snapshot_all(store, client, target)
    finally:
        store.close()
    for result in results:
        print(result.summary())
    return 1 if any(result.failed for result in results) else 0


def cmd_status(config: AppConfig, _args: argparse.Namespace) -> int:
    store = _store(config)
    try:
        coverage = store.coverage()
        print("=== 归档覆盖 ===")
        print(coverage.to_string(index=False))
        print()
        print("=== 最近运行 ===")
        print(store.run_log(limit=10).to_string(index=False))
    finally:
        store.close()
    return 0


def cmd_sync_all(config: AppConfig, args: argparse.Namespace) -> int:
    """统一同步入口：执行注册表中的全部数据源并生成推荐。"""

    def report(event) -> None:
        print(f"[{event.fraction:6.1%}] {event.label} · {event.message}", flush=True)

    run = run_sync_all(
        config,
        preset=args.preset,
        workers=args.workers,
        progress=report,
    )
    print("\n=== 同步汇总 ===")
    for step in run.steps:
        print(f"{step.status.upper():8} {step.label}：{step.summary}")
        if step.detail:
            print(f"           {step.detail}")
    return 0 if run.ok else 1


def cmd_quality(config: AppConfig, _args: argparse.Namespace) -> int:
    """自查归档里有没有混进「回退快照」。

    判据：真实交易日**必带行情字段**、且行级 `source_date` 等于 `trade_date`。
    任一条不满足就是可疑天，应当清除后重抓。
    """

    store = _store(config)
    try:
        suspicious = store.theme_quality()
    finally:
        store.close()
    if suspicious.empty:
        print("✅ 题材归档未发现可疑日（无回退快照）")
        return 0
    print(f"⚠️ 发现 {len(suspicious)} 个可疑日（很可能被上游回退快照污染）：")
    print(suspicious.to_string(index=False))
    print()
    preview = ",".join(str(day)[:10] for day in suspicious["trade_date"][:5])
    print("清除后重抓：")
    print(f"  recommend purge --dates {preview}...")
    return 1


def cmd_purge(config: AppConfig, args: argparse.Namespace) -> int:
    """按日清除已确认被污染的记录（清除后可重跑回补）。"""

    dates = [item.strip() for item in args.dates.split(",") if item.strip()]
    if not dates:
        print("需要 --dates（逗号分隔）", file=sys.stderr)
        return 2
    store = _store(config)
    total = 0
    try:
        for trade_date in dates:
            removed = store.purge_date(args.table, trade_date)
            total += removed
            print(f"  清除 {args.table} {trade_date}：{removed} 行")
    finally:
        store.close()
    print(f"合计清除 {total} 行")
    return 0


def cmd_themes(config: AppConfig, args: argparse.Namespace) -> int:
    store = _store(config)
    try:
        frame = store.top_tags(args.date, args.limit)
    finally:
        store.close()
    if frame.empty:
        print(f"{args.date} 无题材标签记录（归档未覆盖该日，或该日无数据）")
        return 1
    print(f"=== {args.date} 题材热度 TOP{args.limit} ===")
    print(frame.to_string(index=False))
    return 0


# --------------------------------------------------------------------------- #
# 阶段 1：行情引导 / 因子目录 / 确定性筛选
# --------------------------------------------------------------------------- #


def cmd_sync_bars(config: AppConfig, args: argparse.Namespace) -> int:
    """从既有归档一次性引导本项目自己的日线。

    这是本项目唯一「借用」量化项目的东西：**上游原始行情**。
    引导后本项目自己维护 `data/market/daily.parquet`，两边从此各走各的。
    """

    source = Path(args.source) if args.source else config.market.source_parquet
    with MarketStore(config.market.daily_parquet) as store:
        try:
            rows = store.sync_from(source)
        except FileNotFoundError as error:
            print(f"❌ {error}", file=sys.stderr)
            return 1
        print(f"✅ 已引导 {rows:,} 行 → {store.path}")
        dates = store.trade_dates()
        print(f"   区间：{dates[0]} → {dates[-1]}（{len(dates)} 个交易日）")
        base = store.closes_on({"base_924": config.market.since_924})
        covered = int(base["base_924"].notna().sum())
        print(f"   基准日 {config.market.since_924} 有收盘价的标的：{covered} 只")
    return 0


def cmd_fields(_config: AppConfig, _args: argparse.Namespace) -> int:
    """列出 spec 可用的字段 —— 阶段 2 的 LLM 提示词要引用这张表。"""

    print("=== 筛选字段（白名单，未知字段会被拒绝） ===")
    for name, help_text in FILTER_FIELDS.items():
        category = FACTOR_CATEGORY.get(name, "meta")
        print(f"  {name:<22} [{category:<9}] {help_text}")
    print()
    print("=== 排序白名单（位置因子无资格） ===")
    from .screen import ALLOWED_SORTS

    print("  " + ", ".join(ALLOWED_SORTS))
    print()
    print("=== 预置 ===")
    for name, payload in PRESETS.items():
        print(f"  {name:<16} {payload.get('note', '')}")
    return 0


def _load_spec(args: argparse.Namespace) -> dict:
    if args.spec_file:
        path = Path(args.spec_file)
        if not path.exists():
            raise SpecError(f"spec 文件不存在：{path}")
        return json.loads(path.read_text(encoding="utf-8"))
    if args.spec:
        return json.loads(args.spec)
    if args.preset:
        try:
            payload = load_preset(args.preset)
        except KeyError as error:
            raise SpecError(str(error)) from error
        if args.theme:
            payload["themes"] = [item.strip() for item in args.theme.split(",") if item.strip()]
        return payload
    raise SpecError("需要 --spec / --spec-file / --preset 之一")


def _persist_report(result, out: Path) -> None:
    """报告落盘：Markdown（人看）+ CSV（机器看）。

    ⚠️ CSV 用 `utf-8-sig`：不加 BOM 的话 Excel 打开中文列名会是乱码。
    """

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(result), encoding="utf-8")
    print(f"\n📄 报告已写入 {out}")
    csv_path = out.with_suffix(".csv")
    result.frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📄 完整因子明细已写入 {csv_path}")


def cmd_screen(config: AppConfig, args: argparse.Namespace) -> int:
    """确定性筛选：spec → 候选表 + 因子明细 + 剔除日志。"""

    try:
        payload = _load_spec(args)
        if args.limit:
            payload["limit"] = args.limit
        if args.as_of:
            payload["as_of"] = args.as_of
        spec = Spec.from_dict(payload)
    except (SpecError, json.JSONDecodeError) as error:
        print(f"❌ spec 不合法：{error}", file=sys.stderr)
        return 2

    with ScreenContext.open(config, spec.as_of) as ctx:
        table = ctx.build()
        spec = replace(spec, as_of=table.as_of)
        result = run_screen(table, spec, config.rank.as_dict())

    print(f"as-of {result.as_of}")
    print(
        f"全市场 {result.universe_size} 只 → 通过条件 {result.pool_size} 只"
        f" → 取前 {result.kept} 只"
    )
    for line in result.rejection_lines():
        print(line)
    if result.truncated:
        print(f"  - 被 limit 截断: {result.truncated} 只（未显示，不是未通过）")
    for note in result.notes:
        print(f"  ! {note}")
    print()
    if result.frame.empty:
        print("（无命中。检查上面的剔除日志 —— 很可能是某个因子整列缺失。）")
        return 1
    print(to_display(result.frame).to_string(index=False))

    if args.out:
        _persist_report(result, Path(args.out))

    if args.verbose:
        print()
        print("=== 因子列口径 ===")
        for column in result.frame.columns:
            if column in ID_COLUMNS or column.startswith("score"):
                continue
            print(f"  {column:<22} [{FACTOR_CATEGORY.get(column, 'meta')}]")
    return 0


# --------------------------------------------------------------------------- #
# 前向纸盘：每日推进 + 留证 + 对账
# --------------------------------------------------------------------------- #


def _forward_dir(config: AppConfig, override: str | None) -> Path:
    return Path(override) if override else Path(config.reports_dir) / "forward"


def _screen_and_record(config: AppConfig, args: argparse.Namespace, as_of: str) -> int:
    """按冻结 spec 筛选指定 as-of，并把**留证**落盘。返回退出码。

    为什么必须留证而不是"每天看一眼就算"：T2 类数据（题材热度 / 龙虎榜 / 两融）
    **只有当日可取、历史补不回来**。今天不留痕，明天就无法回答
    「按今天的信号选出来的票，后来涨了没有」。这个样本**只能从今天开始攒**。

    因此留证记录里同时存了：当时的 `close`（对账锚点）、`spec` 指纹（保证是同一个
    spec 连续跑）、以及**剔除日志**（否则日后分不清"条件太严"与"因子整列缺失"）。
    """

    # 定时任务不传参也要能跑：默认用 `low-position` 这个冻结预置。
    if not (args.spec or args.spec_file or args.preset):
        args.preset = "low-position"
    try:
        payload = _load_spec(args)
        payload["as_of"] = as_of
        spec = Spec.from_dict(payload)
    except (SpecError, json.JSONDecodeError) as error:
        print(f"❌ spec 不合法：{error}", file=sys.stderr)
        return 2

    with ScreenContext.open(config, spec.as_of) as ctx:
        table = ctx.build()
        spec = replace(spec, as_of=table.as_of)
        result = run_screen(table, spec, config.rank.as_dict())

    print(
        f"as-of {result.as_of}：全市场 {result.universe_size} 只"
        f" → 通过条件 {result.pool_size} 只 → 取前 {result.kept} 只"
    )
    for note in result.notes:
        print(f"  ! {note}")

    record = build_record(result, run_at=local_now())
    directory = _forward_dir(config, args.forward_dir)
    record_path, journal_path = append_record(directory, record)
    print(f"📌 留证 {record_path}")
    print(f"📌 汇总 {journal_path}（spec 指纹 {record.spec_hash}）")

    with MarketStore(config.market.daily_parquet) as market:
        _per_run, grouped, evaluation_path, summary_path = evaluate_and_persist(
            market,
            directory,
        )
    print(f"📈 验证明细 {evaluation_path}")
    print(f"📈 验证汇总 {summary_path}")
    if not grouped.empty:
        print(grouped.to_string(index=False))

    out = Path(args.out) if args.out else Path(config.reports_dir) / f"{result.as_of}.md"
    _persist_report(result, out)
    return 0


def cmd_daily_advance(config: AppConfig, args: argparse.Namespace) -> int:
    """一个交易日的前向纸盘：推进行情 → 按冻结 spec 筛选 → 留证。"""

    with MarketStore(config.market.daily_parquet) as store:
        if not store.ready:
            print(
                f"❌ 行情归档不存在：{store.path}。先跑 `recommend sync-bars` 引导一次。",
                file=sys.stderr,
            )
            return 1
        advance = advance_bars(store, workers=args.workers)
        print(advance.summary())
        # 收盘前不能把盘中日线当成完整 as-of；收盘后才使用当天日期。
        as_of = store.latest_complete_trade_date()

    if args.no_screen:
        return 0
    return _screen_and_record(config, args, as_of)


def cmd_forward_report(config: AppConfig, args: argparse.Namespace) -> int:
    """读回前向纸盘留证，用**实际后续走势**对账。

    ⚠️ 输出的只是描述性统计。**持有期没走完的记录不会按 0% 计入**（那会把
    "还不可测"伪装成"没涨没跌"），而是从 `measured` 里排除 —— 两者必须分开报。
    """

    directory = _forward_dir(config, args.forward_dir)
    with MarketStore(config.market.daily_parquet) as store:
        if not store.ready:
            print(f"❌ 行情归档不存在：{store.path}", file=sys.stderr)
            return 1
        per_run, grouped, evaluation_path, summary_path = evaluate_and_persist(store, directory)

    if per_run.empty:
        print(f"{directory} 还没有前向记录。先跑 `recommend daily-advance`。")
        return 1

    print(f"=== 逐次明细（{directory}） ===")
    print(per_run.to_string(index=False))
    print()
    print("=== 按持有期汇总 ===")
    print(grouped.to_string(index=False))
    print()
    print(f"验证明细已写入：{evaluation_path}")
    print(f"验证汇总已写入：{summary_path}")
    print()
    print("⚠️ 样本天数少时以上差额基本是噪声。要看方向是否一致，不要看点估计大小。")
    return 0


def _registered_daily_job(config: AppConfig, args: argparse.Namespace) -> int:
    """默认定时任务也走统一注册表，避免网页和 CLI 漏掉新数据源。"""

    skipped: set[str] = set()
    if args.skip_themes:
        skipped.add("themes")
    if args.skip_snapshot:
        skipped.add("sector")

    def report(event) -> None:
        print(f"[{event.fraction:6.1%}] {event.label} · {event.message}", flush=True)

    run = run_sync_all(
        config,
        preset=args.preset or "low-position",
        workers=args.workers,
        progress=report,
        skip_sources=skipped,
        skip_recommendation=args.no_screen,
        skip_validation=args.no_screen,
    )
    print("\n=== 每日归档任务汇总 ===")
    for step in run.steps:
        print(f"  · {step.status.upper()} {step.label}：{step.summary}")
    return 0 if run.ok else 1


def _trade_date_from_archive(config: AppConfig) -> str:
    """行情归档里最近完整交易日 —— 快照/筛选**唯一**可信的 as-of 来源。"""

    with MarketStore(config.market.daily_parquet) as store:
        if not store.ready:
            raise FileNotFoundError(
                f"行情归档不存在：{store.path}。快照需要它来确定交易日，"
                "先跑 `recommend sync-bars`。"
            )
        return store.latest_complete_trade_date()


def _latest_archived_date(store: ArchiveStore, table: str) -> str | None:
    """某张归档表最后一个有数据的交易日（没有则 None）。"""

    frame = store.coverage()
    row = frame[frame["table"] == table]
    if row.empty:
        return None
    value = row.iloc[0]["last"]
    return None if value is None or pd.isna(value) else str(value)[:10]


def cmd_daily_job(config: AppConfig, args: argparse.Namespace) -> int:
    """每日归档任务 —— **一次跑齐三类数据，各步互不拖累**。

    三步的"可恢复性"完全不同，所以失败处理也必须不同：

    | 步骤 | 数据档 | 今天没跑成 |
    | :--- | :--- | :--- |
    | 行情推进 | T0/T1 | 明天增量窗口能自动补回 |
    | 题材归因 | T1（按 date 可查） | 明天按日期区间补抓即可 |
    | T2 快照 | **只有当日** | **永久缺失，补不回来** |

    ⇒ 所以即使行情和题材都失败，也要**继续尝试** T2 快照；反过来 T2 失败
    也绝不能中断其它步骤。**结尾必须逐项汇报**，不能用一句"完成"盖过去 ——
    "哪一步悄悄没跑"正是这类任务最容易出的问题。

    另外：as-of 取行情归档里最近一个**完整交易日**，不用系统日期。
    收盘前退回上一个交易日，收盘后才使用当天；周末/节假日使用上一个交易日，
    避免把盘中行情与收盘后数据混在同一份推荐里。
    """

    custom_request = any(
        getattr(args, name, None)
        for name in ("spec", "spec_file", "theme", "out", "forward_dir")
    )
    if config.fundamentals.enabled and not custom_request:
        return _registered_daily_job(config, args)

    report: list[str] = []
    ok = True

    # -- 1. 行情推进 -------------------------------------------------------- #
    with MarketStore(config.market.daily_parquet) as store:
        if not store.ready:
            print(
                f"❌ 行情归档不存在：{store.path}。先跑 `recommend sync-bars` 引导一次。",
                file=sys.stderr,
            )
            return 1
        advance = advance_bars(store, workers=args.workers)
        print(advance.summary())
        as_of = store.latest_complete_trade_date()
        trade_dates = set(store.trade_dates(end=as_of))
    report.append(f"行情：{advance.summary().splitlines()[0]}")
    # 抓取失败不判为致命：行情是 T0/T1，明天增量窗口会补回来。
    if advance.up_to_date or advance.rows:
        pass
    else:
        ok = False

    # -- 2. 题材归因（按日期区间补齐到 as-of） ------------------------------- #
    if args.skip_themes:
        report.append("题材：已跳过（--skip-themes）")
    else:
        store = _store(config)
        client = _client(config)
        try:
            last_theme = _latest_archived_date(store, "theme_attribution")
            # 只补**行情归档认定为交易日**的那些天：不会去试探节假日，
            # 也不会因为主题归档落后而漏补（区间从它自己最后一天之后开始）。
            pending = sorted(
                day
                for day in trade_dates
                if day <= as_of and (last_theme is None or day > last_theme)
            )
            if not pending:
                report.append(f"题材：已是最新（止于 {last_theme}）")
            else:
                started = pending[0]
                print(f"补抓题材 {started} → {as_of}（{len(pending)} 天）")
                result = backfill_themes(store, client, started, as_of)
                print(result.summary())
                report.append(
                    f"题材：{started} → {as_of} 成功 {result.ok} / 写入 {result.rows} 行"
                )
                if result.failed or result.rejected:
                    ok = False
        finally:
            store.close()

    # -- 3. T2 快照（只有当日，补不回来） ----------------------------------- #
    if args.skip_snapshot:
        report.append("T2 快照：已跳过（--skip-snapshot）")
    elif not is_current_eod_ready(as_of):
        report.append(
            f"T2 快照：延后（{as_of} 不是当前已收盘日；请下一个交易日收盘后再同步）"
        )
    else:
        store = _store(config)
        client = _client(config)
        try:
            existing = store.conn.execute(
                "select count(*) from sector_snapshot where trade_date = ?",
                [date.fromisoformat(as_of)],
            ).fetchone()[0]
            if existing:
                report.append(f"T2 快照：{as_of} 已有 {existing} 行，跳过")
            else:
                results = snapshot_all(store, client, as_of)
                for result in results:
                    print(result.summary())
                failed = any(result.failed for result in results)
                report.append(
                    f"T2 快照：{'❌ 失败' if failed else '✅ 完成'}"
                    f"（as-of {as_of}；**只有当日可取，失败即永久缺失**）"
                )
                # T2 失败**不**把整次任务判为失败 —— 否则调度器会一直重试，
                # 而重试并不能把已经过去的那一天补回来。
        finally:
            store.close()

    # -- 4. 筛选 + 留证（前向纸盘的核心产出） ------------------------------- #
    if args.no_screen:
        report.append("筛选留证：已跳过（--no-screen）")
    else:
        print()
        code = _screen_and_record(config, args, as_of)
        if code == 0:
            report.append(f"筛选留证：as-of {as_of} 已落盘（spec 指纹见上）")
        else:
            ok = False
            report.append(f"筛选留证：❌ 失败（退出码 {code}）")

    print()
    print("=== 每日归档任务汇总 ===")
    for line in report:
        print(f"  · {line}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recommend", description="A 股推荐系统 · 归档层")
    parser.add_argument("--config", default=None, help="TOML 配置路径")
    parser.add_argument("--verbose", action="store_true", help="打印调试日志")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="通路诊断（区分被封与端点问题）").set_defaults(
        func=cmd_probe
    )

    backfill = sub.add_parser("backfill", help="T1 历史回补")
    backfill.add_argument("source", choices=["themes", "dragon-tiger", "margin"])
    backfill.add_argument("--start", default=None)
    backfill.add_argument("--end", default=None)
    backfill.add_argument("--force", action="store_true", help="重抓已归档的日期")
    backfill.add_argument("--symbols", default=None, help="两融：逗号分隔的代码")
    backfill.add_argument("--from-themes", action="store_true", help="两融：用题材归档里的标的")
    backfill.add_argument("--limit", type=int, default=200, help="--from-themes 的标的数上限")
    backfill.add_argument("--page-size", type=int, default=60)
    backfill.add_argument(
        "--legacy-per-symbol",
        action="store_true",
        help="两融：使用旧版东财逐标的接口；默认使用交易所官方全市场接口",
    )
    backfill.set_defaults(func=cmd_backfill)

    snapshot = sub.add_parser("snapshot", help="T2 当日快照（不可回补）")
    snapshot.add_argument("--date", default=None)
    snapshot.set_defaults(func=cmd_snapshot)

    sub.add_parser("status", help="归档覆盖与运行记录").set_defaults(func=cmd_status)

    sync_all = sub.add_parser(
        "sync-all", help="按统一注册表同步全部数据源，并重新生成推荐"
    )
    sync_all.add_argument("--preset", default="low-position", choices=sorted(PRESETS))
    sync_all.add_argument("--workers", type=int, default=8, help="行情抓取并发数")
    sync_all.set_defaults(func=cmd_sync_all)

    sub.add_parser("quality", help="自查有没有混进「回退快照」").set_defaults(func=cmd_quality)

    purge = sub.add_parser("purge", help="按日清除被污染的记录")
    purge.add_argument("--dates", required=True, help="逗号分隔的日期")
    purge.add_argument(
        "--table", default="theme_attribution", choices=["theme_attribution", "dragon_tiger"]
    )
    purge.set_defaults(func=cmd_purge)

    themes = sub.add_parser("themes", help="查询某日题材热度")
    themes.add_argument("--date", required=True)
    themes.add_argument("--limit", type=int, default=20)
    themes.set_defaults(func=cmd_themes)

    sync = sub.add_parser("sync-bars", help="从既有归档引导本项目日线（一次性）")
    sync.add_argument("--source", default=None, help="引导源 parquet（默认为配置项）")
    sync.set_defaults(func=cmd_sync_bars)

    sub.add_parser("fields", help="列出 spec 的字段白名单与预置").set_defaults(func=cmd_fields)

    screen = sub.add_parser("screen", help="确定性筛选（阶段 1，无 LLM）")
    screen.add_argument("--spec", default=None, help="内联 JSON spec")
    screen.add_argument("--spec-file", default=None, help="spec JSON 文件路径")
    screen.add_argument("--preset", default=None, choices=sorted(PRESETS), help="预置条件名")
    screen.add_argument("--theme", default=None, help="逗号分隔的题材（配合 --preset）")
    screen.add_argument("--as-of", default=None, help="as-of 交易日（默认取最近一个已收盘交易日）")
    screen.add_argument("--limit", type=int, default=None, help="覆盖 spec 的返回条数")
    screen.add_argument("--out", default=None, help="把 Markdown 报告写到该路径（同时落 CSV）")
    screen.set_defaults(func=cmd_screen)

    advance = sub.add_parser(
        "daily-advance", help="前向纸盘：推进行情 → 按冻结 spec 筛选 → 留证（定时任务用这个）"
    )
    advance.add_argument("--spec", default=None, help="内联 JSON spec")
    advance.add_argument(
        "--spec-file", default=None, help="spec JSON 文件（前向测试建议用冻结文件）"
    )
    advance.add_argument("--preset", default=None, choices=sorted(PRESETS), help="预置条件名")
    advance.add_argument("--theme", default=None, help="逗号分隔的题材（配合 --preset）")
    advance.add_argument("--out", default=None, help="报告路径（默认 data/reports/<as-of>.md）")
    advance.add_argument("--forward-dir", default=None, help="留证目录（默认 reports/forward）")
    advance.add_argument("--workers", type=int, default=8, help="行情抓取并发数")
    advance.add_argument("--no-screen", action="store_true", help="只推进行情，不筛选")
    advance.set_defaults(func=cmd_daily_advance)

    forward = sub.add_parser("forward-report", help="用实际后续走势给前向留证对账")
    forward.add_argument("--forward-dir", default=None, help="留证目录（默认 reports/forward）")
    forward.set_defaults(func=cmd_forward_report)

    job = sub.add_parser(
        "daily-job", help="每日归档任务：行情 + 题材 + T2 快照 + 筛选留证（调度器调这个）"
    )
    job.add_argument("--spec", default=None, help="内联 JSON spec")
    job.add_argument("--spec-file", default=None, help="spec JSON 文件（建议用冻结文件）")
    job.add_argument("--preset", default=None, choices=sorted(PRESETS), help="预置条件名")
    job.add_argument("--theme", default=None, help="逗号分隔的题材（配合 --preset）")
    job.add_argument("--out", default=None, help="报告路径（默认 data/reports/<as-of>.md）")
    job.add_argument("--forward-dir", default=None, help="留证目录（默认 reports/forward）")
    job.add_argument("--workers", type=int, default=8, help="行情抓取并发数")
    job.add_argument("--no-screen", action="store_true", help="跳过筛选留证")
    job.add_argument("--skip-themes", action="store_true", help="跳过题材补抓")
    job.add_argument("--skip-snapshot", action="store_true", help="跳过 T2 快照")
    job.set_defaults(func=cmd_daily_job)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = AppConfig.load(args.config)

    # 回补的默认起止日期来自配置，避免每次手敲。
    if args.command == "backfill":
        if args.source == "themes":
            args.start = args.start or config.backfill.themes_start
            args.end = args.end or _today()
        elif args.source == "dragon-tiger":
            args.start = args.start or config.backfill.dragon_tiger_start
            args.end = args.end or _today()
        elif args.source == "margin" and not (
            args.legacy_per_symbol or args.symbols or args.from_themes
        ):
            args.start = args.start or config.backfill.margin_start
            args.end = args.end or _today()
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    return int(args.func(config, args))


def _configure_console_encoding() -> None:
    """让 CLI 在中文 Windows 控制台也能打印完整诊断信息。

    Windows 传统控制台常把 stdout 设成 GBK；本项目的诊断包含中文、箭头和
    进度符号，直接 print 会在某些机器上抛 UnicodeEncodeError。交互式终端仍
    使用原来的流，只把编码切换到 UTF-8，并在重定向到旧终端时安全替换极少数
    无法表示的字符。
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _today() -> str:
    """**仅**用于回补区间的上界。给行情/结果盖章一律用归档里的交易日。"""

    return date.today().isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
