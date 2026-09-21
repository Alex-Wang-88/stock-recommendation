"""刷新本项目的参考数据（名称 / 上市日表）。

为什么需要这个脚本
----------------
`instrument_master` 被**每次筛选**读取，用于 ST 剔除、次新筛选和展示名称。
它一旦陈旧，症状不是报错，而是**筛选结果悄悄变样** —— 新上市的票没有上市日
（被 `listed_days >= N` 条件判为缺失而剔除）、后来被 ST 的票名称还是旧的
（ST 剔除失效）。

所以：它是本项目自己的副本（不指回量化项目），但**必须能刷新**。
副本 + 无刷新路径 = 慢慢腐烂的静默失效。

用法：
    .venv/Scripts/python.exe scripts/refresh_reference.py
    .venv/Scripts/python.exe scripts/refresh_reference.py --from <路径> --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommend.config import AppConfig  # noqa: E402

# 项目内原始种子（只在**用户主动刷新**时访问；每日任务不读它）。
DEFAULT_SOURCE = "data/reference/source_instrument_master.parquet"


def describe(path: Path) -> tuple[int, str | None]:
    """返回 `(行数, 最晚上市日)`。读不动就返回 `(0, None)`。"""

    try:
        import duckdb

        connection = duckdb.connect()
        try:
            rows = connection.execute(
                f"select count(*), max(cast(listed_date as timestamp)) "
                f"from read_parquet('{path.as_posix()}')"
            ).fetchone()
        finally:
            connection.close()
    except Exception as error:  # noqa: BLE001 - 诊断脚本，读不动就说读不动
        print(f"  （无法读取 {path.name}：{error}）", file=sys.stderr)
        return 0, None
    return int(rows[0]), (str(rows[1])[:10] if rows[1] else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="刷新参考数据（名称 / 上市日表）")
    parser.add_argument("--from", dest="source", default=DEFAULT_SOURCE, help="上游路径")
    parser.add_argument("--dry-run", action="store_true", help="只比较，不写入")
    args = parser.parse_args(argv)

    config = AppConfig.load(None)
    target = Path(config.market.instrument_parquet)
    source = Path(args.source)

    if not source.exists():
        print(f"❌ 上游不存在：{source}", file=sys.stderr)
        print("   若已不再需要刷新，保持现有副本即可 —— 每日任务不读上游。")
        return 1

    old_rows, old_last = describe(target) if target.exists() else (0, None)
    new_rows, new_last = describe(source)

    print(f"上游 {source}")
    print(f"  {new_rows:,} 行，最晚上市日 {new_last}")
    print(f"当前 {target}")
    print(f"  {old_rows:,} 行，最晚上市日 {old_last}" if target.exists() else "  （不存在）")

    if target.exists() and (old_rows, old_last) == (new_rows, new_last):
        print("\n✅ 无变化，无需刷新。")
        return 0

    if args.dry_run:
        print(f"\n（--dry-run）将覆盖 {target}，新增 {new_rows - old_rows:,} 行。")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)  # 原子替换：中途被杀不会留下半个文件
    print(f"\n✅ 已刷新 {target}（{new_rows - old_rows:+,} 行）")
    print("⚠️ 注意：名称含 ST 标记、上市日均为**上游快照**，回看历史仍会含未来信息。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
