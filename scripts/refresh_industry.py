"""刷新项目内的个股行业快照。

推荐计算不在运行时请求行业接口，而是只读取
``data/reference/industry_master.csv``。这样把整个目录复制到另一台 Windows
电脑后，页面仍然可以离线展示并复现当前推荐口径。

默认源是一个公开、无需 API Key 的 A 股基础信息快照；它只负责给展示层提供
``symbol -> industry`` 映射，绝不进入排序权重。抓取失败时保留现有本地文件。

用法：
    .venv\\Scripts\\python.exe scripts\\refresh_industry.py
    .venv\\Scripts\\python.exe scripts\\refresh_industry.py --dry-run
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

DEFAULT_URL = (
    "https://raw.githubusercontent.com/whz06/deep-learning2026final-lab/"
    "main/train_li/data/basic.csv"
)
DEFAULT_TARGET = Path("data/reference/industry_master.csv")


def fetch_industry(url: str, timeout: float = 30.0) -> pd.DataFrame:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content), dtype=str)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    if "symbol" not in columns or "industry" not in columns:
        raise ValueError("行业源必须包含 symbol 和 industry 两列")

    out = frame.loc[:, [columns["symbol"], columns["industry"]]].copy()
    out.columns = ["symbol", "industry"]
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    out["industry"] = out["industry"].astype("string").str.strip()
    out = out.loc[out["symbol"].notna() & out["industry"].notna()]
    out = out.loc[out["industry"] != ""]
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    if out.empty:
        raise ValueError("行业源没有可用的 symbol / industry 映射")
    return out.sort_values("symbol").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="刷新本地个股行业快照")
    parser.add_argument("--url", default=DEFAULT_URL, help="公开 CSV 行业源")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="项目内输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不写入")
    args = parser.parse_args(argv)

    target = Path(args.target)
    frame = fetch_industry(args.url)
    print(f"源：{args.url}")
    print(f"可用行业映射：{len(frame):,} 只")
    print(f"目标：{target}")
    if args.dry_run:
        print("（--dry-run）未写入本地文件。")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(target)
    print(f"已写入：{target}")
    print("行业只用于展示，不参与筛选、排序或综合分。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
