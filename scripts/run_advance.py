"""手动跑一次行情增量推进（也是定时任务调用的入口）。

用法：
    .venv/Scripts/python.exe scripts/run_advance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommend.bars import advance_bars  # noqa: E402
from recommend.config import AppConfig  # noqa: E402
from recommend.market import MarketStore  # noqa: E402


def main() -> int:
    config = AppConfig.load(None)
    started = time.time()
    with MarketStore(config.market.daily_parquet) as store:
        result = advance_bars(store)
        print(result.summary(), flush=True)
        if not result.up_to_date and result.rows:
            print(f"归档现在止于 {store.latest_trade_date()}", flush=True)
    print(f"总耗时 {time.time() - started:.0f}s", flush=True)
    return 1 if result.failed and not result.rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
