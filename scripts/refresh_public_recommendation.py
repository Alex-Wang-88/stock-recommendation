"""Rebuild the published recommendation from saved data without network requests."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommend.bars import local_now  # noqa: E402
from recommend.config import AppConfig  # noqa: E402
from recommend.context import ScreenContext  # noqa: E402
from recommend.forward import evaluate_and_persist, spec_fingerprint  # noqa: E402
from recommend.locking import sync_lock  # noqa: E402
from recommend.market import MarketStore, is_closed_eod_ready  # noqa: E402
from recommend.report import render_markdown  # noqa: E402
from recommend.screen import Spec, run_screen  # noqa: E402
from recommend.screen.presets import load_preset  # noqa: E402

SYNC_WARNING = (
    "本次上游同步存在警告或未完成；推荐使用已保存的数据，"
    "缺失项见数据说明和 Actions 日志。"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="low-position")
    parser.add_argument("--sync-exit-code", default="")
    args = parser.parse_args()
    config = AppConfig.load()
    with sync_lock(Path(config.logs_dir) / "sync.lock"):
        with MarketStore(config.market.daily_parquet) as market:
            as_of = market.latest_complete_trade_date()
        if not is_closed_eod_ready(as_of):
            raise RuntimeError(f"No completed market session available: {as_of}")
        spec = Spec.from_dict({**load_preset(args.preset), "as_of": as_of})
        with ScreenContext.open(config, as_of) as context:
            table = context.build()
            screened = run_screen(table, replace(spec, as_of=table.as_of), config.rank.as_dict())
        stem = Path(config.reports_dir) / f"{screened.as_of}__{spec_fingerprint(screened.spec)}"
        markdown_path = stem.with_suffix(".md")
        previous_warning = (
            not args.sync_exit_code
            and markdown_path.exists()
            and SYNC_WARNING in markdown_path.read_text(encoding="utf-8")
        )
        if args.sync_exit_code not in ("", "0") or previous_warning:
            screened.notes.insert(0, SYNC_WARNING)
        stem.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_markdown(
                screened,
                spec_hash=spec_fingerprint(screened.spec),
                generated_at=local_now(),
            ),
            encoding="utf-8",
        )
        screened.frame.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        # Publishing saved data must not create or replace a frozen forward record.
        with MarketStore(config.market.daily_parquet) as market:
            evaluate_and_persist(market, Path(config.reports_dir) / "forward")
    print(f"Saved recommendation refreshed: as_of={screened.as_of}, candidates={screened.kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
