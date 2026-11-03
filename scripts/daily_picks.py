"""Run the pure-code daily strategy and build its static web product."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommend.resources import background_limits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-only", action="store_true", help="No network or forward backfill")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--freeze-saved",
        action="store_true",
        help="Freeze timely saved EOD data before the next open; never backfill",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    workers = background_limits(args.workers)
    from recommend.daily_job import public_daily_payload, run_daily

    payload = run_daily(
        ROOT, sync=not args.saved_only, workers=workers, freeze_saved=args.freeze_saved
    )
    public_daily_payload(ROOT / "data", ROOT / "site", args.output_dir)
    # Failed freshness is operational failure even though an honest historical page was built.
    return 0 if args.saved_only or payload["state"] == "live" else 1


if __name__ == "__main__":
    raise SystemExit(main())
