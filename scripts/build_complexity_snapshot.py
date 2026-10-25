"""Refresh small public daily series and build a file:// compatible site snapshot."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def configure_resources() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if os.name == "nt":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
        handle = kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(handle, 0x4000) or kernel.GetPriorityClass(handle) != 0x4000:
            raise ctypes.WinError(ctypes.get_last_error())


def main() -> int:
    configure_resources()
    from recommend.complexity import (
        FACTORS,
        SECTORS,
        build_snapshot,
        download_source,
        seed_cache,
        write_snapshot,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "site")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "complexity")
    args = parser.parse_args()
    seed_cache(args.cache_dir, ROOT / "site" / "complexity-data.json")
    now = datetime.now(UTC)
    sources = []
    for spec in (
        *FACTORS,
        *(
            {**s, "provider": "腾讯财经", "kind": "etf", "unit": "CNY", "zone": "Asia/Shanghai"}
            for s in SECTORS
        ),
    ):
        source = download_source(spec, now, args.cache_dir)
        sources.append(source)
        print(f"{spec['id']}: {source['status']} / {len(source['observations'])} rows", flush=True)
    payload = build_snapshot(sources, now)
    write_snapshot(payload, args.output_dir)
    print(f"Snapshot: basket={payload['basket']['as_of']}, warnings={len(payload['warnings'])}")
    # Partial sources still yield an honest usable snapshot, without invented values.
    return 0 if any(s["observations"] for s in sources) else 1


if __name__ == "__main__":
    raise SystemExit(main())
