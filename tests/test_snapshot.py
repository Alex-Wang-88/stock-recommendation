"""T2 板块快照的幂等与失败口径测试。"""

from __future__ import annotations

import pandas as pd

from recommend.archive import snapshot as snapshot_module
from recommend.archive.snapshot import snapshot_all
from recommend.archive.sources import SECTOR_COLUMNS
from recommend.archive.store import ArchiveStore


def test_existing_sector_snapshot_skips_connectivity_probe(tmp_path, monkeypatch):
    """已有当日快照时，后续重复运行不应因网络探测失败而报警。"""

    archive_path = tmp_path / "archive.duckdb"
    with ArchiveStore(archive_path) as store:
        frame = pd.DataFrame(
            [
                {
                    "trade_date": "2026-09-21",
                    "sector_code": f"BK{index:04d}",
                    "sector_name": f"测试行业{index}",
                    "change_pct": 1.0,
                    "main_net_inflow": 100.0,
                    "up_count": 10,
                    "down_count": 2,
                    "fetched_at": pd.Timestamp("2026-09-21 02:00:00"),
                }
                for index in range(20)
            ]
        )
        store.upsert(
            "sector_snapshot",
            frame,
            ("trade_date", *SECTOR_COLUMNS, "fetched_at"),
        )

        def fail_if_probed(_client):
            raise AssertionError("已有快照不应再次探测网络")

        monkeypatch.setattr(snapshot_module, "check_connectivity", fail_if_probed)
        results = snapshot_all(store, None, "2026-09-21")  # type: ignore[arg-type]

        assert len(results) == 1
        assert results[0].failed == 0
        assert results[0].skipped == 1
        assert store.conn.execute(
            "select count(*) from sector_snapshot where trade_date = date '2026-09-21'"
        ).fetchone()[0] == 20
