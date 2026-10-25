"""Freeze official price and total-return benchmarks after finalist selection."""

from __future__ import annotations

import json

from design_index_improvement import OUT
from qlib_external_sources import network_session
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import pandas as pd
    from prepare_market_weight_batch import sha256
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    assert (OUT / "finalists_locked.json").exists()
    folder = OUT / "validation_indices"
    folder.mkdir(exist_ok=True)
    session = network_session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.3)))
    rows, receipts = [], []
    for cohort, start, end in [
        ("external", "20181201", "20220630"),
        ("recent", "20260901", "20261015"),
    ]:
        for code in [
            "000300",
            "000905",
            "000852",
            "000985",
            "H00300",
            "H00905",
            "H00852",
            "H00985",
        ]:
            source = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
            params = {"indexCode": code, "startDate": start, "endDate": end}
            raw = folder / f"{cohort}_{code}.json"
            if not raw.exists():
                response = session.get(source, params=params, timeout=30)
                response.raise_for_status()
                assert response.json().get("data"), (cohort, code)
                raw.write_bytes(response.content)
            frame = pd.DataFrame(json.loads(raw.read_bytes())["data"])
            selected = pd.DataFrame(
                {
                    "day": pd.to_datetime(frame.tradeDate, format="%Y%m%d").dt.strftime("%Y-%m-%d"),
                    "code": code,
                    "open": pd.to_numeric(frame.open, errors="coerce"),
                    "close": pd.to_numeric(frame.close, errors="coerce"),
                    "cohort": cohort,
                }
            )
            assert not selected.day.duplicated().any() and selected.close.gt(0).all()
            rows.append(selected)
            receipts.append(
                {"source": source, "params": params, "sha256": sha256(raw), "rows": len(frame)}
            )
            print(cohort, code, len(frame), "official levels", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(folder / "levels.parquet", index=False)
    (folder / "receipt.json").write_text(
        json.dumps(
            {
                "files": receipts,
                "levels_sha256": sha256(folder / "levels.parquet"),
                "finalists_sha256": sha256(OUT / "finalists_locked.json"),
                "resource": RESOURCE_CONTEXT,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
