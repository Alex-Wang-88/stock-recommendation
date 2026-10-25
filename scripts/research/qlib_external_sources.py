"""Freeze upstream reference files and probe dated index membership without running repo code."""

from __future__ import annotations

import contextlib
import io
import json
import socket
from pathlib import Path

from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/qlib-comparison-20261015"


def network_session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "stock-recommendation-research"
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
            proxy = winreg.QueryValueEx(key, "ProxyServer")[0]
        if enabled and "=" not in proxy:
            session.proxies = {"http": "http://" + proxy, "https": "http://" + proxy}
    except (ImportError, OSError):
        pass
    return session


def fetch_membership(bs, pool, day):
    import pandas as pd

    folder = OUT / "membership"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{pool}_{day}.csv"
    if destination.exists():
        return pd.read_csv(destination, dtype=str)
    method = bs.query_hs300_stocks if pool == "csi300" else bs.query_zz500_stocks
    result = method(date=day)
    if result.error_code != "0":
        raise RuntimeError(f"{pool} {day}: {result.error_code} {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    frame = pd.DataFrame(rows, columns=result.fields)
    expected = 300 if pool == "csi300" else 500
    if len(frame) != expected or frame.code.duplicated().any():
        raise ValueError(f"Unexpected {pool} membership size {len(frame)} for {day}")
    if not frame.updateDate.le(day).all():
        raise ValueError(f"Future membership update returned for {day}")
    frame.to_csv(destination, index=False)
    return frame


def main():
    from prepare_market_weight_batch import sha256

    OUT.mkdir(exist_ok=True)
    folder = OUT / "upstream"
    folder.mkdir(exist_ok=True)
    session = network_session()
    receipt_path = folder / "receipt.json"
    if not receipt_path.exists():
        response = session.get(
            "https://api.github.com/repos/microsoft/qlib/commits/main", timeout=30
        )
        response.raise_for_status()
        commit = response.json()["sha"]
        paths = [
            "LICENSE",
            "qlib/contrib/data/loader.py",
            "qlib/contrib/data/handler.py",
            "qlib/data/ops.py",
            "qlib/contrib/model/gbdt.py",
            "examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml",
            "examples/benchmarks/Linear/workflow_config_linear_Alpha158.yaml",
        ]
        receipts = []
        for path in paths:
            url = f"https://raw.githubusercontent.com/microsoft/qlib/{commit}/{path}"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            saved = folder / path.replace("/", "__")
            saved.write_bytes(response.content)
            receipts.append({"url": url, "file": saved.name, "sha256": sha256(saved)})
        receipt_path.write_text(
            json.dumps({"commit": commit, "files": receipts}, indent=2), encoding="utf-8"
        )
        print("Qlib upstream frozen at", commit, flush=True)
    vnpy_receipt = folder / "vnpy_receipt.json"
    if not vnpy_receipt.exists():
        response = session.get("https://api.github.com/repos/vnpy/vnpy/commits/master", timeout=30)
        response.raise_for_status()
        vnpy_commit = response.json()["sha"]
        vnpy_files = []
        for path in ["LICENSE", "vnpy/alpha/model/models/lasso_model.py"]:
            url = f"https://raw.githubusercontent.com/vnpy/vnpy/{vnpy_commit}/{path}"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            saved = folder / ("vnpy__" + path.replace("/", "__"))
            saved.write_bytes(response.content)
            vnpy_files.append({"url": url, "file": saved.name, "sha256": sha256(saved)})
        vnpy_receipt.write_text(
            json.dumps({"commit": vnpy_commit, "files": vnpy_files}, indent=2), encoding="utf-8"
        )
    import baostock as bs

    socket.setdefaulttimeout(20)
    with contextlib.redirect_stdout(io.StringIO()):
        result = bs.login()
    assert result.error_code == "0", result.error_msg
    proofs = []
    try:
        for pool in ["csi300", "csi500"]:
            for day in ["2023-04-03", "2024-01-31", "2026-08-13"]:
                frame = fetch_membership(bs, pool, day)
                row = {
                    "pool": pool,
                    "requested": day,
                    "rows": len(frame),
                    "update_min": frame.updateDate.min(),
                    "update_max": frame.updateDate.max(),
                    "symbols": frame.code.tolist(),
                }
                proofs.append(row)
                print({k: v for k, v in row.items() if k != "symbols"}, flush=True)
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            bs.logout()
    (OUT / "membership_probe.json").write_text(json.dumps(proofs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
