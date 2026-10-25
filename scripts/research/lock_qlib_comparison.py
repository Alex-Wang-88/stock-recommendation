"""Lock two A-share pools, three scorers and chronological fitting before reading results."""

from __future__ import annotations

import contextlib
import io
import json
import socket

from qlib_external_sources import OUT, ROOT, fetch_membership
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import baostock as bs
    import pandas as pd
    from prepare_market_weight_batch import sha256

    source = ROOT / "logs/market-direction-study-20261015/windows_locked.json"
    windows = json.loads(source.read_text(encoding="utf-8"))["windows"]
    levels = pd.read_parquet(
        ROOT / "logs/market-rule-optimization-20261014/index_data/levels.parquet"
    )
    calendar = sorted(levels.loc[levels.code == "000985", "day"].tolist())
    signals = sorted({s for w in windows for s in w["signal_dates"]})
    sample_days = sorted(set(calendar[65::5]) | set(signals))
    folds = []
    for window in windows:
        signal = window["signal_dates"][0]
        t = calendar.index(signal)
        valid_start = t - 69
        train_end = valid_start - 7
        train_start = max(65, train_end - 251)
        assert train_end - train_start >= 100
        folds.append(
            {
                "window": window["label"],
                "test_first_signal": signal,
                "train_start": calendar[train_start],
                "train_end": calendar[train_end],
                "valid_start": calendar[valid_start],
                "valid_end": calendar[t - 7],
                "test_signals": window["signal_dates"],
            }
        )
    previous = json.loads(
        (ROOT / "config/research/market_direction_candidates.json").read_text(encoding="utf-8")
    )
    config = {
        "version": 1,
        "date": "2026-10-15",
        "role": "previously_viewed_development_comparison",
        "upstream_commit": json.loads((OUT / "upstream/receipt.json").read_text(encoding="utf-8"))[
            "commit"
        ],
        "market_sha256": sha256(
            ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet"
        ),
        "pools": ["csi300", "csi500"],
        "scorers": ["P4", "Lasso", "LightGBM"],
        "features": "Qlib Alpha158 without VWAP0 (157 features); no substituted VWAP",
        "train_sampling": "every fifth trading day from calendar index 65, plus evaluation signals",
        "training_eligibility": (
            "dated index member, at least 60 observed bars, positive volume, "
            "20d amount >=50m, finite open-to-open label; ST status not used for fitting"
        ),
        "inference_eligibility": (
            "same old shared market-only pool intersected with dated index membership; "
            "additional ML feature missing values are imputed without removing symbols"
        ),
        "label": (
            "open[t+6]/open[t+1]-1 (5 trading days), "
            "cross-section zscore on each dated training pool"
        ),
        "boundary_rule": (
            "label exit date strictly before next segment start; no fold updates inside 30d window"
        ),
        "preprocessing": (
            "training-only medians and median absolute deviations times 1.4826; "
            "clip +/-3, missing scaled values to zero; same transformation for Lasso and LightGBM"
        ),
        "lasso": {
            "alpha": 0.0005,
            "max_iter": 10000,
            "fit_intercept": False,
            "selection": "cyclic",
        },
        "lightgbm": {
            "objective": "regression",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.2,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 1,
            "verbosity": -1,
            "seed": 42,
            "deterministic": True,
            "force_col_wise": True,
        },
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
        "portfolio": {
            "positions": 25,
            "retain_rank": 40,
            "rebalance_days": 5,
            "weighting": "equal",
        },
        "exposure_modes": {"full": None, "vol15": 0.15},
        "cost_bps": [0, 10, 20],
        "common": {**previous["common"], "volume_trend_20_min": 1.1},
        "sample_days": sample_days,
        "calendar": calendar,
        "windows": windows,
        "folds": folds,
        "fresh_holdout": False,
        "no_hyperparameter_search": True,
        "adaptations": [
            "No VWAP: adjusted-price versus raw amount/volume consistency not established",
            "Short local training history, 5-day next-open label "
            "rather than original 1-day close label",
            "Shared 25-stock buffered 5-day execution "
            "rather than original daily 50-stock TopkDropout",
            "Historical index snapshots are weekly provider updates; "
            "not an official daily membership audit",
            "Lasso is fit on training only rather than vnpy example merging train and validation",
        ],
    }
    path = ROOT / "config/research/qlib_comparison.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == config, "Plan changed after lock"
    else:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    locked = OUT / "plan_locked.json"
    if not locked.exists():
        locked.write_bytes(path.read_bytes())
    assert sha256(path) == sha256(locked)
    print(f"Locked {len(windows)} windows, {len(sample_days)} sample days, two pools", flush=True)
    socket.setdefaulttimeout(20)
    with contextlib.redirect_stdout(io.StringIO()):
        result = bs.login()
    assert result.error_code == "0"
    frames = []
    try:
        for i, day in enumerate(sample_days):
            for pool in config["pools"]:
                frame = fetch_membership(bs, pool, day)
                frame["day"], frame["pool"] = day, pool
                frame["symbol"] = frame.code.str.split(".").str[-1]
                frames.append(frame[["day", "pool", "symbol", "updateDate"]])
            if i % 20 == 0:
                print(f"Membership snapshots {i + 1}/{len(sample_days)}", flush=True)
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            bs.logout()
    members = pd.concat(frames, ignore_index=True)
    members.to_parquet(OUT / "membership.parquet", index=False)
    (OUT / "plan_receipt.json").write_text(
        json.dumps(
            {
                "plan_sha256": sha256(locked),
                "membership_sha256": sha256(OUT / "membership.parquet"),
                "resource": RESOURCE_CONTEXT,
                "windows_source_sha256": sha256(source),
                "membership_source": "BaoStock dated query_hs300_stocks / query_zz500_stocks",
                "membership_rows": len(members),
                "unique_symbols": members.symbol.nunique(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Membership collection complete", len(members), "rows", flush=True)


if __name__ == "__main__":
    main()
