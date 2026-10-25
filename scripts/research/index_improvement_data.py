"""Bounded common data loading and expansion of already-frozen model predictions."""

from __future__ import annotations

import json

from design_index_improvement import OUT, PREVIOUS, ROOT


def load_development():
    import duckdb
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import READ_COLUMNS, sha256
    from qlib_alpha_features import transform

    config = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    members = pd.read_parquet(PREVIOUS / "membership.parquet")
    signals = sorted({s for w in config["windows"] for s in w["signal_dates"]})
    members = members.loc[members.day.isin(signals)].copy()
    frozen_plan = json.loads((PREVIOUS / "plan_locked.json").read_text(encoding="utf-8"))
    fit_qa = json.loads((PREVIOUS / "fit_qa.json").read_text(encoding="utf-8"))
    assert sha256(PREVIOUS / "predictions.parquet") == fit_qa["prediction_sha256"]
    old_predictions = pd.read_parquet(PREVIOUS / "predictions.parquet")
    expanded_path = OUT / "development_predictions.parquet"
    if expanded_path.exists():
        expansion_qa = json.loads(
            (OUT / "prediction_expansion_qa.json").read_text(encoding="utf-8")
        )
        assert sha256(expanded_path) == expansion_qa["predictions_sha256"]
        predictions = pd.read_parquet(expanded_path)
    else:
        feature_receipt = json.loads(
            (PREVIOUS / "feature_receipt.json").read_text(encoding="utf-8")
        )
        feature_names = feature_receipt["feature_names"]
        feature_frame = pd.read_parquet(
            PREVIOUS / "alpha157.parquet",
            columns=["day", "symbol", *feature_names],
            filters=[("day", "in", signals)],
        )
        pieces = []
        for pool in config["pools"]:
            for fold in frozen_plan["folds"]:
                tag = pool + "_" + fold["window"]
                frame = members.loc[
                    (members.pool == pool) & members.day.isin(fold["test_signals"])
                ].merge(feature_frame, on=["day", "symbol"], validate="one_to_one")
                parameters = json.loads(
                    (PREVIOUS / "models" / (tag + "_transformer.json")).read_text(encoding="utf-8")
                )
                assert parameters["features"] == feature_names
                x = transform(
                    frame[feature_names].to_numpy(dtype=np.float32),
                    np.asarray(parameters["medians"], dtype=np.float32),
                    np.asarray(parameters["scale"], dtype=np.float32),
                )
                prediction = frame[["day", "symbol", "pool"]].copy()
                prediction["window"] = fold["window"]
                prediction["Lasso"] = x @ np.asarray(
                    parameters["lasso_coefficients"], dtype=np.float32
                )
                model = lgb.Booster(model_file=str(PREVIOUS / "models" / (tag + "_lightgbm.txt")))
                prediction["LightGBM"] = model.predict(x, num_threads=1)
                pieces.append(prediction)
        predictions = pd.concat(pieces, ignore_index=True)
        parity = predictions.merge(
            old_predictions,
            on=["day", "symbol", "pool", "window"],
            suffixes=("", "_old"),
            validate="one_to_one",
        )
        errors = {
            name: float((parity[name] - parity[name + "_old"]).abs().max())
            for name in ["Lasso", "LightGBM"]
        }
        assert len(parity) == len(old_predictions)
        assert max(errors.values()) < 1e-5
        # Preserve exactly the original known scores for literal old-account parity.
        keys = ["day", "symbol", "pool", "window"]
        predictions = predictions.set_index(keys)
        predictions.update(old_predictions.set_index(keys)[["Lasso", "LightGBM"]])
        predictions = predictions.reset_index()
        predictions.to_parquet(expanded_path, index=False)
        (OUT / "prediction_expansion_qa.json").write_text(
            json.dumps(
                {
                    "rows": len(predictions),
                    "old_rows": len(parity),
                    "max_errors": errors,
                    "new_model_fits": 0,
                    "future_labels_read_for_scoring": False,
                    "predictions_sha256": sha256(expanded_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    factors = pd.read_parquet(
        ROOT / "logs/market-direction-study-20261015/research_factors.parquet",
        columns=list(READ_COLUMNS),
    )
    factors["symbol"] = factors.symbol.astype(str).str.zfill(6)
    factors["day"] = factors.day.astype(str).str[:10]
    factors = factors.merge(
        members[["day", "symbol", "pool"]], on=["day", "symbol"], validate="one_to_one"
    )
    days = sorted({d for w in config["windows"] for d in w["days"] + w["signal_dates"]})
    symbols = sorted(members.symbol.unique())
    status = pd.read_parquet(
        ROOT / "logs/market-direction-study-20261015/execution_status_research.parquet"
    )
    status["symbol"] = status.symbol.astype(str).str.zfill(6)
    status = status.loc[status.symbol.isin(symbols)].copy()
    market = ROOT / "logs/historical-weight-validation-20261014/inputs/market.parquet"
    assert sha256(market) == config["frozen_market_sha256"]
    con = duckdb.connect(config={"threads": 1, "memory_limit": "512MB"})
    con.register("wanted_days", pd.DataFrame({"day": days}))
    con.register("wanted_symbols", pd.DataFrame({"symbol": symbols}))
    prices = con.execute(
        "select b.symbol,cast(b.trade_date as date)::varchar as day,"
        "b.open,b.high,b.low,b.close,b.pre_close,b.volume "
        "from read_parquet(?) b join wanted_days d "
        "on cast(b.trade_date as date)=cast(d.day as date) join wanted_symbols using(symbol)",
        [market.as_posix()],
    ).df()
    con.close()
    levels = pd.read_parquet(
        ROOT / "logs/market-rule-optimization-20261014/index_data/levels.parquet"
    )
    return config, factors, predictions, prices, status, levels


def price_arrays(prices, status):
    import numpy as np

    assert not prices.duplicated(["day", "symbol"]).any()
    grid = prices.pivot(index="day", columns="symbol", values="close")
    symbols, days = grid.columns.tolist(), grid.index.tolist()
    arrays = {}
    for field in ["open", "high", "low", "close", "volume", "pre_close"]:
        arrays["prior_close" if field == "pre_close" else field] = (
            prices.pivot(index="day", columns="symbol", values=field)
            .reindex(index=days, columns=symbols)
            .to_numpy(dtype=float, na_value=np.nan)
        )
    for target, column, default in [("tradable", "tradestatus", 0), ("is_st", "isST", 1)]:
        arrays[target] = (
            status.pivot(index="day", columns="symbol", values=column)
            .reindex(index=days, columns=symbols)
            .fillna(default)
            .to_numpy()
            == 1
        )
    arrays["growth_board"] = np.array([s.startswith(("30", "688", "689")) for s in symbols])
    return arrays, {s: i for i, s in enumerate(symbols)}, {d: i for i, d in enumerate(days)}


def historical_growth_flags(symbols, start, end, *, legacy_codes=False):
    import numpy as np

    if start < "2020-08-24" <= end:
        raise ValueError("A fixed per-window flag cannot cross the ChiNext rule change")
    chinext_prefixes = ("300", "301") if legacy_codes else ("30",)
    return np.array(
        [
            s.startswith(("688", "689"))
            or (s.startswith(chinext_prefixes) and start >= "2020-08-24")
            for s in symbols
        ]
    )
