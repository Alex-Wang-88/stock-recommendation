"""Fit frozen public ML baselines on past labels, then freeze all 15-window predictions."""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path

from qlib_external_sources import OUT, ROOT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import READ_COLUMNS, pool_frame, sha256
    from qlib_alpha_features import fit_transformer, purged_fold_masks, transform
    from sklearn.linear_model import Lasso

    started = time.monotonic()
    config = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    plan_sha = sha256(OUT / "plan_locked.json")
    assert plan_sha == sha256(ROOT / "config/research/qlib_comparison.json")
    feature_receipt = json.loads((OUT / "feature_receipt.json").read_text(encoding="utf-8"))
    assert sha256(OUT / "alpha157.parquet") == feature_receipt["sha256"]
    feature_names = feature_receipt["feature_names"]
    features = pd.read_parquet(OUT / "alpha157.parquet")
    assert not features.duplicated(["day", "symbol"]).any()
    members = pd.read_parquet(OUT / "membership.parquet")
    historical = pd.read_parquet(
        ROOT / "logs/market-direction-study-20261015/research_factors.parquet",
        columns=list(READ_COLUMNS),
    )
    historical["symbol"] = historical.symbol.astype(str).str.zfill(6)
    historical["day"] = historical.day.astype(str).str[:10]
    model_dir = OUT / "models"
    model_dir.mkdir(exist_ok=True)
    receipts, predictions, diagnostics = [], [], []
    for pool in config["pools"]:
        joined = members.loc[members.pool == pool].merge(
            features, on=["day", "symbol"], how="left", validate="one_to_one"
        )
        learning = joined.loc[joined.fit_eligible.eq(True)].copy()  # fitting only
        group = learning.groupby("day").label_raw
        mu, sd = group.transform("mean"), group.transform(lambda x: x.std(ddof=0))
        learning["label"] = (learning.label_raw - mu) / sd
        learning = learning.loc[np.isfinite(learning.label)].copy()
        inference = {}
        for signal in sorted({s for w in config["windows"] for s in w["signal_dates"]}):
            frame, _ = pool_frame(
                historical.loc[historical.day == signal], "broader_liquid", config["common"]
            )
            names = members.loc[(members.pool == pool) & (members.day == signal), "symbol"]
            frame = frame.loc[frame.symbol.isin(names)].copy()
            assert len(frame) > 0, (pool, signal)
            inference[signal] = frame.merge(
                joined.loc[joined.day == signal, ["symbol", *feature_names, "label_raw"]],
                on="symbol",
                how="left",
                validate="one_to_one",
            )
        for fold in config["folds"]:
            tag = f"{pool}_{fold['window']}"
            checkpoint = model_dir / (tag + "_predictions.parquet")
            metadata_path = model_dir / (tag + "_receipt.json")
            if checkpoint.exists() and metadata_path.exists():
                prior = json.loads(metadata_path.read_text(encoding="utf-8"))
                assert prior["plan_sha256"] == plan_sha
                assert prior["feature_sha256"] == feature_receipt["sha256"]
                assert prior["training_code_sha256"] == sha256(Path(__file__))
                receipts.append(prior)
                predictions.append(pd.read_parquet(checkpoint))
                continue
            train_mask, valid_mask = purged_fold_masks(learning.day, learning.label_end, fold)
            train, valid = learning.loc[train_mask], learning.loc[valid_mask]
            assert len(train) >= 500 and len(valid) >= 100, (tag, len(train), len(valid))
            assert train.label_end.max() < fold["valid_start"]
            assert valid.label_end.max() < fold["test_first_signal"]
            raw_train = train[feature_names].to_numpy(dtype=np.float32)
            medians, scale = fit_transformer(raw_train)
            x_train = transform(raw_train, medians, scale)
            x_valid = transform(valid[feature_names].to_numpy(dtype=np.float32), medians, scale)
            y_train = train.label.to_numpy(dtype=float)
            y_valid = valid.label.to_numpy(dtype=float)
            fit_started = time.monotonic()
            linear = Lasso(**config["lasso"])
            linear.fit(x_train, y_train)
            model = lgb.train(
                config["lightgbm"],
                lgb.Dataset(x_train, label=y_train, feature_name=feature_names),
                num_boost_round=config["num_boost_round"],
                valid_sets=[lgb.Dataset(x_valid, label=y_valid, reference=None)],
                callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False)],
            )
            model.save_model(str(model_dir / (tag + "_lightgbm.txt")))
            transformer = {
                "features": feature_names,
                "medians": medians.tolist(),
                "scale": scale.tolist(),
                "lasso_coefficients": linear.coef_.tolist(),
            }
            (model_dir / (tag + "_transformer.json")).write_text(
                json.dumps(transformer), encoding="utf-8"
            )
            frames = []
            for signal in fold["test_signals"]:
                frame = inference[signal]
                x = transform(frame[feature_names].to_numpy(dtype=np.float32), medians, scale)
                prediction = frame[["day", "symbol", "label_raw"]].copy()
                prediction["pool"], prediction["window"] = pool, fold["window"]
                prediction["Lasso"] = linear.predict(x)
                prediction["LightGBM"] = model.predict(x, num_threads=1)
                assert np.isfinite(prediction[["Lasso", "LightGBM"]].to_numpy()).all()
                frames.append(prediction)
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(checkpoint, index=False)
            receipt = {
                "pool": pool,
                **fold,
                "train_rows": len(train),
                "valid_rows": len(valid),
                "train_days": train.day.nunique(),
                "valid_days": valid.day.nunique(),
                "train_label_end_max": train.label_end.max(),
                "valid_label_end_max": valid.label_end.max(),
                "iterations": model.best_iteration,
                "lasso_iterations": int(linear.n_iter_),
                "lasso_dual_gap": float(linear.dual_gap_),
                "lasso_nonzero": int(np.count_nonzero(linear.coef_)),
                "valid_mse_lasso": float(np.mean((linear.predict(x_valid) - y_valid) ** 2)),
                "valid_mse_lightgbm": float(
                    np.mean((model.predict(x_valid, num_threads=1) - y_valid) ** 2)
                ),
                "valid_mse_zero": float(np.mean(y_valid**2)),
                "fit_seconds": time.monotonic() - fit_started,
                "plan_sha256": plan_sha,
                "feature_sha256": feature_receipt["sha256"],
                "training_code_sha256": sha256(Path(__file__)),
            }
            metadata_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            receipts.append(receipt)
            predictions.append(combined)
            print(
                tag,
                "train",
                len(train),
                "valid",
                len(valid),
                "trees",
                model.best_iteration,
                "Lasso nonzero",
                receipt["lasso_nonzero"],
                "seconds",
                round(receipt["fit_seconds"], 2),
                flush=True,
            )
    result = pd.concat(predictions, ignore_index=True)
    assert not result.duplicated(["pool", "window", "day", "symbol"]).any()
    result.to_parquet(OUT / "predictions.parquet", index=False)
    pd.DataFrame(receipts).to_csv(OUT / "folds.csv", index=False)
    for (pool, window, day), frame in result.groupby(["pool", "window", "day"]):
        for scorer in ["Lasso", "LightGBM"]:
            diagnostics.append(
                {
                    "pool": pool,
                    "window": window,
                    "signal": day,
                    "scorer": scorer,
                    "rows": len(frame),
                    "scored_labels": frame.label_raw.notna().sum(),
                    "rank_ic": frame[scorer].corr(frame.label_raw, method="spearman"),
                    "prediction_std": frame[scorer].std(),
                }
            )
    pd.DataFrame(diagnostics).to_csv(OUT / "signal_diagnostics.csv", index=False)
    (OUT / "fit_qa.json").write_text(
        json.dumps(
            {
                "technical_status": "PASS",
                "folds": len(receipts),
                "models_fit": len(receipts) * 2,
                "feature_count": len(feature_names),
                "scorers": config["scorers"],
                "no_cross_boundary_labels": True,
                "training_only_feature_preprocessing": True,
                "prediction_sha256": sha256(OUT / "predictions.parquet"),
                "resource": RESOURCE_CONTEXT,
                "plan_sha256": plan_sha,
                "financial_industry_inputs": [],
                "fresh_out_of_sample": False,
                "versions": {
                    m: importlib.metadata.version(m)
                    for m in [
                        "numpy",
                        "pandas",
                        "scipy",
                        "scikit-learn",
                        "lightgbm",
                        "duckdb",
                        "baostock",
                    ]
                },
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Predictions frozen", len(result), "rows", flush=True)


if __name__ == "__main__":
    main()
