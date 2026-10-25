"""Refit only the sealed model families using strictly earlier validation-cohort data."""

from __future__ import annotations

import json

from design_index_improvement import OUT, PREVIOUS
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import argparse

    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import sha256
    from qlib_alpha_features import fit_transformer, purged_fold_masks, transform
    from sklearn.linear_model import Lasso

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["external", "recent", "both"], default="both")
    args = parser.parse_args()
    cohorts = ["external", "recent"] if args.cohort == "both" else [args.cohort]
    seal = OUT / "finalists_locked.json"
    chosen = json.loads(seal.read_text(encoding="utf-8"))
    request = json.loads((OUT / "validation_download_request.json").read_text(encoding="utf-8"))
    assert request["finalists_sha256"] == sha256(seal)
    old_plan = json.loads((PREVIOUS / "plan_locked.json").read_text(encoding="utf-8"))
    feature_names = json.loads((PREVIOUS / "feature_receipt.json").read_text(encoding="utf-8"))[
        "feature_names"
    ]
    members = pd.read_parquet(OUT / "validation_all_membership.parquet")
    model_dir = OUT / "validation_models"
    model_dir.mkdir(exist_ok=True)
    qa_path = OUT / "validation_fit_qa.json"
    receipts = []
    if qa_path.exists():
        old_qa = json.loads(qa_path.read_text(encoding="utf-8"))
        assert old_qa["finalists_sha256"] == sha256(seal)
        receipts = [f for f in old_qa["folds"] if f["cohort"] not in cohorts]
    for cohort in cohorts:
        features = pd.read_parquet(OUT / (cohort + "_alpha157.parquet"))
        predictions = []
        for candidate in chosen["finalists"]:
            if candidate["scorer"] == "P4":
                continue
            pool, scorer = candidate["pool"], candidate["scorer"]
            data = members.loc[members.pool == pool].merge(
                features,
                on=["day", "symbol"],
                validate="one_to_one",
            )
            learning = data.loc[data.fit_eligible.eq(True)].copy()
            grouped = learning.groupby("day").label_raw
            learning["label"] = (
                learning.label_raw - grouped.transform("mean")
            ) / grouped.transform(lambda x: x.std(ddof=0))
            learning = learning.loc[np.isfinite(learning.label)].copy()
            for fold in [f for f in request["folds"] if f["cohort"] == cohort]:
                train_mask, valid_mask = purged_fold_masks(learning.day, learning.label_end, fold)
                train, valid = learning.loc[train_mask], learning.loc[valid_mask]
                assert len(train) >= 500 and len(valid) >= 100
                assert train.label_end.max() < fold["valid_start"]
                assert valid.label_end.max() < fold["test_first_signal"]
                medians, scale = fit_transformer(train[feature_names].to_numpy(dtype=np.float32))
                x_train = transform(train[feature_names].to_numpy(dtype=np.float32), medians, scale)
                if scorer == "Lasso":
                    model = Lasso(**old_plan["lasso"])
                    model.fit(x_train, train.label.to_numpy(dtype=float))
                    model_meta = {
                        "iterations": int(model.n_iter_),
                        "coefficients": model.coef_.tolist(),
                        "dual_gap": float(model.dual_gap_),
                    }
                else:
                    import lightgbm as lgb

                    x_valid = transform(
                        valid[feature_names].to_numpy(dtype=np.float32), medians, scale
                    )
                    model = lgb.train(
                        old_plan["lightgbm"],
                        lgb.Dataset(x_train, label=train.label),
                        num_boost_round=old_plan["num_boost_round"],
                        valid_sets=[lgb.Dataset(x_valid, label=valid.label)],
                        callbacks=[
                            lgb.early_stopping(old_plan["early_stopping_rounds"], verbose=False)
                        ],
                    )
                    model_meta = {"iterations": model.best_iteration}
                    model.save_model(str(model_dir / f"{cohort}_{pool}_{fold['window']}.txt"))
                inference = data.loc[data.day.isin(fold["test_signals"])].copy()
                x = transform(inference[feature_names].to_numpy(dtype=np.float32), medians, scale)
                part = inference[["day", "symbol", "pool"]].copy()
                part["window"] = fold["window"]
                part[scorer] = model.predict(x)
                assert np.isfinite(part[scorer]).all()
                predictions.append(part)
                metadata = {
                    "cohort": cohort,
                    "pool": pool,
                    "scorer": scorer,
                    **fold,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "train_label_end_max": train.label_end.max(),
                    "valid_label_end_max": valid.label_end.max(),
                    "finalists_sha256": sha256(seal),
                    "medians": medians.tolist(),
                    "scale": scale.tolist(),
                    **model_meta,
                }
                (model_dir / f"{cohort}_{pool}_{fold['window']}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                receipts.append(
                    {
                        k: v
                        for k, v in metadata.items()
                        if k not in ["medians", "scale", "coefficients"]
                    }
                )
                print(cohort, pool, fold["window"], "causal fit complete", len(train), flush=True)
        if predictions:
            pd.concat(predictions, ignore_index=True).to_parquet(
                OUT / (cohort + "_predictions.parquet"), index=False
            )
        else:
            pd.DataFrame(columns=["day", "symbol", "pool", "window"]).to_parquet(
                OUT / (cohort + "_predictions.parquet"), index=False
            )
    qa_path.write_text(
        json.dumps(
            {
                "technical_status": "PASS",
                "folds": receipts,
                "models_fitted": len(receipts),
                "parameter_search_after_finalist_lock": False,
                "resource": RESOURCE_CONTEXT,
                "finalists_sha256": sha256(seal),
                "financial_industry_inputs": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
