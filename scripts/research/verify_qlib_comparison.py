"""Cross-check saved account paths, fee ledgers, label boundaries and provenance."""

from __future__ import annotations

import json
import shutil

from qlib_external_sources import OUT, ROOT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import pandas as pd
    from prepare_market_weight_batch import sha256

    plan = json.loads((OUT / "plan_locked.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(OUT / "summary.csv")
    nav = pd.read_csv(OUT / "daily_nav.csv")
    trades = pd.read_csv(OUT / "trades_10bps.csv")
    folds = pd.read_csv(OUT / "folds.csv")
    fit_qa = json.loads((OUT / "fit_qa.json").read_text(encoding="utf-8"))
    replay_qa = json.loads((OUT / "replay_qa.json").read_text(encoding="utf-8"))
    keys = ["pool", "window", "scorer", "mode", "cost_bps"]
    assert not summary.duplicated(keys).any()
    assert len(summary) == 720
    assert nav.groupby(keys).size().eq(30).all()
    final = nav.sort_values("day").groupby(keys).tail(1)[[*keys, "nav"]]
    joined = summary.merge(final, on=keys, validate="one_to_one")
    max_path_error = float((joined["return"] - (joined.nav - 1)).abs().max())
    assert max_path_error < 1e-12
    ledger = trades.groupby(keys).fee.sum().reset_index(name="ledger_fees")
    costed = (
        summary.loc[summary.cost_bps == 10]
        .merge(ledger, on=keys, how="left", validate="one_to_one")
        .fillna({"ledger_fees": 0.0})
    )
    assert len(costed) == 240
    max_fee_error = float((costed.fees - costed.ledger_fees).abs().max())
    assert max_fee_error < 1e-12
    fee_formula_error = float((trades.fee - trades.notional * 0.001).abs().max())
    assert fee_formula_error < 1e-12
    costs = summary.pivot(index=keys[:-1], columns="cost_bps", values="return")
    assert (costs[0] + 1e-10 >= costs[10]).all()
    assert (costs[10] + 1e-10 >= costs[20]).all()
    assert (folds.train_label_end_max < folds.valid_start).all()
    assert (folds.valid_label_end_max < folds.test_first_signal).all()
    assert sha256(OUT / "plan_locked.json") == sha256(ROOT / "config/research/qlib_comparison.json")
    assert sha256(OUT / "predictions.parquet") == fit_qa["prediction_sha256"]
    assert sha256(ROOT / "config/default.toml") == (
        "a39f2116505a00b4aff9ba9b6130d8fda5ec1be364548c81a8e30e3479bd1d52"
    )
    assert replay_qa["account_rows"] == len(summary)
    assert not plan["fresh_holdout"]
    formula = json.loads((OUT / "formula_audit.json").read_text(encoding="utf-8"))
    assert formula["technical_status"] == "PASS" and formula["formula_count"] == 157
    assert formula["port_sha256"] == sha256(ROOT / "scripts/research/qlib_alpha_features.py")
    snapshots = OUT / "code_snapshot"
    snapshots.mkdir(exist_ok=True)
    files = [
        *[
            ROOT / "scripts/research" / name
            for name in [
                "qlib_external_sources.py",
                "lock_qlib_comparison.py",
                "qlib_alpha_features.py",
                "prepare_qlib_features.py",
                "fit_qlib_comparison.py",
                "run_qlib_comparison.py",
                "report_qlib_comparison.py",
                "audit_qlib_formulas.py",
                "verify_qlib_comparison.py",
                "rule_account_replay.py",
                "account_replay.py",
                "prepare_market_weight_batch.py",
                "research_resources.py",
            ]
        ],
        ROOT / "tests/research/test_qlib_comparison.py",
        ROOT / "tests/research/test_rule_account_replay.py",
    ]
    provenance = []
    for path in files:
        destination = snapshots / path.name
        if not destination.exists():
            shutil.copyfile(path, destination)
        assert sha256(path) == sha256(destination), f"Snapshot changed: {path}"
        provenance.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    result = {
        "technical_status": "PASS",
        "account_rows": len(summary),
        "max_terminal_path_difference": max_path_error,
        "max_fee_ledger_difference": max_fee_error,
        "max_fee_formula_difference": fee_formula_error,
        "all_accounts_cost_monotonic": True,
        "all_folds_purged": True,
        "production_configuration_unchanged": True,
        "all_157_ported_formulas_checked_on_complete_bars": True,
        "fresh_out_of_sample_validation": False,
        "resource": RESOURCE_CONTEXT,
        "code_receipts": provenance,
    }
    (OUT / "account_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in result.items() if k != "code_receipts"}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
