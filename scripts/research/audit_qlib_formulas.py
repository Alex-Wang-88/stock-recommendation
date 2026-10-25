"""Check the local port against expressions from the pinned upstream formula builder."""

from __future__ import annotations

import ast
import json

from qlib_external_sources import OUT
from research_resources import configure_background_job

RESOURCE_CONTEXT = configure_background_job()


def main():
    import numpy as np
    import pandas as pd
    from prepare_market_weight_batch import sha256
    from qlib_alpha_features import alpha157

    receipt = json.loads((OUT / "upstream/receipt.json").read_text(encoding="utf-8"))
    entry = next(x for x in receipt["files"] if x["file"] == "qlib__contrib__data__loader.py")
    path = OUT / "upstream" / entry["file"]
    assert sha256(path) == entry["sha256"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Alpha158DL")
    builder = next(
        x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "get_feature_config"
    )
    builder.decorator_list = []
    # Only this reviewed pure-expression builder is run, with no imports or filesystem builtins.
    namespace = {"__builtins__": {"range": range, "str": str}}
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(path), "exec"), namespace)
    expressions, names = namespace["get_feature_config"]()
    assert len(names) == 158
    rng = np.random.default_rng(123)
    close = 20 * np.exp(np.cumsum(rng.normal(0.001, 0.015, 150)))
    bars = pd.DataFrame(
        {
            "open": close * rng.uniform(0.98, 1.02, len(close)),
            "high": close * 1.05,
            "low": close * 0.95,
            "close": close,
            "volume": rng.integers(1_000, 10_000, len(close)).astype(float),
        }
    )
    actual = alpha157(bars)

    def rolling(x, w):
        return x.rolling(w, min_periods=1)

    def linear_value(x, kind):
        if len(x) < 2:
            return np.nan
        grid = np.arange(len(x))
        beta, intercept = np.polyfit(grid, x, 1)
        if kind == "slope":
            return beta
        if kind == "rsquare":
            return np.corrcoef(grid, x)[0, 1] ** 2
        return x[-1] - (beta * grid[-1] + intercept)

    operators = {
        "Ref": lambda x, w: x.shift(w),
        "Greater": np.maximum,
        "Less": np.minimum,
        "Abs": np.abs,
        "Log": np.log,
        "Mean": lambda x, w: rolling(x, w).mean(),
        "Sum": lambda x, w: rolling(x, w).sum(),
        "Std": lambda x, w: rolling(x, w).std(ddof=1),
        "Max": lambda x, w: rolling(x, w).max(),
        "Min": lambda x, w: rolling(x, w).min(),
        "Quantile": lambda x, w, q: rolling(x, w).quantile(q),
        "Rank": lambda x, w: rolling(x, w).rank(pct=True),
        "IdxMax": lambda x, w: rolling(x, w).apply(lambda a: a.argmax() + 1, raw=True),
        "IdxMin": lambda x, w: rolling(x, w).apply(lambda a: a.argmin() + 1, raw=True),
        "Slope": lambda x, w: rolling(x, w).apply(lambda a: linear_value(a, "slope"), raw=True),
        "Rsquare": lambda x, w: rolling(x, w).apply(lambda a: linear_value(a, "rsquare"), raw=True),
        "Resi": lambda x, w: rolling(x, w).apply(lambda a: linear_value(a, "residual"), raw=True),
        "Corr": lambda x, y, w: rolling(x, w).corr(y),
    }
    environment = {**operators, **{c: bars[c] for c in bars.columns}}
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.USub,
        ast.UAdd,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Compare,
        ast.Gt,
        ast.Lt,
    )
    expected_names, differences = [], {}
    for expression, name in zip(expressions, names, strict=True):
        if name == "VWAP0":
            continue
        parsed = ast.parse(expression.replace("$", ""), mode="eval")
        assert all(isinstance(node, allowed) for node in ast.walk(parsed))
        expected = eval(
            compile(parsed, "<pinned-formula>", "eval"), {"__builtins__": {}}, environment
        )
        np.testing.assert_allclose(actual[name].iloc[65:], expected.iloc[65:], rtol=1e-7, atol=1e-7)
        differences[name] = float((actual[name].iloc[65:] - expected.iloc[65:]).abs().max())
        expected_names.append(name)
    assert actual.columns.tolist() == expected_names
    result = {
        "technical_status": "PASS",
        "upstream_commit": receipt["commit"],
        "formula_count": len(expected_names),
        "omitted": ["VWAP0"],
        "representative_complete_bar_max_absolute_error": max(differences.values()),
        "full_missing_bar_operator_equivalence_claimed": False,
        "independent_regression_reference": "numpy.polyfit / corrcoef",
        "port_sha256": sha256(OUT.parents[1] / "scripts/research/qlib_alpha_features.py"),
    }
    (OUT / "formula_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
