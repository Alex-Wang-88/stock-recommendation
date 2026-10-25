"""Price-only, fixed cross-sectional scores; market state must be known at the signal close."""

from __future__ import annotations

import numpy as np

NEW_FACTORS = ("ret_5", "ret_10", "ret_60_skip5")


def direction_ranking(frame, scorer, above_ma60):
    from prepare_market_weight_batch import FACTORS

    columns = list(FACTORS) + list(NEW_FACTORS)
    projected = frame.loc[:, ["symbol", *columns]].copy()
    if not np.isfinite(projected[columns].to_numpy(dtype=float)).all():
        raise ValueError("All new and old factors must be finite for every common pool member")
    ranks = projected[columns].rank(pct=True)
    risk = ranks[["volatility_value", "drawdown_value", "downside_volatility_value"]].mean(axis=1)
    reverse5 = (-projected.ret_5).rank(pct=True)
    if scorer == "regime_switch":
        scorer = "trend_pullback" if above_ma60 else "reverse_risk"
    if scorer == "trend5":
        score = ranks.ret_5
    elif scorer in ["reverse5", "reverse10", "reverse20"]:
        column = {"reverse5": "ret_5", "reverse10": "ret_10", "reverse20": "ret_20"}[scorer]
        score = (-projected[column]).rank(pct=True)
    elif scorer == "trend60":
        score = ranks.ret_60
    elif scorer == "trend60_skip5":
        score = ranks.ret_60_skip5
    elif scorer == "trend_pullback":
        score = 0.5 * ranks.ret_60_skip5 + 0.3 * reverse5 + 0.2 * risk
    elif scorer == "reverse_risk":
        score = 0.5 * reverse5 + 0.3 * risk + 0.2 * ranks.turnover_value_20
    elif scorer == "dual_reverse":
        score = 0.5 * reverse5 + 0.5 * (-projected.ret_20).rank(pct=True)
    else:
        raise ValueError("Unknown direction scorer: " + scorer)
    result = projected[["symbol"]].copy()
    result["score"] = score
    assert result.score.between(0, 1 + 1e-12).all()
    return result.sort_values(["score", "symbol"], ascending=[False, True])
