"""本地风险/确认因子的离线测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommend.factors import build_factor_table
from recommend.factors.risk import add_risk_factors


def test_risk_factors_are_full_window_and_directional(bars_factory) -> None:
    def close_for(_symbol_index: int, day_index: int) -> float:
        if day_index == 55:
            return 5.0
        return 10.0 + day_index * 0.01

    bars = bars_factory(n_days=100, symbols=("000001",), close_for=close_for)
    out = add_risk_factors(bars)
    row = out.iloc[0]
    assert np.isfinite(row["realized_vol_20"])
    assert np.isfinite(row["downside_vol_20"])
    assert row["volatility_value"] == -row["realized_vol_20"]
    assert row["downside_volatility_value"] == -row["downside_vol_20"]
    assert row["max_drawdown_60"] < 0
    assert row["drawdown_value"] == row["max_drawdown_60"]
    assert row["signed_amount_pressure_20"] > 0

    short = add_risk_factors(bars_factory(n_days=20, symbols=("000001",)))
    assert short["realized_vol_20"].isna().all()
    assert short["downside_vol_20"].isna().all()
    assert short["max_drawdown_60"].isna().all()
    assert short["signed_amount_pressure_20"].isna().all()


def test_factor_table_includes_risk_category_without_changing_rank_members(bars_factory) -> None:
    bars = bars_factory(n_days=80, symbols=("000001",))
    table = build_factor_table(
        universe=bars[["symbol"]].drop_duplicates(),
        bars=bars,
        base_closes=None,
        attribution=pd.DataFrame(columns=["symbol", "reason", "big_order_net"]),
        tag_daily=pd.DataFrame(columns=["tag", "stock_count"]),
        as_of="2025-04-22",
    )
    assert "max_drawdown_60" in table.frame.columns
    assert table.categories()["max_drawdown_60"] == "risk"
