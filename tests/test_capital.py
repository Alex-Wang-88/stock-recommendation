"""资金因子对主/备数据源缺失语义的测试。"""

from __future__ import annotations

import pandas as pd

from recommend.factors.capital import add_capital_factors


def test_sina_event_only_rows_do_not_turn_missing_net_buy_into_zero() -> None:
    universe = pd.DataFrame({"symbol": ["000001"]})
    dragon_tiger = pd.DataFrame(
        {
            "trade_date": ["2026-09-17"],
            "symbol": ["000001"],
            "net_buy": [pd.NA],
        }
    )
    out = add_capital_factors(universe, dragon_tiger=dragon_tiger)
    assert out.loc[0, "lhb_count_20"] == 1
    assert pd.isna(out.loc[0, "lhb_net_buy_20"])
    assert pd.isna(out.loc[0, "lhb_net_buy_rel"])
