"""Verify cash, real-inventory buffers and blocked positions in the research engine."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from rule_account_replay import replay_rules


def case():
    days = [f"d{i:02}" for i in range(30)]
    signals = ["prior", *[days[i - 1] for i in range(5, 30, 5)]]
    index = {day: i for i, day in enumerate(["prior", *days])}
    arrays = {k: np.full((31, 2), 10.0) for k in ["open", "high", "low", "close", "prior_close"]}
    arrays.update(
        volume=np.ones((31, 2)),
        tradable=np.ones((31, 2), dtype=bool),
        is_st=np.zeros((31, 2), dtype=bool),
        growth_board=np.zeros(2, dtype=bool),
    )
    contexts = {
        signal: {
            "ranking": ["X", "Y"] if i == 0 else ["Y", "X"],
            "pool": ["X", "Y"],
            "exposure": 1.0,
            "volatility": {0: 0.01, 1: 0.02},
        }
        for i, signal in enumerate(signals)
    }
    window = {"days": days, "signal_dates": signals}
    rule = {"rebalance_days": 10, "positions": 1, "retain_rank": 1, "weighting": "equal"}
    return window, contexts, arrays, {"X": 0, "Y": 1}, index, rule


def test_buffer_keeps_actual_inventory_and_avoids_a_roundtrip():
    window, contexts, arrays, symbols, index, rule = case()
    plain = replay_rules(window, contexts, arrays, symbols, index, 10, rule)
    buffered = replay_rules(
        window, contexts, arrays, symbols, index, 10, {**rule, "retain_rank": 2}
    )
    assert len(plain[1]) == 4
    assert len(buffered[1]) == 2
    assert all(t["symbol"] == 0 for t in buffered[1])
    assert buffered[0][-1] > plain[0][-1]


def test_blocked_inventory_occupies_the_only_slot_until_it_can_be_sold():
    window, contexts, arrays, symbols, index, rule = case()
    arrays["tradable"][index["d10"], 0] = False
    arrays["tradable"][index["d20"], 0] = False
    result = replay_rules(window, contexts, arrays, symbols, index, 10, rule)
    assert all(t["symbol"] == 0 for t in result[1])
    assert len(result[1]) == 2
    assert len(result[2]) == 2
    assert all(d["positions"] == 1 for d in result[4])
    assert result[3] == 0


@pytest.mark.parametrize("bps", [0, 10, 20])
def test_fractional_exposure_funds_orders_without_leverage_or_free_fees(bps):
    window, contexts, arrays, symbols, index, rule = case()
    for context in contexts.values():
        context["exposure"] = 0.5
    result = replay_rules(window, contexts, arrays, symbols, index, bps, {**rule, "retain_rank": 2})
    first_buy = result[1][0]["notional"]
    assert first_buy == pytest.approx(0.5 / (1 + 0.5 * bps / 10000), abs=1e-12)
    assert result[0][-1] + sum(t["fee"] for t in result[1]) == pytest.approx(1, abs=1e-12)
    assert result[0].max() <= 1 + 1e-12


def test_inverse_volatility_cap_is_applied_to_the_actual_order_allocations():
    names = [f"s{i:02}" for i in range(50)]
    arrays = {k: np.full((3, 50), 10.0) for k in ["open", "high", "low", "close", "prior_close"]}
    arrays.update(
        volume=np.ones((3, 50)),
        tradable=np.ones((3, 50), dtype=bool),
        is_st=np.zeros((3, 50), dtype=bool),
        growth_board=np.zeros(50, dtype=bool),
    )
    contexts = {
        "prior": {
            "ranking": names,
            "pool": names,
            "exposure": 1.0,
            "volatility": {i: 0.001 if i == 0 else 0.02 for i in range(50)},
        }
    }
    rule = {"rebalance_days": 10, "positions": 50, "retain_rank": 75, "weighting": "inverse_vol"}
    result = replay_rules(
        {"days": ["a", "b"], "signal_dates": ["prior"]},
        contexts,
        arrays,
        {s: i for i, s in enumerate(names)},
        {"prior": 0, "a": 1, "b": 2},
        10,
        rule,
    )
    buys = [t["notional"] for t in result[1] if t["side"] == "buy"]
    assert len(buys) == 50
    assert max(buys) / sum(buys) == pytest.approx(0.04, abs=1e-12)
    assert result[0][-1] + sum(t["fee"] for t in result[1]) == pytest.approx(1, abs=1e-12)
