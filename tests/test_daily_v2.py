"""Causality, immutable evidence, cash arithmetic and blocked execution for daily v2."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recommend.daily_data import parse_quotes
from recommend.daily_strategy import (
    build_features,
    canonical_hash,
    freeze_signal,
    load_protocol,
    select_daily,
)
from recommend.daily_validation import buy_fill, evaluate_cohort, materialize_entry
from recommend.market import SHANGHAI_TIMEZONE


@pytest.fixture
def protocol():
    return load_protocol(Path(__file__).parents[1] / "config/daily_strategy.json")


def now(day, hour=17):
    return datetime.fromisoformat(day).replace(hour=hour, tzinfo=SHANGHAI_TIMEZONE)


def quote(day, open_=10, close=10, prior=10, volume=10000):
    return {
        "symbol": "600001",
        "name": "样本公司",
        "status_date": day,
        "open": open_,
        "close": close,
        "pre_close": prior,
        "volume": volume,
    }


def snapshot(day, q):
    result = {"as_of": day, "fetched_at": now(day).isoformat(), "quotes": {"600001": q}}
    result["sha256"] = canonical_hash(result)
    return result


def signal(protocol, picks=True):
    selection = {
        "as_of": "2026-11-02",
        "picks": [{"symbol": "600001"}] if picks else [],
        "pool_symbols": ["600001"],
    }
    result = {
        "selection": selection,
        "protocol": protocol,
        "protocol_hash": canonical_hash(protocol),
        "recorded_at": now("2026-11-02").isoformat(),
    }
    result["signal_hash"] = canonical_hash(result)
    return result


def test_features_do_not_read_future_prices(bars_factory):
    bars = bars_factory(n_days=280, symbols=("600001",), start="2025-01-01")
    as_of = str(bars.iloc[249].trade_date)[:10]
    expected = build_features(bars.iloc[:250], as_of)
    bars.loc[bars.trade_date > as_of, ["close", "open", "high", "low"]] *= 100
    pd.testing.assert_frame_equal(expected, build_features(bars, as_of))
    c = bars.iloc[:250].close
    assert expected.iloc[0].trend == pytest.approx(c.iloc[-21] / c.iloc[-121] - 1)
    assert expected.iloc[0].stability == pytest.approx(-c.pct_change().tail(60).std(ddof=0))


def test_duplicate_prices_and_missing_factors_are_not_scored(bars_factory):
    bars = bars_factory(n_days=250, symbols=("600001",))
    as_of = str(bars.iloc[-1].trade_date)[:10]
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(pd.concat([bars, bars.tail(1)]), as_of)
    bars.loc[bars.index[-1], "amount"] = np.nan
    assert build_features(bars, as_of).empty


def fixture_pool():
    rows = [
        {
            "symbol": f"60000{i}",
            "close": 12,
            "ma120": 10,
            "trend": 0.1 + i / 20,
            "stability": -0.01 - i / 1000,
            "pullback": i / 100,
            "amount_20": 2e8,
            "active": True,
            "valid_ohlc": True,
        }
        for i in range(1, 7)
    ]
    meta = pd.DataFrame(
        [
            {
                "symbol": r["symbol"],
                "name": f"公司{i}",
                "industry": str(i // 2),
                "listed_date": "2000-01-01",
                "status_date": "2026-11-02",
            }
            for i, r in enumerate(rows)
        ]
    )
    return pd.DataFrame(rows), meta


def test_selection_is_fixed_weight_deterministic_and_industry_diverse(protocol):
    features, meta = fixture_pool()
    first = select_daily(features, meta, protocol, "2026-11-02")
    second = select_daily(features.sample(frac=1, random_state=7), meta, protocol, "2026-11-02")
    assert first == second
    assert len(first["picks"]) == 3
    assert len({p["industry"] for p in first["picks"]}) == 3
    for pick in first["picks"]:
        assert pick["score"] == pytest.approx(sum(p["contribution"] for p in pick["parts"]))
        assert sum(p["weight"] for p in pick["parts"]) == 1


def test_st_stale_names_and_missing_values_do_not_enter_pool(protocol):
    features, meta = fixture_pool()
    meta.loc[0, "name"] = "*ST测试"
    meta.loc[1, "status_date"] = "2026-10-30"
    features.loc[2, "trend"] = np.nan
    result = select_daily(features, meta, protocol, "2026-11-02")
    assert not {"600001", "600002", "600003"} & set(result["pool_symbols"])


def test_market_gate_records_cash_without_relaxing_weights(protocol):
    features, meta = fixture_pool()
    features["ma120"] = 20
    result = select_daily(features, meta, protocol, "2026-11-02")
    assert result["picks"] == []
    assert result["market_breadth"] == 0


def test_signal_freezing_never_replaces_a_signal_and_refuses_backdating(tmp_path, protocol):
    selection = signal(protocol)["selection"]
    path = freeze_signal(tmp_path, selection, protocol, now("2026-11-02"))
    original = path.read_bytes()
    changed = copy.deepcopy(selection)
    changed["picks"] = []
    freeze_signal(tmp_path, changed, protocol, now("2026-11-02", 18))
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="window"):
        freeze_signal(tmp_path, selection, protocol, now("2026-11-03"))
    altered = copy.deepcopy(protocol)
    altered["selection_count"] = 3
    with pytest.raises(ValueError, match="version"):
        freeze_signal(tmp_path, selection, altered, now("2026-11-02"))


def test_buy_cash_lots_cost_and_limit_up_are_accounted(protocol):
    fill = buy_fill(quote("2026-11-03"), 20000, protocol)
    assert fill["quantity"] == 1900
    assert fill["cost"] == 38
    assert fill["spent"] == 19038
    assert buy_fill(quote("2026-11-03", open_=11), 20000, protocol)["quantity"] == 0
    assert buy_fill(quote("2026-11-03", volume=0), 20000, protocol)["quantity"] == 0
    assert buy_fill(quote("2026-11-03"), 500, protocol)["quantity"] == 0


def test_entry_ledger_is_immutable_and_rejects_tampering(tmp_path, protocol):
    s = signal(protocol)
    snap = snapshot("2026-11-03", quote("2026-11-03"))
    path = tmp_path / "entry.json"
    a = materialize_entry(s, "2026-11-03", snap, path)
    snap["quotes"]["600001"]["open"] = 5
    assert materialize_entry(s, "2026-11-03", snap, path) == a
    modified = json.loads(path.read_text(encoding="utf-8"))
    modified["cash"] = 1e10
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="integrity"):
        materialize_entry(s, "2026-11-03", snap, path)


def test_late_entry_capture_cannot_create_a_forward_fill(tmp_path, protocol):
    snap = snapshot("2026-11-03", quote("2026-11-03"))
    snap["fetched_at"] = now("2026-11-05").isoformat()
    with pytest.raises(ValueError, match="late"):
        materialize_entry(signal(protocol), "2026-11-03", snap, tmp_path / "entry.json")


def test_net_return_matches_independent_cash_ledger_and_index(tmp_path, protocol):
    p = {**protocol, "holding_sessions": 2}
    days = ["2026-11-02", "2026-11-03", "2026-11-04"]
    quotes = {
        days[1]: snapshot(days[1], quote(days[1], close=10.2)),
        days[2]: snapshot(days[2], quote(days[2], close=10.5, prior=10.2)),
    }
    index = {days[1]: {"open": 100}, days[2]: {"close": 102}}
    result = evaluate_cohort(signal(p), days, quotes, index, tmp_path)
    quantity = 9900
    expected_cash = 100000 - quantity * 10 * 1.002 + quantity * 10.5 * 0.998
    assert result["status"] == "matured"
    assert result["net_return"] == pytest.approx(expected_cash / 100000 - 1)
    assert result["excess"] == pytest.approx(result["net_return"] - 0.02)
    assert result["fees"] == pytest.approx(quantity * (10 + 10.5) * 0.002)


def test_failed_entry_allocation_stays_cash_without_replacement(tmp_path, protocol):
    p = {**protocol, "holding_sessions": 2}
    days = ["2026-11-02", "2026-11-03", "2026-11-04"]
    quotes = {day: snapshot(day, quote(day, open_=11, close=11, prior=10)) for day in days[1:]}
    result = evaluate_cohort(signal(p), days, quotes, {}, tmp_path)
    assert result["status"] == "matured"
    assert result["net_return"] == 0
    assert result["filled"] == 0
    assert result["excess"] is None


def test_limit_down_exit_is_deferred_and_not_dropped(tmp_path, protocol):
    p = {**protocol, "holding_sessions": 2}
    days = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"]
    quotes = {
        days[1]: snapshot(days[1], quote(days[1])),
        days[2]: snapshot(days[2], quote(days[2], close=9)),
        days[3]: snapshot(days[3], quote(days[3], open_=9, close=9.1, prior=9)),
    }
    blocked = evaluate_cohort(signal(p), days[:3], quotes, {}, tmp_path)
    assert blocked["status"] == "exit_blocked"
    assert blocked["net_return"] is None
    finished = evaluate_cohort(signal(p), days, quotes, {}, tmp_path)
    assert finished["status"] == "matured"
    assert finished["exit_day"] == days[3]


def test_missing_quotes_and_corporate_actions_block_acceptance(tmp_path, protocol):
    p = {**protocol, "holding_sessions": 2}
    days = ["2026-11-02", "2026-11-03", "2026-11-04"]
    quotes = {days[1]: snapshot(days[1], quote(days[1]))}
    missing = evaluate_cohort(signal(p), days, quotes, {}, tmp_path)
    assert missing["status"] == "data_gap"
    quotes[days[2]] = snapshot(days[2], quote(days[2], prior=5, close=5.1))
    action = evaluate_cohort(signal(p), days, quotes, {}, tmp_path)
    assert action["status"] == "data_gap"
    assert "公司行动" in action["issues"][0]


def test_zero_pick_signal_also_matures_as_cash_against_index(tmp_path, protocol):
    p = {**protocol, "holding_sessions": 2}
    days = ["2026-11-02", "2026-11-03", "2026-11-04"]
    quotes = {d: snapshot(d, quote(d)) for d in days[1:]}
    result = evaluate_cohort(
        signal(p, picks=False),
        days,
        quotes,
        {days[1]: {"open": 100}, days[2]: {"close": 110}},
        tmp_path,
    )
    assert result["status"] == "matured"
    assert result["net_return"] == 0
    assert result["excess"] == pytest.approx(-0.1)


def test_quote_parser_distinguishes_explicit_no_trade_and_stale_prices():
    fields = [""] * 31
    fields[0:7] = ["51", "样本", "600001", "10", "10", "0", "0"]
    fields[30] = "20261102090000"
    text = 'v_sh600001="' + "~".join(fields) + '";'
    result = parse_quotes(text, "2026-11-02")
    assert len(result) == 1 and result[0]["session_traded"] is False
    assert parse_quotes(text, "2026-11-03") == []
