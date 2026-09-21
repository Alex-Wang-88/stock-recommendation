"""基本面归档与因子方向的无网络测试。"""

from __future__ import annotations

import pandas as pd

from recommend.archive import ArchiveStore
from recommend.factors.fundamental import _history_metrics, add_fundamental_factors
from recommend.fundamentals import (
    FINANCIAL_COLUMNS,
    VALUATION_COLUMNS,
    _prepare_derived_valuation,
)


def test_fundamental_tables_are_point_in_time(tmp_path):
    path = tmp_path / "archive.duckdb"
    with ArchiveStore(path) as store:
        valuation = pd.DataFrame(
            [
                {
                    "trade_date": "2026-09-17",
                    "symbol": "000001",
                    "pe_ttm": 8.0,
                    "source": "test",
                },
                {
                    "trade_date": "2026-09-18",
                    "symbol": "000001",
                    "pe_ttm": 10.0,
                    "source": "test",
                },
            ]
        )
        for column in VALUATION_COLUMNS:
            if column not in valuation:
                valuation[column] = pd.NA
        valuation["fetched_at"] = "2026-09-18 10:00:00"
        store.upsert("valuation_daily", valuation, VALUATION_COLUMNS)

        financial = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "pub_date": "2026-08-20",
                    "stat_date": "2026-06-30",
                    "roe_avg": 0.12,
                    "yoy_net_income": 0.25,
                    "source": "test",
                    "fetched_at": "2026-08-21 10:00:00",
                },
                {
                    "symbol": "000001",
                    "pub_date": "2026-09-20",
                    "stat_date": "2026-09-30",
                    "roe_avg": 0.90,
                    "yoy_net_income": 9.0,
                    "source": "test",
                    "fetched_at": "2026-09-20 10:00:00",
                },
            ]
        )
        for column in FINANCIAL_COLUMNS:
            if column not in financial:
                financial[column] = pd.NA
        store.upsert("financial_quarterly", financial, FINANCIAL_COLUMNS)

        valuation_asof = store.valuation_asof("2026-09-18")
        financial_asof = store.financial_asof("2026-09-18")

    assert valuation_asof.loc[0, "pe_ttm"] == 10.0
    assert financial_asof.loc[0, "roe_avg"] == 0.12


def test_financial_history_asof_returns_multiple_latest_published_periods(tmp_path):
    path = tmp_path / "archive.duckdb"
    with ArchiveStore(path) as store:
        financial = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "pub_date": "2025-08-20",
                    "stat_date": "2025-06-30",
                    "revenue": 80.0,
                    "source": "test",
                    "fetched_at": "2025-08-21",
                },
                {
                    "symbol": "000001",
                    "pub_date": "2026-04-20",
                    "stat_date": "2026-03-31",
                    "revenue": 110.0,
                    "source": "test",
                    "fetched_at": "2026-04-21",
                },
                {
                    "symbol": "000001",
                    "pub_date": "2026-08-20",
                    "stat_date": "2026-06-30",
                    "revenue": 120.0,
                    "source": "test",
                    "fetched_at": "2026-08-21",
                },
                {
                    "symbol": "000002",
                    "pub_date": "2026-09-20",
                    "stat_date": "2026-06-30",
                    "revenue": 999.0,
                    "source": "test",
                    "fetched_at": "2026-09-21",
                },
            ]
        )
        for column in FINANCIAL_COLUMNS:
            if column not in financial:
                financial[column] = pd.NA
        store.upsert("financial_quarterly", financial, FINANCIAL_COLUMNS)
        history = store.financial_history_asof("2026-08-31", periods=5)

    assert history["symbol"].unique().tolist() == ["000001"]
    assert history["stat_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-06-30",
        "2026-03-31",
        "2025-06-30",
    ]


def test_fundamental_factor_directions_are_explainable():
    frame = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "industry": ["银行", "银行"],
            "close": [10.0, 20.0],
            "ret_20": [0.1, -0.1],
            "ret_60": [0.2, 0.1],
        }
    )
    valuation = pd.DataFrame(
        {"symbol": ["000001", "000002"], "pe_ttm": [5.0, 20.0], "pb_mrq": [1.0, 2.0]}
    )
    financial = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "roe_avg": [0.10, 0.20],
            "gross_profit_margin": [0.30, 0.40],
            "yoy_net_income": [0.20, -0.10],
            "yoy_eps_basic": [0.30, -0.20],
            "total_share": [100.0, 100.0],
        }
    )
    out = add_fundamental_factors(frame, valuation, financial).set_index("symbol")

    assert out.loc["000001", "pe_ttm_value"] > out.loc["000002", "pe_ttm_value"]
    assert out.loc["000001", "pe_industry_value"] > out.loc["000002", "pe_industry_value"]
    assert out.loc["000002", "industry_leader_score"] > out.loc["000001", "industry_leader_score"]


def test_history_metrics_capture_growth_and_consistency_without_fabricating_periods():
    history = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "stat_date": "2025-06-30",
                "revenue": 80.0,
                "net_profit": 10.0,
                "yoy_net_income": 0.10,
                "yoy_eps_basic": 0.08,
            },
            {
                "symbol": "000001",
                "stat_date": "2026-03-31",
                "revenue": 110.0,
                "net_profit": 12.0,
                "yoy_net_income": 0.15,
                "yoy_eps_basic": 0.12,
            },
            {
                "symbol": "000001",
                "stat_date": "2026-06-30",
                "revenue": 120.0,
                "net_profit": 15.0,
                "yoy_net_income": 0.25,
                "yoy_eps_basic": 0.30,
            },
        ]
    )
    out = _history_metrics(history).set_index("symbol")

    assert out.loc["000001", "revenue_yoy"] == 0.5
    assert out.loc["000001", "profit_growth_acceleration"] == 0.10
    assert out.loc["000001", "eps_growth_acceleration"] == 0.18
    assert out.loc["000001", "positive_profit_streak"] == 3
    assert out.loc["000001", "positive_growth_streak"] == 3


def test_pe_ttm_uses_local_close_and_eps_without_fake_multiples():
    financial = pd.DataFrame(
        {"symbol": ["600519", "000001"], "eps_ttm": [10.0, -1.0]}
    )
    out = _prepare_derived_valuation(
        financial,
        {"600519": 150.0, "000001": 12.0},
        "2026-09-21",
        pd.Timestamp("2026-09-21 10:00:00").to_pydatetime(),
    ).set_index("symbol")

    assert out.loc["600519", "pe_ttm"] == 15.0
    assert pd.isna(out.loc["000001", "pe_ttm"])
    assert out["pb_mrq"].isna().all()
