"""前向纸盘测试 —— 指纹稳定性、留证完整性、对账的「不可测」处理。

对账部分最重要的一条：**持有期没走完的记录不能按 0% 计入**。
那会把「还不可测」伪装成「没涨没跌」，从而在样本还小的时候系统性地把均值拉向 0，
看起来像"很稳"。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from recommend.forward import (
    append_record,
    build_record,
    evaluate,
    load_journal,
    spec_fingerprint,
)
from recommend.market import MarketStore
from recommend.screen import ScreenResult, Spec


def make_result(frame: pd.DataFrame, payload: dict) -> ScreenResult:
    spec = Spec.from_dict(payload)
    return ScreenResult(
        frame=frame,
        spec=spec,
        as_of="2026-09-18",
        universe_size=5000,
        pool_size=len(frame),
        rejections={"close_position_250 不在区间": 4000},
        notes=["题材归档为空 ⇒ 题材因子缺失"],
        weights={"momentum": 0.35},
    )


# -- 指纹 ------------------------------------------------------------------- #


def test_fingerprint_ignores_as_of():
    """前向测试要的是「同一个 spec 连续跑 N 天」。

    as_of 每天都变，若算进指纹，每一天都成了一个新 spec，样本永远归不了组。
    """

    base = {"close_position_250": [None, 0.35], "limit": 25}
    today = spec_fingerprint(Spec.from_dict(base))
    tomorrow = spec_fingerprint(Spec.from_dict({**base, "as_of": "2026-10-01"}))
    assert today == tomorrow


def test_fingerprint_changes_when_the_condition_changes():
    """条件一动，指纹必须变 —— 否则两个不同的 spec 会被混进同一组统计。"""

    a = spec_fingerprint(Spec.from_dict({"close_position_250": [None, 0.35]}))
    b = spec_fingerprint(Spec.from_dict({"close_position_250": [None, 0.30]}))
    assert a != b


def test_fingerprint_is_stable_across_runs():
    payload = {"themes": ["算力"], "volume_trend_20": ">1.5", "limit": 10}
    assert spec_fingerprint(Spec.from_dict(payload)) == spec_fingerprint(Spec.from_dict(payload))


# -- 留证 ------------------------------------------------------------------- #


def test_build_record_keeps_the_close_price_used_as_a_reference():
    """只存代码是不够的：日后复权基准变了，用现在的行情回算当时的推荐会失真。"""

    frame = pd.DataFrame(
        {"symbol": ["000001"], "name": ["平安银行"], "close": [11.70], "score": [0.5]}
    )
    record = build_record(make_result(frame, {"limit": 5}), run_at="2026-09-19T13:00:00")
    assert record.candidates == [
        {"symbol": "000001", "name": "平安银行", "close": 11.70, "score": 0.5}
    ]
    assert record.spec_hash == spec_fingerprint(Spec.from_dict({"limit": 5}))


def test_build_record_captures_weights_breakdown_and_reasons():
    frame = pd.DataFrame(
        {
            "symbol": ["000001"],
            "name": ["A"],
            "close": [10.0],
            "score": [0.8],
            "score_momentum": [0.9],
            "score_coverage": [1.0],
            "ret_20": [0.1],
            "volume_trend_20": [1.5],
            "theme_tags": ["算力"],
        }
    )
    result = make_result(frame, {"limit": 5})
    result.weights = {"momentum": 0.35}
    record = build_record(result, run_at="t")

    candidate = record.candidates[0]
    assert record.weights == {"momentum": 0.35}
    assert candidate["score_breakdown"] == {
        "momentum": 0.9,
        "theme": None,
        "capital": None,
        "liquidity": None,
    }
    assert any("momentum" in line for line in candidate["reasons"])


def test_build_record_marks_a_missing_close_as_none():
    """当时没有收盘价的推荐**无法对账** —— 显式标 None，不要留 NaN 假装是 0%。"""

    frame = pd.DataFrame(
        {"symbol": ["000001"], "name": ["停牌股"], "close": [None], "score": [0.5]}
    )
    record = build_record(make_result(frame, {"limit": 5}), run_at="t")
    assert record.candidates[0]["close"] is None


def test_build_record_keeps_rejections_and_notes():
    """没有剔除日志，「那天只选出 2 只」事后分不清是条件严还是因子整列缺失。"""

    frame = pd.DataFrame({"symbol": ["000001"], "close": [1.0], "score": [0.1]})
    record = build_record(make_result(frame, {"limit": 5}), run_at="t")
    assert record.rejections
    assert any("题材归档为空" in note for note in record.notes)
    assert record.universe == 5000 and record.kept == 1


def test_append_record_is_idempotent_per_day_and_appends_the_journal(tmp_path):
    frame = pd.DataFrame({"symbol": ["000001"], "name": ["A"], "close": [1.0], "score": [0.1]})
    result = make_result(frame, {"limit": 5})

    first = build_record(result, run_at="2026-09-19T13:00:00")
    path, journal = append_record(tmp_path, first)
    assert path.exists()
    # 同日同 spec 重跑：单次记录被覆盖（不是堆出第二个文件），jsonl 仍追加一行。
    append_record(tmp_path, build_record(result, run_at="2026-09-19T14:00:00"))

    assert len(list(tmp_path.glob("*.json"))) == 1
    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["symbols"] == ["000001"]
    assert load_journal(tmp_path)["kept"].tolist() == [1, 1]


def test_load_journal_is_empty_not_an_error_before_the_first_run(tmp_path):
    assert load_journal(tmp_path / "还没建").empty


# -- 对账 ------------------------------------------------------------------- #


@pytest.fixture
def rising_store(tmp_path, parquet_writer):
    """30 个交易日：`000001` 每日 +1%，其余 4 只横盘。"""

    dates = pd.bdate_range("2026-08-03", periods=30)
    rows = []
    for day_index, day in enumerate(dates):
        rows.append(
            {
                "symbol": "000001",
                "trade_date": day,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0 * 1.01**day_index,
                "pre_close": 10.0,
                "volume": 1.0,
                "amount": 1.0,
            }
        )
        for symbol in ("000002", "000003", "000004", "000005"):
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "pre_close": 10.0,
                    "volume": 1.0,
                    "amount": 1.0,
                }
            )
    store = MarketStore(parquet_writer(pd.DataFrame(rows), tmp_path / "daily.parquet"))
    yield store, [day.strftime("%Y-%m-%d") for day in dates]
    store.close()


def test_evaluate_measures_the_forward_return(tmp_path, rising_store):
    store, dates = rising_store
    frame = pd.DataFrame({"symbol": ["000001"], "name": ["A"], "close": [10.0], "score": [0.5]})
    record = build_record(make_result(frame, {"limit": 5}), run_at="t")
    record.as_of = dates[0]
    append_record(tmp_path, record)

    per_run, grouped = evaluate(store, tmp_path, horizons=(1, 5))
    day_one = per_run[(per_run["horizon"] == 1)].iloc[0]
    assert day_one["n"] == 1
    assert day_one["pick_ret"] == pytest.approx(0.01, rel=1e-6)
    # 基准是 5 只等权 ⇒ 只有 1/5 的涨幅被算进去。
    assert day_one["universe_ret"] == pytest.approx(0.01 / 5, rel=1e-6)
    assert day_one["excess"] == pytest.approx(0.01 - 0.01 / 5, rel=1e-6)
    assert set(grouped["horizon"]) == {1, 5}


def test_evaluate_excludes_periods_that_have_not_finished(tmp_path, rising_store):
    """as-of 落在样本末端 ⇒ 持有期还没走完 ⇒ `n = 0`、`excess = None`。

    绝不能让这些记录以 0% 参与平均 —— 那会把「还不可测」算成「没涨没跌」。
    """

    store, dates = rising_store
    frame = pd.DataFrame({"symbol": ["000001"], "name": ["A"], "close": [10.0], "score": [0.5]})
    record = build_record(make_result(frame, {"limit": 5}), run_at="t")
    record.as_of = dates[-1]  # 最后一天，任何持有期都走不完
    append_record(tmp_path, record)

    per_run, grouped = evaluate(store, tmp_path, horizons=(1, 5, 20))
    assert (per_run["n"] == 0).all()
    assert per_run["excess"].isna().all()
    assert per_run["n_requested"].eq(1).all()  # 请求了 1 只，只是量不到
    # 汇总里 `measured` 必须是 0 —— 这一列才是"到底有几条算得出"。
    assert (grouped["measured"] == 0).all()


def test_evaluate_counts_symbols_missing_at_exit_as_unmeasurable(tmp_path, rising_store):
    """推荐里的标的在出场日停牌/退市 ⇒ 该条**不可测**，但请求数要如实保留。"""

    store, dates = rising_store
    frame = pd.DataFrame(
        {"symbol": ["000001", "999999"], "name": ["A", "不存在的票"], "close": [10.0, 5.0],
         "score": [0.5, 0.4]}
    )
    record = build_record(make_result(frame, {"limit": 5}), run_at="t")
    record.as_of = dates[0]
    append_record(tmp_path, record)

    per_run, _ = evaluate(store, tmp_path, horizons=(1,))
    row = per_run.iloc[0]
    assert row["n_requested"] == 2
    assert row["n"] == 1  # 只有 000001 量得到
