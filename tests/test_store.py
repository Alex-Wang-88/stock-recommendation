"""归档库的幂等性与记账口径。"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from recommend.archive.store import ArchiveStore, split_tags


def _store(tmp_path) -> ArchiveStore:
    return ArchiveStore(tmp_path / "archive.duckdb")


def _theme_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600519", "000001"],
            "name": ["贵州茅台", "平安银行"],
            "reason": ["白酒+央企", "银行+金融科技"],
            "close": [1500.0, 12.0],
            "change_pct": [3.2, 1.1],
            "turnover_pct": [0.8, 1.5],
            "amount": [1e9, 2e9],
            "volume": [1e6, 2e6],
            "big_order_net": [1e7, -2e6],
            "market": ["沪市", "深市"],
            "source_date": ["2026-09-18", "2026-09-18"],
        }
    )


# --------------------------------------------------------------------------- #
# 标签拆分
# --------------------------------------------------------------------------- #


def test_tags_split_on_the_plus_separator() -> None:
    assert split_tags("光通信测试+CPO+伽蓝特订单+中报增长") == [
        "光通信测试",
        "CPO",
        "伽蓝特订单",
        "中报增长",
    ]


def test_other_separators_are_also_recognised() -> None:
    assert split_tags("算力、CPO，光模块|液冷") == ["算力", "CPO", "光模块", "液冷"]


def test_duplicate_tags_collapse_and_order_is_stable() -> None:
    """同一只股票重复写同一个标签只算一次，且保持首次出现的顺序。"""

    assert split_tags("算力+CPO+算力") == ["算力", "CPO"]


def test_missing_reason_yields_no_tags() -> None:
    assert split_tags(None) == []
    assert split_tags("") == []
    assert split_tags("++") == []


# --------------------------------------------------------------------------- #
# 幂等写入
# --------------------------------------------------------------------------- #


def test_writing_the_same_day_twice_does_not_duplicate(tmp_path) -> None:
    """回补必须可重跑 —— 这是断点续跑的前提。"""

    store = _store(tmp_path)
    try:
        store.write_themes(_theme_frame(), "2026-09-18")
        first = store.conn.execute("select count(*) from theme_attribution").fetchone()[0]
        store.write_themes(_theme_frame(), "2026-09-18")
        second = store.conn.execute("select count(*) from theme_attribution").fetchone()[0]
        assert first == second == 2
    finally:
        store.close()


def test_rewriting_a_day_with_fewer_rows_replaces_rather_than_accumulates(tmp_path) -> None:
    """上游修正了当日数据（变少）时，库里应当收敛到新值而不是叠加。"""

    store = _store(tmp_path)
    try:
        store.write_themes(_theme_frame(), "2026-09-18")
        trimmed = _theme_frame().head(1)
        store.write_themes(trimmed, "2026-09-18")
        assert store.conn.execute("select count(*) from theme_attribution").fetchone()[0] == 1
    finally:
        store.close()


def test_tag_counts_are_rebuilt_from_scratch(tmp_path) -> None:
    """标签聚合必须按日**重建**，否则替换单日数据后计数会残留。"""

    store = _store(tmp_path)
    try:
        store.write_themes(_theme_frame(), "2026-09-18")
        tags = dict(
            store.conn.execute("select tag, stock_count from theme_tag_daily").fetchall()
        )
        assert tags == {"白酒": 1, "央企": 1, "银行": 1, "金融科技": 1}

        store.write_themes(_theme_frame().head(1), "2026-09-18")
        tags = dict(
            store.conn.execute("select tag, stock_count from theme_tag_daily").fetchall()
        )
        assert tags == {"白酒": 1, "央企": 1}
    finally:
        store.close()


def test_empty_frame_writes_nothing(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        assert store.write_themes(pd.DataFrame(), "2026-09-18") == 0
    finally:
        store.close()


def test_missing_column_raises_instead_of_silently_filling(tmp_path) -> None:
    """缺列必须报错。静默补空会把"数据源改了字段名"变成"这些值本来就缺失"。"""

    store = _store(tmp_path)
    try:
        broken = _theme_frame().drop(columns=["reason"])
        try:
            store.write_themes(broken, "2026-09-18")
        except ValueError as exc:
            assert "reason" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("缺列时应当报错")
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# 记账口径
# --------------------------------------------------------------------------- #


def test_empty_and_failed_are_recorded_separately(tmp_path) -> None:
    """非交易日返回空是**正常**的，接口失效返回空是**故障** —— 两者不能混为一谈。"""

    store = _store(tmp_path)
    try:
        started = datetime.now(UTC)
        store.record_run("backfill-themes", "2026-09-19", started, 0, "EMPTY", "非交易日")
        store.record_run("backfill-themes", "2026-09-20", started, 0, "FAILED", "连接被重置")

        statuses = dict(
            store.conn.execute(
                "select status, count(*) from archive_runs group by status"
            ).fetchall()
        )
        assert statuses == {"EMPTY": 1, "FAILED": 1}
    finally:
        store.close()


def test_archived_dates_drives_resume(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        store.write_themes(_theme_frame(), "2026-09-17")
        store.write_themes(_theme_frame(), "2026-09-18")
        store.record_run("backfill-themes", "2026-09-19", datetime.now(UTC), 0, "EMPTY")
        # 只有真正写入过数据的日期才算已归档 —— 空日期要被重试。
        assert store.archived_dates("theme_attribution") == {"2026-09-17", "2026-09-18"}
    finally:
        store.close()


def test_coverage_reports_span_and_day_count(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        store.write_themes(_theme_frame(), "2026-09-17")
        store.write_themes(_theme_frame(), "2026-09-18")
        coverage = store.coverage().set_index("table")
        row = coverage.loc["theme_attribution"]
        assert row["rows"] == 4
        assert row["days"] == 2
        assert str(row["first"]) == "2026-09-17"
        assert str(row["last"]) == "2026-09-18"
    finally:
        store.close()


def test_top_tags_ranks_by_stock_count(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        frame = _theme_frame()
        frame["reason"] = ["白酒+央企", "白酒"]
        frame = pd.concat([frame, frame.iloc[[0]].assign(symbol="600521", name="丙")])
        store.write_themes(frame, "2026-09-18")
        top = store.top_tags("2026-09-18", limit=5)
        assert list(top["tag"]) == ["白酒", "央企"]
        assert list(top["stock_count"]) == [3, 2]
    finally:
        store.close()
