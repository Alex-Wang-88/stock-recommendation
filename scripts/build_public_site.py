"""Build a small, public, read-only recommendation site from the latest report.

The generated site contains the displayed recommendation snapshot, recent
OHLC rows for those candidates, and the forward-validation summary. It never
copies DuckDB, the full market history, source records, announcement queues,
or the full runtime-state directory to GitHub Pages.
"""

from __future__ import annotations

import argparse
import csv
import duckdb
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommend.config import AppConfig  # noqa: E402
from recommend.rank.score import FACTOR_LABELS, RANK_MEMBERS  # noqa: E402
from recommend.screen.spec import FILTER_FIELDS  # noqa: E402

REPORT_NAME = re.compile(
    r"(?P<as_of>\d{4}-\d{2}-\d{2})(?:__(?P<spec_hash>[0-9a-fA-F]{8,}))?\.csv$"
)
CATEGORY_LABELS = {
    "industry": "行业",
    "growth": "增长",
    "quality": "财务质量",
    "valuation": "估值",
    "momentum": "动量",
    "risk": "风险",
    "theme": "题材",
    "capital": "资金",
    "liquidity": "流动性",
}
SCREEN_FILTER_LABELS = {
    "close_position_250": "近 250 日收盘区间位置",
    "dist_to_52w_high": "距近 250 日最高收盘的回撤",
    "history_days": "有效行情历史",
    "listed_days": "上市天数",
    "turnover_value_20": "近 20 日日均成交额",
    "volume_trend_20": "近 5 日均量 / 近 20 日均量",
}
NUMERIC_FIELDS = (
    "close",
    "score",
    "score_coverage",
    "buy_low",
    "buy_high",
    "pullback_price",
    "no_chase_price",
    "invalidation_price",
    "target_1",
    "target_2",
    "ret_20",
    "relative_ret_20",
    "pe_ttm",
    "roe_avg",
)
TEXT_FIELDS = (
    "symbol",
    "name",
    "industry",
    "operation_signal",
    "expectation_state",
    "operation_reason",
    "reason",
    "announcement_alert",
)
EVALUATION_FIELDS = (
    "horizon",
    "runs",
    "measured",
    "pick_ret",
    "universe_ret",
    "excess_mean",
    "excess_std",
)


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_recommendation_report(reports_dir: Path) -> tuple[Path, str, str] | None:
    candidates: list[tuple[str, int, Path, str]] = []
    for path in reports_dir.glob("*.csv"):
        match = REPORT_NAME.fullmatch(path.name)
        if not match:
            continue
        candidates.append(
            (
                match.group("as_of"),
                path.stat().st_mtime_ns,
                path,
                match.group("spec_hash") or "",
            )
        )
    if not candidates:
        return None
    _, _, path, spec_hash = max(candidates, key=lambda item: (item[0], item[1]))
    as_of_match = REPORT_NAME.fullmatch(path.name)
    assert as_of_match is not None
    return path, as_of_match.group("as_of"), spec_hash


def _public_recommendations(report_path: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for row in _read_csv(report_path):
        item: dict[str, object] = {
            field: (row.get(field) or "").strip() or None for field in TEXT_FIELDS
        }
        item.update({field: _number(row.get(field)) for field in NUMERIC_FIELDS})
        if item.get("symbol"):
            items.append(item)
    return items


def _public_strategy() -> list[dict[str, object]]:
    """Expose the active configured scoring categories and their member factors."""
    config = AppConfig.load(ROOT / "config" / "default.toml")
    weights = config.rank.as_dict()
    total = sum(weights.values())
    if total <= 0:
        return []

    return [
        {
            "key": category,
            "label": CATEGORY_LABELS.get(category, category),
            "weight": weight / total,
            "factors": [
                FACTOR_LABELS.get(factor, factor)
                for factor in RANK_MEMBERS.get(category, ())
            ],
        }
        for category, weight in sorted(weights.items(), key=lambda item: -item[1])
    ]


def _screening_rule_text(line: str) -> str | None:
    text = line.removeprefix("- ").strip()
    if text.startswith("数据截至 "):
        return None
    if text == "剔除 ST / 退市风险名称":
        return "剔除 ST 与退市风险名称"

    sort_match = re.fullmatch(r"按 (\S+) (降|升)序取前 (\d+) 条", text)
    if sort_match:
        sort_key, direction, limit = sort_match.groups()
        sort_label = {"score": "综合得分", "ret_20": "20 日收益"}.get(sort_key, sort_key)
        return f"按{sort_label}{'降' if direction == '降' else '升'}序，最多展示 {limit} 只"

    filter_match = re.fullmatch(r"([a-zA-Z0-9_]+) ∈ (.+)", text)
    if not filter_match:
        return text or None

    field, condition = filter_match.groups()
    label = SCREEN_FILTER_LABELS.get(field, FILTER_FIELDS.get(field, field))
    bound_match = re.fullmatch(
        r"([≥≤])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)",
        condition,
        re.IGNORECASE,
    )
    if not bound_match:
        return f"{label}：{condition}"

    operator, raw_value = bound_match.groups()
    value = float(raw_value)
    if field == "close_position_250":
        return f"{label} {operator} {value * 100:.0f}%"
    if field == "dist_to_52w_high":
        threshold = f"{value * 100:.0f}%"
        clarification = f"（低于高点至少 {abs(value) * 100:.0f}%）" if value < 0 else ""
        return f"{label} {operator} {threshold}{clarification}"
    if field == "turnover_value_20":
        return f"{label} {operator} {value / 10_000:,.0f} 万元"
    if field == "history_days":
        return f"{label} {operator} {value:g} 个交易日"
    if field == "listed_days":
        return f"{label} {operator} {value:g} 天"
    if field == "volume_trend_20":
        return f"{label} {operator} {value:.2f} 倍"
    return f"{label} {operator} {value:g}"


def _public_screening_rules(report_path: Path | None) -> list[str]:
    if report_path is None:
        return []
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        return []

    lines: list[str] = []
    in_filters = False
    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped == "## 筛选条件":
            in_filters = True
            continue
        if in_filters and stripped.startswith("## "):
            break
        if in_filters and stripped.startswith("- "):
            formatted = _screening_rule_text(stripped)
            if formatted:
                lines.append(formatted)
    return lines


def _public_kline_bars(
    market_path: Path,
    recommendations: list[dict[str, object]],
    as_of: str | None,
    window: int = 120,
) -> dict[str, list[list[object]]]:
    """Read a small, recent OHLC window for only the displayed candidates."""
    if not as_of or not market_path.is_file() or not recommendations:
        return {}

    symbols = sorted(
        {
            str(item.get("symbol"))
            for item in recommendations
            if re.fullmatch(r"\d{6}", str(item.get("symbol") or ""))
        }
    )
    if not symbols:
        return {}

    path_literal = market_path.resolve().as_posix().replace("'", "''")
    symbol_placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        with candidate_bars as (
            select
                lpad(cast(symbol as varchar), 6, '0') as symbol,
                cast(trade_date as date) as trade_date,
                cast(open as double) as open,
                cast(high as double) as high,
                cast(low as double) as low,
                cast(close as double) as close,
                cast(volume as bigint) as volume
            from read_parquet('{path_literal}')
            where lpad(cast(symbol as varchar), 6, '0') in ({symbol_placeholders})
                and cast(trade_date as date) <= cast(? as date)
        )
        select
            symbol,
            strftime(trade_date, '%Y-%m-%d') as trade_date,
            open,
            high,
            low,
            close,
            volume
        from candidate_bars
        qualify row_number() over (partition by symbol order by trade_date desc) <= ?
        order by symbol, trade_date
    """

    connection = duckdb.connect()
    try:
        rows = connection.execute(query, [*symbols, as_of, max(1, min(int(window), 250))]).fetchall()
    finally:
        connection.close()

    history: dict[str, list[list[object]]] = {symbol: [] for symbol in symbols}
    for symbol, trade_date, open_, high, low, close, volume in rows:
        history[symbol].append(
            [
                str(trade_date),
                _number(open_),
                _number(high),
                _number(low),
                _number(close),
                _number(volume),
            ]
        )
    return history


def _evaluation(reports_dir: Path, spec_hash: str) -> list[dict[str, object]]:
    summary_path = reports_dir / "forward" / "evaluation_summary.csv"
    if not summary_path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for source in _read_csv(summary_path):
        if spec_hash and (source.get("spec_hash") or "").strip() != spec_hash:
            continue
        rows.append(
            {
                field: _number(source.get(field))
                for field in EVALUATION_FIELDS
            }
        )
    return sorted(rows, key=lambda row: row.get("horizon") or 0)


def _report_notes(report_path: Path | None) -> tuple[list[str], int | None]:
    if report_path is None:
        return ["尚无可发布的推荐报告；首次完整收盘同步后会自动更新。"], None
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        return [], None
    source = markdown_path.read_text(encoding="utf-8")
    notes: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ⚠", "- ⚠️")):
            note = stripped[1:].lstrip(" ⚠️").strip()
            if note:
                notes.append(note)
    pool_match = re.search(r"通过全部条件[：:]\s*\**([\d,]+)", source)
    qualified_count = int(pool_match.group(1).replace(",", "")) if pool_match else None
    return notes[:8], qualified_count


def build_site(data_dir: Path, source_dir: Path, output_dir: Path) -> dict[str, object]:
    reports_dir = data_dir / "reports"
    latest = _latest_recommendation_report(reports_dir) if reports_dir.is_dir() else None
    if latest:
        report_path, as_of, spec_hash = latest
        recommendations = _public_recommendations(report_path)
        evaluation = _evaluation(reports_dir, spec_hash)
        notes, qualified_count = _report_notes(report_path)
        bars_by_symbol = _public_kline_bars(
            data_dir / "market" / "daily.parquet", recommendations, as_of
        )
        for item in recommendations:
            item["bars"] = bars_by_symbol.get(str(item.get("symbol") or ""), [])
        if recommendations and not any(item["bars"] for item in recommendations):
            notes.append("候选个股的近期日线数据暂缺，K 线图表当前无法显示。")
    else:
        report_path, as_of, spec_hash = None, None, ""
        recommendations, evaluation = [], []
        notes, qualified_count = _report_notes(None)
        bars_by_symbol = {}

    payload: dict[str, object] = {
        "schema_version": 2,
        "title": "每日股票候选",
        "as_of": as_of,
        "spec_hash": spec_hash or None,
        "published_at": datetime.now(timezone(timedelta(hours=8))).isoformat(
            timespec="minutes"
        ),
        "qualified_count": qualified_count,
        "recommendations": recommendations,
        "evaluation": evaluation,
        "strategy": _public_strategy(),
        "screening_rules": _public_screening_rules(report_path),
        "notes": notes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
    (output_dir / "data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "site")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    payload = build_site(args.data_dir, args.source_dir, args.output_dir)
    print(
        "Public site generated: "
        f"as_of={payload['as_of'] or 'none'}, "
        f"recommendations={len(payload['recommendations'])}, "
        f"kline_symbols={sum(bool(item.get('bars')) for item in payload['recommendations'])}, "
        f"evaluation_rows={len(payload['evaluation'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
