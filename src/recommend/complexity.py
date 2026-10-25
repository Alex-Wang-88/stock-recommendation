"""Public daily cross-market snapshots. No credentials, model scores, or synthetic fallback."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path
from time import sleep
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests

CST = timezone(timedelta(hours=8))
WINDOWS = (20, 60, 120)
THRESHOLD = 0.30
MIN_SAMPLES = 20
FACTORS = (
    {
        "id": "ndx",
        "label": "纳斯达克100",
        "symbol": "^NDX",
        "unit": "点",
        "provider": "Yahoo Finance",
        "kind": "index",
        "zone": "America/New_York",
    },
    {
        "id": "kospi",
        "label": "KOSPI",
        "symbol": "^KS11",
        "unit": "点",
        "provider": "Yahoo Finance",
        "kind": "index",
        "zone": "Asia/Seoul",
    },
    {
        "id": "cny",
        "label": "USD/CNY 参考汇率",
        "symbol": "USD/CNY",
        "unit": "CNY / USD",
        "provider": "Frankfurter / ECB",
        "kind": "fx",
        "zone": "Europe/Brussels",
    },
    {
        "id": "gold",
        "label": "COMEX 黄金近月期货",
        "symbol": "GC=F",
        "unit": "USD / 盎司",
        "provider": "Yahoo Finance",
        "kind": "future",
        "zone": "America/New_York",
    },
    {
        "id": "brent",
        "label": "布伦特近月期货",
        "symbol": "BZ=F",
        "unit": "USD / 桶",
        "provider": "Yahoo Finance",
        "kind": "future",
        "zone": "America/New_York",
    },
    {
        "id": "ust10y",
        "label": "美国10年期国债平价收益率",
        "symbol": "BC_10YEAR",
        "unit": "%",
        "provider": "U.S. Treasury",
        "kind": "yield",
        "zone": "America/New_York",
    },
)
SECTORS = (
    {"id": "semi", "label": "半导体ETF", "symbol": "sh512480"},
    {"id": "metals", "label": "有色金属ETF", "symbol": "sh512400"},
    {"id": "energy", "label": "全指能源ETF", "symbol": "sz159945"},
    {"id": "newenergy", "label": "新能源ETF", "symbol": "sh516160"},
    {"id": "consumer", "label": "主要消费ETF", "symbol": "sz159928"},
)


def finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean_rows(rows: list[dict], last_day: str) -> list[dict]:
    """Strict source dates, positive price levels, no repeated or future rows."""
    result = {}
    for row in rows:
        try:
            day = date.fromisoformat(row["date"]).isoformat()
        except (ValueError, TypeError, KeyError):
            continue
        value = finite(row.get("close"))
        if day > last_day or value is None or value <= 0:
            continue
        amount = finite(row.get("amount"))
        result[day] = {
            "date": day,
            "close": value,
            "amount": amount if amount is not None and amount >= 0 else None,
        }
    return [result[day] for day in sorted(result)]


def parse_tencent(payload: dict, symbol: str, last_day: str) -> list[dict]:
    node = (payload.get("data") or {}).get(symbol) or {}
    # Explicit raw-price request for every ETF; no silent adjustment-mode fallback.
    rows = node.get("day")
    if not isinstance(rows, list):
        raise ValueError("缺少未复权 day 日线")
    return clean_rows(
        [
            {
                "date": r[0],
                "close": r[2],
                "amount": finite(r[8]) * 10000 if len(r) > 8 and finite(r[8]) is not None else None,
            }
            for r in rows
            if isinstance(r, list) and len(r) >= 6
        ],
        last_day,
    )


def parse_yahoo(payload: dict, spec: dict, now: datetime) -> list[dict]:
    chart = payload.get("chart") or {}
    if chart.get("error") or not chart.get("result"):
        raise ValueError("行情源没有返回该系列")
    result = chart["result"][0]
    if result.get("meta", {}).get("symbol") != spec["symbol"]:
        raise ValueError("返回的标识与请求系列不一致")
    values = result["indicators"]["quote"][0]["close"]
    zone = ZoneInfo(spec["zone"])
    local_now = now.astimezone(zone)
    # Futures close/publication conventions vary: exclude the entire current source day.
    close_time = time(15, 45) if spec["id"] == "kospi" else time(16, 30)
    if spec["kind"] == "future":
        close_time = time(23, 59, 59)
    rows = []
    for stamp, value in zip(result.get("timestamp", []), values, strict=True):
        day = datetime.fromtimestamp(stamp, UTC).astimezone(zone).date()
        if day > local_now.date():
            continue
        if day == local_now.date() and local_now.time().replace(tzinfo=None) < close_time:
            continue
        rows.append({"date": day.isoformat(), "close": value})
    return clean_rows(rows, local_now.date().isoformat())


def parse_treasury(content: bytes, last_day: str) -> list[dict]:
    root = ET.fromstring(content)
    rows = []
    for props in root.iter():
        if props.tag.rsplit("}", 1)[-1] != "properties":
            continue
        fields = {node.tag.rsplit("}", 1)[-1]: node.text for node in props}
        if fields.get("NEW_DATE"):
            rows.append({"date": fields["NEW_DATE"][:10], "close": fields.get("BC_10YEAR")})
    return clean_rows(rows, last_day)


def previous_complete_day(now: datetime) -> str:
    local = now.astimezone(CST)
    day = (
        local.date()
        if local.time().replace(tzinfo=None) >= time(15, 30)
        else (local.date() - timedelta(days=1))
    )
    return day.isoformat()


def download_source(spec: dict, now: datetime, cache_dir: Path) -> dict:
    """One bounded serial request per series, plus two annual Treasury files."""
    cache_path = cache_dir / f"{spec['id']}.json"
    checked = now.isoformat(timespec="seconds")
    is_etf = spec.get("kind") == "etf"
    source_url = (
        f"https://gu.qq.com/{spec['symbol']}"
        if is_etf
        else "https://frankfurter.dev/"
        if spec["kind"] == "fx"
        else "https://home.treasury.gov/treasury-daily-interest-rate-xml-feed"
        if spec["kind"] == "yield"
        else f"https://finance.yahoo.com/quote/{quote(spec['symbol'], safe='')}/history/"
    )
    base = {
        **spec,
        "source_url": source_url,
        "checked_at": checked,
        "adjustment": "unadjusted_price" if is_etf else "source_close",
    }
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
    receipts = []

    def get(url: str, params: dict | None = None) -> requests.Response:
        for attempt in range(2):
            try:
                response = session.get(url, params=params, timeout=(5, 18))
                response.raise_for_status()
                break
            except requests.RequestException as error:
                status = error.response.status_code if error.response is not None else None
                if attempt or (status is not None and status < 500 and status != 429):
                    raise
                sleep(0.5)
        receipts.append(
            {"url": response.url, "sha256": hashlib.sha256(response.content).hexdigest()}
        )
        return response

    try:
        if is_etf:
            response = get(
                "https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get",
                {"param": f"{spec['symbol']},day,,,650,"},
            )
            rows = parse_tencent(response.json(), spec["symbol"], previous_complete_day(now))
        elif spec["kind"] == "fx":
            response = get(
                "https://api.frankfurter.dev/v2/providers/ecb/rates",
                {
                    "from": (now.date() - timedelta(days=900)).isoformat(),
                    "to": (now.date() - timedelta(days=1)).isoformat(),
                    "base": "USD",
                    "quotes": "CNY",
                },
            )
            raw = response.json()
            if not isinstance(raw, list) or any(
                r.get("base") != "USD" or r.get("quote") != "CNY" for r in raw
            ):
                raise ValueError("汇率系列标识不匹配")
            rows = clean_rows(
                [{"date": r["date"], "close": r["rate"]} for r in raw],
                (now.date() - timedelta(days=1)).isoformat(),
            )
        elif spec["kind"] == "yield":
            rows = []
            for year in (now.year - 2, now.year - 1, now.year):
                response = get(
                    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml",
                    {"data": "daily_treasury_yield_curve", "field_tdr_date_value": year},
                )
                # Conservative: current U.S. day is never considered a completed daily release.
                last = now.astimezone(ZoneInfo(spec["zone"])).date() - timedelta(days=1)
                rows.extend(parse_treasury(response.content, last.isoformat()))
            rows = clean_rows(rows, now.date().isoformat())
        else:
            response = get(
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                + quote(spec["symbol"], safe=""),
                {"range": "5y", "interval": "1d"},
            )
            rows = parse_yahoo(response.json(), spec, now)
        if len(rows) < 2:
            raise ValueError("有效完整日线少于两条")
        result = {
            **base,
            "status": "ok",
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "observations": rows,
            "receipts": receipts,
            "error": None,
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf8"
        )
        temporary.replace(cache_path)
        return result
    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        ET.ParseError,
    ) as exc:
        error = f"{type(exc).__name__}: {str(exc)[:240]}"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf8"))
            if cached.get("symbol") != spec["symbol"]:
                raise ValueError("缓存系列不匹配")
            if cached.get("adjustment") != base["adjustment"]:
                raise ValueError("缓存价格调整口径不匹配")
            rows = clean_rows(cached.get("observations", []), now.date().isoformat())
            if len(rows) < 2:
                raise ValueError("缓存无可用日线")
            return {**cached, **base, "status": "cached", "error": error, "observations": rows}
        except (OSError, ValueError, TypeError):
            return {
                **base,
                "status": "unavailable",
                "fetched_at": None,
                "observations": [],
                "receipts": [],
                "error": error,
            }
    finally:
        session.close()


def correlation(pairs: list[tuple[float, float]]) -> dict:
    n = len(pairs)
    result = {"rho": None, "n": n, "ci95": None}
    if n < MIN_SAMPLES:
        return result
    x, y = zip(*pairs, strict=True)
    if statistics.pstdev(x) < 1e-12 or statistics.pstdev(y) < 1e-12:
        return result
    rho = max(-1.0, min(1.0, statistics.correlation(x, y)))
    # Fisher interval is descriptive: no independence or multiple-testing claim.
    z = math.atanh(max(-0.999999, min(0.999999, rho)))
    margin = 1.96 / math.sqrt(n - 3)
    result.update(rho=rho, ci95=[math.tanh(z - margin), math.tanh(z + margin)])
    return result


def percentile(value: float | None, history: list[float]) -> float | None:
    if value is None or len(history) < 60:
        return None
    # Midrank, past observations only; current observation is not part of its reference sample.
    return (
        100
        * (sum(x < value for x in history) + 0.5 * sum(x == value for x in history))
        / len(history)
    )


def returns(rows: list[dict]) -> dict[str, float]:
    return {r["date"]: r["close"] / p["close"] - 1 for p, r in zip(rows, rows[1:], strict=False)}


def lagged_changes(source: dict, a_days: list[str]) -> dict[str, float]:
    """Align levels strictly before each A-share date, then difference on the A-share grid.

    A foreign holiday therefore contributes zero, not a duplicated prior daily return.
    This is date-lagged descriptive research, not point-in-time historical availability data.
    """
    rows = source.get("observations", [])
    dates = [r["date"] for r in rows]
    result = {}
    previous = None
    for day in a_days:
        index = bisect.bisect_left(dates, day) - 1
        current = rows[index] if index >= 0 else None
        if current and (date.fromisoformat(day) - date.fromisoformat(current["date"])).days > 7:
            current = None
        if current and previous:
            change = (
                (current["close"] - previous["close"]) * 100
                if source["kind"] == "yield"
                else current["close"] / previous["close"] - 1
            )
            result[day] = change
        previous = current
    return result


def window_network(sources: list[dict], sectors: list[dict], days: list[str], window: int) -> dict:
    selected = days[-window:]
    changes = {s["id"]: lagged_changes(s, days) for s in sources}
    common_days = set(days)
    sector_returns = {
        s["id"]: returns([r for r in s.get("observations", []) if r["date"] in common_days])
        for s in sectors
    }
    relations = []
    for sector in sectors:
        for source in sources:
            aligned = [
                (changes[source["id"]][d], sector_returns[sector["id"]][d])
                for d in selected
                if d in changes[source["id"]] and d in sector_returns[sector["id"]]
            ]
            stats = correlation(aligned)
            relations.append(
                {
                    "factor": source["id"],
                    "sector": sector["id"],
                    **stats,
                    "connected": stats["rho"] is not None and abs(stats["rho"]) >= THRESHOLD,
                }
            )
    valid = [r for r in relations if r["rho"] is not None]
    connected = [r for r in valid if r["connected"]]
    strongest = max(valid, key=lambda r: abs(r["rho"])) if valid else None
    return {
        "days": window,
        "start": selected[0] if selected else None,
        "end": selected[-1] if selected else None,
        "observed_days": len(selected),
        "relations": relations,
        "valid_pairs": len(valid),
        "possible_pairs": len(relations),
        "connected_pairs": len(connected),
        "density": len(connected) / len(valid) if valid else None,
        "average_strength": statistics.mean(abs(r["rho"]) for r in connected)
        if connected
        else None,
        "strongest": strongest,
    }


def basket_history(sectors: list[dict]) -> list[dict]:
    """Five explicitly named ETF proxies, never presented as all A-share stocks."""
    maps = [{r["date"]: r for r in s.get("observations", [])} for s in sectors]
    if len(maps) != len(SECTORS) or any(not m for m in maps):
        return []
    days = sorted(set.intersection(*(set(m) for m in maps)))
    result = []
    changes = [returns([m[d] for d in days]) for m in maps]
    for i, day in enumerate(days):
        if i < 19 or any(day not in r for r in changes):
            continue
        breadth = (
            100
            * sum(
                m[day]["close"] > statistics.mean(m[d]["close"] for d in days[i - 19 : i + 1])
                for m in maps
            )
            / len(maps)
        )
        dispersion = statistics.pstdev(r[day] for r in changes) * 100
        amounts = [m[day].get("amount") for m in maps]
        hhi = None
        if all(v is not None and v >= 0 for v in amounts) and sum(amounts) > 0:
            hhi = sum((v / sum(amounts)) ** 2 for v in amounts)
        pairwise = []
        for a in range(len(maps)):
            for b in range(a + 1, len(maps)):
                pairs = [
                    (changes[a][d], changes[b][d])
                    for d in days[max(0, i - 59) : i + 1]
                    if d in changes[a] and d in changes[b]
                ]
                rho = correlation(pairs)["rho"]
                if rho is not None:
                    pairwise.append(abs(rho))
        synchrony = sum(r >= THRESHOLD for r in pairwise) / len(pairwise) if pairwise else None
        previous = result[-252:]
        result.append(
            {
                "date": day,
                "breadth": breadth,
                "dispersion": dispersion,
                "hhi": hhi,
                "synchrony": synchrony,
                "dispersion_percentile": percentile(
                    dispersion, [r["dispersion"] for r in previous]
                ),
                "hhi_percentile": percentile(
                    hhi, [r["hhi"] for r in previous if r["hhi"] is not None]
                ),
                "sample": len(maps),
                "synchrony_pairs": len(pairwise),
            }
        )
    return result


def source_summary(source: dict, now: datetime) -> dict:
    result = {k: v for k, v in source.items() if k != "observations"}
    rows = source.get("observations", [])
    result.update(
        as_of=rows[-1]["date"] if rows else None,
        rows=len(rows),
        value=rows[-1]["close"] if rows else None,
        previous_date=rows[-2]["date"] if len(rows) >= 2 else None,
        change=None,
        age_days=None,
    )
    if len(rows) >= 2:
        result["change"] = (
            (rows[-1]["close"] - rows[-2]["close"]) * 100
            if source["kind"] == "yield"
            else (rows[-1]["close"] / rows[-2]["close"] - 1) * 100
        )
        result["age_days"] = (now.astimezone(CST).date() - date.fromisoformat(result["as_of"])).days
    result["stale"] = result["age_days"] is not None and result["age_days"] > 4
    result["published_at"] = None
    return result


def build_snapshot(all_sources: list[dict], now: datetime) -> dict:
    by_id = {s["id"]: s for s in all_sources}
    factors = [by_id[s["id"]] for s in FACTORS]
    sectors = [by_id[s["id"]] for s in SECTORS]
    date_sets = [set(r["date"] for r in s["observations"]) for s in sectors]
    # Use a common complete basket date, never mix fund dates in one observation.
    days = sorted(set.intersection(*date_sets)) if date_sets else []
    history = basket_history(sectors)
    summaries = [source_summary(s, now) for s in all_sources]
    problems = [
        f"{s['label']}：{'数据源失败，使用旧缓存' if s['status'] == 'cached' else '暂无可用数据'}"
        for s in summaries
        if s["status"] != "ok"
    ]
    problems.extend(
        f"{s['label']}：源交易日 {s['as_of']}，距今 {s['age_days']} 天"
        for s in summaries
        if s["stale"]
    )
    return {
        "schema_version": 1,
        "generated_at": now.isoformat(timespec="seconds"),
        "mode": "daily_public_snapshot",
        "warnings": problems,
        "factors": [s for s in summaries if s["id"] in {f["id"] for f in FACTORS}],
        "sectors": [s for s in summaries if s["id"] in {f["id"] for f in SECTORS}],
        "basket": {
            "expected": len(SECTORS),
            "available": sum(bool(x) for x in date_sets),
            "as_of": days[-1] if days else None,
            "latest": history[-1] if history else None,
            "previous": history[-2] if len(history) > 1 else None,
            "history": history,
        },
        "windows": {str(w): window_network(factors, sectors, days, w) for w in WINDOWS},
        "source_observations": {s["id"]: s["observations"] for s in all_sources},
        "method": {
            "correlation": "Pearson",
            "threshold": THRESHOLD,
            "minimum_samples": MIN_SAMPLES,
            "alignment": (
                "外部源日期严格早于A股交易日；先对齐价格水平，"
                "再计算A股网格上的变化；超过7日的旧值不配对。"
                "仅日期滞后描述统计，非历史点时回测。"
            ),
            "scope": (
                "五只行业/主题ETF代理篮子；未复权价格收益，除息有影响；"
                "不代表A股全市场或完整行业指数。"
            ),
            "confidence": ("Fisher变换近似95%区间；未校正自相关或多重比较，不是显著性承诺。"),
            "fx": "ECB每日USD/CNY参考汇率，不是离岸USD/CNH，也不是盘前实时报价。",
            "futures": "GC=F与BZ=F为供应商近月期货序列，不是现货；换月可能影响收益。",
            "publication": (
                "公开接口未提供可核验的逐历史发布时间，published_at为空；"
                "抓取时间不能视为历史可得时间。"
            ),
        },
    }


def write_snapshot(payload: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for filename, content in (
        ("complexity-data.json", text),
        ("complexity-data.js", "window.CX_SNAPSHOT = " + text.rstrip() + ";\n"),
    ):
        temporary = output / (filename + ".tmp")
        temporary.write_text(content, encoding="utf8")
        temporary.replace(output / filename)


def seed_cache(cache_dir: Path, snapshot_path: Path) -> None:
    """Bootstrap a first CI run from the last public real snapshot, retaining source times."""
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf8"))
        if payload.get("schema_version") != 1:
            return
        known = {s["id"]: s["symbol"] for s in (*FACTORS, *SECTORS)}
        for summary in (*payload["factors"], *payload["sectors"]):
            sid = summary["id"]
            target = cache_dir / f"{sid}.json" if sid in known else None
            if target is None or target.exists() or summary["symbol"] != known[sid]:
                continue
            if not summary.get("fetched_at"):
                continue
            rows = payload.get("source_observations", {}).get(sid, [])
            if len(rows) < 2:
                continue
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = {**summary, "observations": rows, "seeded_from": "bundled-public-snapshot"}
            target.write_text(
                json.dumps(cached, ensure_ascii=False, allow_nan=False), encoding="utf8"
            )
    except (OSError, ValueError, KeyError, TypeError):
        # Invalid/missing seed is not a usable source. A live request must still be attempted.
        return
