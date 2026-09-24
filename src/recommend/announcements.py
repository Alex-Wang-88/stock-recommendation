"""Incremental public-exchange announcement archive and review queue.

Only exchange listing metadata is downloaded. Clear event titles are categorized
locally; a small queue of ambiguous titles is exported for the scheduled Codex
Plus task to classify. Announcement signals are report annotations and never
enter ranking, screening, or price calculations.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd

from .archive import ArchiveStore
from .http import FetchError, ThrottledClient

SSE_API = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
SSE_REFERER = "https://www.sse.com.cn/disclosure/listedinfo/announcement/"
CNINFO_API = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_REFERER = "https://www.cninfo.com.cn/new/disclosure/stock?code=000001"

PAGE_SIZE = 100
CNINFO_PAGE_SIZE = 30
MAX_PAGES_PER_EXCHANGE = 400
INITIAL_LOOKBACK_DAYS = 7
OVERLAP_DAYS = 5
CLASSIFICATION_BATCH_SIZE = 25

SSE_A_SHARE_PREFIXES = ("600", "601", "603", "605", "688", "689")
SZSE_A_SHARE_PREFIXES = ("000", "001", "002", "003", "300", "301")

CATEGORIES = {
    "shareholder_reduction",
    "shareholder_increase",
    "unlock",
    "earnings",
    "dividend",
    "repurchase",
    "pledge",
    "major_contract",
    "regulatory",
    "index_change",
    "major_event",
    "uncertain",
    "ignore",
}

CATEGORY_LABELS = {
    "shareholder_reduction": "股东减持",
    "shareholder_increase": "股东增持",
    "unlock": "限售股解禁",
    "earnings": "业绩与定期报告",
    "dividend": "分红派息",
    "repurchase": "股份回购",
    "pledge": "股权质押",
    "major_contract": "重大合同/中标",
    "regulatory": "监管与合规",
    "index_change": "指数成份调整",
    "major_event": "重大事项",
    "uncertain": "标题信息不足",
    "ignore": "忽略",
}

SIGNALS = {
    "shareholder_reduction": "risk",
    "unlock": "risk",
    "pledge": "event_watch",
    "regulatory": "risk",
    "shareholder_increase": "event_watch",
    "earnings": "event_watch",
    "dividend": "event_watch",
    "repurchase": "event_watch",
    "major_contract": "event_watch",
    "index_change": "event_watch",
    "major_event": "event_watch",
    "uncertain": "none",
    "ignore": "none",
}

# Exact title phrases that can be categorized without a model. Ordered so a
# specific risk category wins before the broader major-event category.
RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("shareholder_reduction", ("减持",)),
    ("shareholder_increase", ("增持",)),
    ("unlock", ("解除限售", "限售股上市流通", "解禁")),
    (
        "regulatory",
        ("立案", "行政处罚", "监管措施", "问询函", "关注函", "纪律处分", "退市风险"),
    ),
    (
        "earnings",
        (
            "业绩预告",
            "业绩快报",
            "业绩说明会",
            "年度报告",
            "半年度报告",
            "季度报告",
            "年报",
            "半年报",
            "一季报",
            "三季报",
        ),
    ),
    ("dividend", ("利润分配", "现金分红", "权益分派", "派息", "分红")),
    ("repurchase", ("回购",)),
    ("pledge", ("质押",)),
    ("index_change", ("指数样本", "指数成份", "成份股调整", "样本股调整")),
    ("major_contract", ("重大合同", "中标", "重大项目")),
    (
        "major_event",
        (
            "重大资产重组",
            "控制权变更",
            "签署协议",
            "重大诉讼",
            "资金占用",
            "重大事项",
            "违规担保",
        ),
    ),
)

AMBIGUOUS_TERMS = (
    "股份变动",
    "股份转让",
    "股权转让",
    "实际控制人变更",
    "合同终止",
    "终止合作",
    "诉讼",
    "仲裁",
    "冻结",
    "异常波动",
    "关联交易",
    "重大变更",
    "承诺",
    "对外担保",
    "资金拆借",
    "募集资金用途变更",
)


def classify_title(title: str) -> tuple[str | None, str, str, str, float | None]:
    """Return category, signal, summary, status, confidence for one title."""

    compact = re.sub(r"\s+", "", str(title or ""))
    for category, phrases in RULES:
        if any(phrase in compact for phrase in phrases):
            label = CATEGORY_LABELS[category]
            return category, SIGNALS[category], f"标题命中本地规则：{label}", "classified", 1.0
    if any(term in compact for term in AMBIGUOUS_TERMS):
        return None, "none", "", "pending", None
    return "ignore", "none", "", "ignored", 1.0


def queue_path(reports_dir: str | Path) -> Path:
    return Path(reports_dir) / "announcements" / "pending.json"


def _local_today() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def _iso_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        try:
            timestamp = float(value)
            if timestamp > 10_000_000_000:
                timestamp /= 1000
            return datetime.fromtimestamp(timestamp, UTC).astimezone(
                ZoneInfo("Asia/Shanghai")
            ).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    match = re.search(r"(20\d{2})[-/]?(\d{2})[-/]?(\d{2})", str(value))
    if not match:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3])).isoformat()
    except ValueError:
        return None


def _symbol(value: object) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits.zfill(6) if len(digits) == 6 else None


def _official_url(exchange: str, value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("finalpage/"):
        raw = urljoin("https://static.cninfo.com.cn/", raw)
    base = "https://www.sse.com.cn/" if exchange == "SSE" else "https://www.szse.cn/"
    if raw.startswith("//"):
        raw = "https:" + raw
    elif not raw.startswith(("https://", "http://")):
        raw = urljoin(base, raw)
    host = (urlparse(raw).hostname or "").lower()
    allowed = (
        host.endswith("cninfo.com.cn")
        or (host.endswith("sse.com.cn") if exchange == "SSE" else host.endswith("szse.cn"))
    )
    return raw if allowed else ""


def _stable_id(
    exchange: str, row: dict[str, Any], url: str, symbol: str, day: str, title: str
) -> str:
    for key in ("BULLETIN_ID", "ID", "id", "announcementId", "annId", "docId", "doc_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    identity = url or f"{symbol}|{day}|{title}"
    return hashlib.sha256(f"{exchange}|{identity}".encode("utf-8")).hexdigest()[:32]


def _sse_rows(
    client: ThrottledClient, start: str, end: str
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for page_no in range(1, MAX_PAGES_PER_EXCHANGE + 1):
        params = {
            "isPagination": "true",
            "productId": "",
            "keyWord": "",
            "securityType": "0101,120100,020100,020200,120200",
            "reportType2": "",
            "START_DATE": start,
            "END_DATE": end,
            "beginDate": start,
            "endDate": end,
            "pageHelp.pageSize": PAGE_SIZE,
            "pageHelp.pageNo": page_no,
            "pageHelp.beginPage": page_no,
            "pageHelp.cacheSize": 1,
            "pageHelp.endPage": page_no,
        }
        response = client.get(SSE_API, params=params, headers={"Referer": SSE_REFERER})
        try:
            try:
                payload = json.loads(response.content.decode("utf-8"))
            except UnicodeDecodeError:
                payload = json.loads(response.content.decode("gbk"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FetchError(f"上交所公告 JSON 无法解码：{error}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("pageHelp"), dict):
            raise FetchError("上交所公告响应缺少 pageHelp")
        page = payload["pageHelp"]
        rows = page.get("data") or []
        if not isinstance(rows, list):
            raise FetchError("上交所公告响应缺少分页 data")
        all_rows.extend(row for row in rows if isinstance(row, dict))

        dates = [
            parsed
            for row in rows
            if (parsed := _iso_date(row.get("SSEDATE") or row.get("ADDDATE")))
        ]
        if dates and min(dates) < start:
            break
        page_count = int(page.get("pageCount") or 0)
        actual_page_size = int(page.get("pageSize") or PAGE_SIZE)
        if not rows or len(rows) < actual_page_size or (page_count and page_no >= page_count):
            break
    else:
        raise FetchError(f"上交所公告超过 {MAX_PAGES_PER_EXCHANGE} 页，未确认完整性")
    return all_rows


def _post_cninfo_json(client: ThrottledClient, page_no: int, start: str, end: str) -> Any:
    if client.session is None:
        raise FetchError("HTTP 会话尚未初始化")
    payload = {
        "pageNum": str(page_no),
        "pageSize": str(CNINFO_PAGE_SIZE),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": "",
        "searchkey": "",
        "secid": "",
        "category": "",
        "seDate": f"{start}~{end}",
    }
    last_error: Exception | None = None
    for attempt in range(1, client.retries + 1):
        client._wait()
        try:
            response = client.session.post(
                CNINFO_API,
                data=payload,
                headers={
                    "Referer": CNINFO_REFERER,
                    "Origin": "https://www.cninfo.com.cn",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=client.timeout,
            )
            client.last_call = time.time()
            if response.status_code >= 400:
                raise FetchError(f"巨潮资讯公告 HTTP {response.status_code}")
            try:
                return json.loads(response.content.decode("utf-8"))
            except UnicodeDecodeError:
                return json.loads(response.content.decode("gbk"))
        except Exception as error:  # noqa: BLE001 - 按共享 HTTP 策略重试一次来源
            client.last_call = time.time()
            last_error = error
            if attempt < client.retries:
                time.sleep(client.backoff * attempt)
    raise FetchError(f"巨潮资讯公告请求失败：{last_error}") from last_error


def _cninfo_rows(client: ThrottledClient, start: str, end: str) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for page_no in range(1, MAX_PAGES_PER_EXCHANGE + 1):
        payload = _post_cninfo_json(client, page_no, start, end)
        rows = payload.get("announcements") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise FetchError("巨潮资讯公告响应缺少 announcements 列表")
        all_rows.extend(row for row in rows if isinstance(row, dict))
        dates = [
            parsed
            for row in rows
            if (parsed := _iso_date(row.get("announcementTime")))
        ]
        if dates and min(dates) < start:
            break
        total_pages = int(payload.get("totalpages") or 0)
        if not rows or len(rows) < CNINFO_PAGE_SIZE or (total_pages and page_no >= total_pages):
            break
    else:
        raise FetchError(f"巨潮资讯公告超过 {MAX_PAGES_PER_EXCHANGE} 页，未确认完整性")
    return all_rows


def _normalize_row(
    exchange: str, row: dict[str, Any], start: str, end: str
) -> dict[str, Any] | None:
    symbol = _symbol(
        row.get("SECURITY_CODE")
        or row.get("secCode")
        or row.get("stockCode")
        or row.get("securityCode")
    )
    if symbol is None:
        return None
    exchange_prefixes = (
        SSE_A_SHARE_PREFIXES if exchange == "SSE" else SZSE_A_SHARE_PREFIXES
    )
    if not symbol.startswith(exchange_prefixes):
        return None
    if exchange == "SSE":
        day = _iso_date(row.get("SSEDATE") or row.get("ADDDATE"))
        title = str(row.get("TITLE") or row.get("BULLETIN_HEADING") or "").strip()
        name = str(row.get("SECURITY_NAME") or "").strip()
        raw_url = row.get("URL") or row.get("url") or row.get("FILE_PATH")
    else:
        day = _iso_date(
            row.get("announcementTime")
            or row.get("publishTime")
            or row.get("publishDate")
            or row.get("publish_time")
            or row.get("time")
        )
        title = str(row.get("title") or row.get("announcementTitle") or "").strip()
        name = str(
            row.get("secName") or row.get("stockName") or row.get("companyName") or ""
        ).strip()
        raw_url = (
            row.get("attachPath")
            or row.get("adjunctUrl")
            or row.get("attachmentUrl")
            or row.get("pdfUrl")
            or row.get("url")
            or row.get("urlPath")
        )
    if not day or not start <= day <= end or not title:
        return None
    url = _official_url(exchange, raw_url)
    category, signal, summary, status, confidence = classify_title(title)
    return {
        "exchange": exchange,
        "announcement_id": _stable_id(exchange, row, url, symbol, day, title),
        "symbol": symbol,
        "company_name": name,
        "announcement_date": day,
        "title": title[:500],
        "url": url,
        "category": category,
        "signal": signal,
        "summary": summary,
        "classification_status": status,
        "classification_source": "local-rule" if status != "pending" else None,
        "confidence": confidence,
        "fetched_at": datetime.now(UTC).replace(tzinfo=None),
        "classified_at": datetime.now(UTC).replace(tzinfo=None) if status != "pending" else None,
    }


def _last_success_date(archive: ArchiveStore, exchange: str) -> date | None:
    row = archive.conn.execute(
        "select last_success_date from announcement_sync_state where exchange = ?",
        [exchange],
    ).fetchone()
    if not row or row[0] is None:
        return None
    value = row[0]
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def _start_date(archive: ArchiveStore, exchange: str, end: date) -> date:
    previous = _last_success_date(archive, exchange)
    if previous is None:
        start = end - timedelta(days=INITIAL_LOOKBACK_DAYS)
    else:
        start = previous - timedelta(days=OVERLAP_DAYS)
    if start > end:
        start = end - timedelta(days=OVERLAP_DAYS)
    return start


def _store_success(
    archive: ArchiveStore, exchange: str, end: str, sync_run_id: str
) -> None:
    archive.conn.execute(
        "insert or replace into announcement_sync_state "
        "(exchange, last_success_date, updated_at, sync_run_id) "
        "values (?, ?, current_timestamp, ?)",
        [exchange, date.fromisoformat(end), sync_run_id],
    )


def write_pending_queue(
    archive: ArchiveStore, path: str | Path, *, limit: int = CLASSIFICATION_BATCH_SIZE
) -> int:
    """Write a small JSON queue for the existing scheduled Plus task."""

    pending = archive.pending_company_announcements(limit)
    items: list[dict[str, str]] = []
    for row in pending.to_dict("records"):
        value = row.get("announcement_date")
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        items.append(
            {
                "exchange": str(row.get("exchange") or ""),
                "announcement_id": str(row.get("announcement_id") or ""),
                "symbol": str(row.get("symbol") or ""),
                "company_name": str(row.get("company_name") or ""),
                "announcement_date": str(value or ""),
                "title": str(row.get("title") or ""),
                "url": str(row.get("url") or ""),
            }
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "allowed_categories": sorted(CATEGORIES),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return len(items)


def sync_public_announcements(
    archive: ArchiveStore,
    client: ThrottledClient,
    *,
    as_of: str,
    sync_run_id: str,
    reports_dir: str | Path,
) -> tuple[str, str, str]:
    """Incrementally sync exchange listings, preserve gaps for later retry."""

    end = max(date.fromisoformat(as_of[:10]), _local_today())
    exchange_fetchers = (("SSE", _sse_rows), ("SZSE", _cninfo_rows))
    summaries: list[str] = []
    failures: list[str] = []
    fetched_total = 0
    for exchange, fetcher in exchange_fetchers:
        try:
            chunk_start = _start_date(archive, exchange, end)
            exchange_count = 0
            while chunk_start <= end:
                chunk_end = min(chunk_start + timedelta(days=6), end)
                start_text, end_text = chunk_start.isoformat(), chunk_end.isoformat()
                rows = fetcher(client, start_text, end_text)
                normalized = [
                    record
                    for row in rows
                    if (
                        record := _normalize_row(exchange, row, start_text, end_text)
                    )
                    is not None
                ]
                if normalized:
                    frame = pd.DataFrame(normalized)
                    frame["sync_run_id"] = sync_run_id
                    archive.insert_new_announcements(frame)
                _store_success(archive, exchange, end_text, sync_run_id)
                exchange_count += len(normalized)
                chunk_start = chunk_end + timedelta(days=1)
            fetched_total += exchange_count
            summaries.append(f"{exchange} 拉取 {exchange_count} 条")
        except Exception as error:  # noqa: BLE001 - one exchange must not block the other
            failures.append(f"{exchange}: {type(error).__name__}: {error}")
            progress = _last_success_date(archive, exchange)
            cursor_note = f"已追到 {progress.isoformat()}" if progress else "游标未建立"
            summaries.append(f"{exchange} 失败（{cursor_note}）")

    pending_path = queue_path(reports_dir)
    queued = write_pending_queue(archive, pending_path)
    pending_total = int(
        archive.conn.execute(
            "select count(*) from company_announcements where classification_status = 'pending'"
        ).fetchone()[0]
    )
    summary = (
        f"{'；'.join(summaries)}；本次拉取公告元数据 {fetched_total} 条，"
        f"未分类积压 {pending_total} 条，本轮导出 {queued} 条待 Plus 分类"
    )
    detail = (
        f"增量游标按交易所分别保存；断档会从上次成功日继续追补，"
        f"并重抓最近 {OVERLAP_DAYS} 天。队列文件：{pending_path}。"
    )
    if failures:
        return "warning", summary, "\n".join(failures) + "\n" + detail
    return "done", summary, detail


def apply_classifications(
    archive: ArchiveStore, input_path: str | Path, *, reports_dir: str | Path
) -> tuple[int, int, int]:
    """Apply the scheduled model's title-only labels to pending archive rows."""

    payload = json.loads(Path(input_path).read_text(encoding="utf-8-sig"))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("分类文件必须包含 items 数组")
    if len(items) > CLASSIFICATION_BATCH_SIZE:
        raise ValueError(f"单次最多应用 {CLASSIFICATION_BATCH_SIZE} 条分类结果")

    applied = ignored = unmatched = 0
    conn = archive.conn
    conn.execute("begin transaction")
    try:
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("items 中的每项都必须是对象")
            exchange = str(item.get("exchange") or "").upper()
            announcement_id = str(item.get("announcement_id") or "")
            category = str(item.get("category") or "").strip().lower()
            summary = re.sub(r"\s+", " ", str(item.get("summary") or "")).strip()[:180]
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError) as error:
                raise ValueError(f"公告 {announcement_id} 的 confidence 无效") from error
            if exchange not in {"SSE", "SZSE"} or not announcement_id:
                raise ValueError("分类结果必须包含有效的 exchange 和 announcement_id")
            if category not in CATEGORIES:
                raise ValueError(f"公告 {announcement_id} 的 category 无效：{category}")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"公告 {announcement_id} 的 confidence 必须在 0 到 1 之间")
            if category != "ignore" and not summary:
                raise ValueError(f"公告 {announcement_id} 的 summary 不能为空")

            existing = conn.execute(
                "select title from company_announcements "
                "where exchange = ? and announcement_id = ? "
                "and classification_status = 'pending'",
                [exchange, announcement_id],
            ).fetchone()
            if not existing:
                unmatched += 1
                continue
            if category == "ignore":
                status, signal, stored_category = "ignored", "none", "ignore"
                stored_summary = ""
                ignored += 1
            else:
                stored_category = category if confidence >= 0.65 else "uncertain"
                status = "classified"
                signal = SIGNALS[stored_category]
                stored_summary = summary if stored_category != "uncertain" else "标题信息不足，未推断事件影响"
                applied += 1
            conn.execute(
                "update company_announcements set category = ?, signal = ?, summary = ?, "
                "classification_status = ?, classification_source = 'codex-plus', "
                "confidence = ?, classified_at = current_timestamp "
                "where exchange = ? and announcement_id = ? "
                "and classification_status = 'pending'",
                [
                    stored_category,
                    signal,
                    stored_summary,
                    status,
                    confidence,
                    exchange,
                    announcement_id,
                ],
            )
        write_pending_queue(archive, queue_path(reports_dir))
        conn.execute("commit")
    except Exception:
        conn.execute("rollback")
        raise
    return applied, ignored, unmatched


__all__ = [
    "CATEGORIES",
    "CATEGORY_LABELS",
    "CLASSIFICATION_BATCH_SIZE",
    "SIGNALS",
    "apply_classifications",
    "classify_title",
    "queue_path",
    "sync_public_announcements",
    "write_pending_queue",
]
