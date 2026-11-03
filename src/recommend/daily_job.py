"""The production daily path: prices -> fixed scores -> immutable signal -> validation."""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

from .bars import advance_bars
from .daily_data import atomic_json, fetch_quotes, refresh_benchmark, save_quote_snapshot
from .daily_strategy import (
    FACTOR_LABELS,
    MAIN_BOARD,
    build_features,
    canonical_hash,
    freeze_signal,
    load_protocol,
    select_daily,
)
from .daily_validation import evaluate_daily
from .locking import sync_lock
from .market import SHANGHAI_TIMEZONE, MarketStore, is_forward_eod_ready


def run_daily(
    root: Path, *, sync: bool = True, workers: int = 6, freeze_saved: bool = False
) -> dict:
    protocol = load_protocol(root / "config/daily_strategy.json")
    directory = root / "data/reports/daily-v2"
    directory.mkdir(parents=True, exist_ok=True)
    issues = []
    now = datetime.now(SHANGHAI_TIMEZONE)
    intraday = now.weekday() < 5 and time(9) <= now.time() < time(15, 30)
    if sync and intraday:
        print("Market session: rebuilding the completed signal without network", flush=True)
        sync = False
    with sync_lock(root / "logs/sync.lock"):
        with MarketStore(root / "data/market/daily.parquet") as store:
            meta = store.symbol_meta(root / "data/reference/instrument_master.parquet")
            meta = meta.loc[meta.symbol.map(lambda s: bool(MAIN_BOARD.fullmatch(s)))].copy()
            meta = meta.loc[~meta.name.fillna("").str.contains("退", regex=False)]
            expected = sorted(meta.symbol.unique().tolist())
            if not expected:
                raise ValueError("Local instrument reference has no main-board stocks")
            if sync:
                print(f"[1/4] Updating {len(expected)} main-board price histories", flush=True)
                advance = advance_bars(
                    store, symbols=expected, workers=max(1, min(workers, 6)), progress_every=500
                )
                if not advance.rows and not advance.up_to_date:
                    issues.extend(advance.notes or ["行情更新未完成"])
                if advance.failed:
                    issues.append(f"行情请求失败 {len(set(advance.failed))} 只；缺失股票不参与评分")
            as_of = store.latest_complete_trade_date()
            days = store.trade_dates(end=as_of)
            bars = store.bars(end=as_of, window=270)
        # Migration/recovery may restore an older market artifact than a newly
        # deployed frozen signal. Never replace a newer verified signal with an
        # older recomputation while waiting for the next EOD sync.
        previous_path = directory / "latest.json"
        if previous_path.exists():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if previous.get("as_of", "") > as_of and previous.get(
                "protocol_hash"
            ) == canonical_hash(protocol):
                signal_path = directory / (previous.get("signal_file") or "absent")
                if signal_path.is_file():
                    frozen = json.loads(signal_path.read_text(encoding="utf-8"))
                    checksum = frozen.pop("signal_hash", None)
                    if checksum != canonical_hash(frozen):
                        raise ValueError("Recovered signal integrity failure")
                    previous["issues"] = list(
                        dict.fromkeys(
                            [
                                *previous.get("issues", []),
                                f"恢复行情归档截至 {as_of}；保留较新的冻结记录，等待收盘同步追平",
                            ]
                        )
                    )
                    following = datetime.fromisoformat(previous["as_of"]).date()
                    following += timedelta(days=1)
                    while following.weekday() >= 5:
                        following += timedelta(days=1)
                    fresh_until = datetime.combine(following, time(15, 30), SHANGHAI_TIMEZONE)
                    previous["state"] = "live" if now < fresh_until else "historical"
                    atomic_json(previous_path, previous)
                    return previous
        # No fundamental, theme, announcement, financial database or model call occurs here.
        quote_path = directory / "quotes" / f"{as_of}.json"
        if sync:
            print(f"[2/4] Capturing dated raw quotes for {as_of}", flush=True)
            raw = fetch_quotes(expected, as_of, datetime.now(SHANGHAI_TIMEZONE))
            save_quote_snapshot(directory / "quotes", raw)
        if quote_path.exists():
            raw = json.loads(quote_path.read_text(encoding="utf-8"))
            status = pd.DataFrame(
                list(raw["quotes"].values()),
                columns=["symbol", "name", "status_date", "open", "volume"],
            )
            status["quote_active"] = status.volume.gt(0) & status.open.gt(0)
            status = status[["symbol", "name", "status_date", "quote_active"]]
            meta = meta.drop(columns="name").merge(status, on="symbol", how="left")
        else:
            raw = {"coverage": 0, "sha256": None}
            # Historical preview only. It is never written into the forward ledger.
            meta["status_date"] = as_of
            issues.append("缺少同日原始报价留证；当前仅为历史试算")
        industry_path = root / "data/reference/industry_master.csv"
        if industry_path.exists():
            industry = pd.read_csv(industry_path, dtype={"symbol": str})
            if {"symbol", "industry"} <= set(industry):
                industry["symbol"] = industry.symbol.astype(str).str.zfill(6)
                meta = meta.merge(
                    industry[["symbol", "industry"]].drop_duplicates("symbol"),
                    on="symbol",
                    how="left",
                )
        features = build_features(bars, as_of, protocol["minimum_history"])
        selection = select_daily(features, meta, protocol, as_of)
        current_bars = bars.loc[pd.to_datetime(bars.trade_date).dt.strftime("%Y-%m-%d").eq(as_of)]
        # Explicit no-trade quotes are not missing price downloads. Coverage is
        # measured against stocks the quote source says actually traded that day.
        active_expected = {
            s for s, q in raw.get("quotes", {}).items() if q["volume"] > 0 and q["open"] > 0
        }
        coverage_universe = active_expected or set(expected)
        fresh_codes = set(current_bars.symbol.astype(str)) & coverage_universe
        bar_coverage = len(fresh_codes) / max(len(coverage_universe), 1)
        selection["inputs"] = {
            "quote_sha256": raw.get("sha256"),
            "bar_coverage": bar_coverage,
            "quote_coverage": raw.get("coverage", 0),
            "universe_reference": "data/reference/instrument_master.parquet",
            "industry_reference": "data/reference/industry_master.csv",
        }
        healthy = (
            raw.get("coverage", 0) >= protocol["minimum_quote_coverage"]
            and bar_coverage >= protocol["minimum_bar_coverage"]
        )
        now = datetime.now(SHANGHAI_TIMEZONE)
        timely = is_forward_eod_ready(as_of, now)
        forward_path = None
        if (sync or freeze_saved) and healthy and timely:
            forward_path = freeze_signal(directory / "signals", selection, protocol, now)
            signal = json.loads(forward_path.read_text(encoding="utf-8"))
            selection = signal["selection"]
        else:
            existing = directory / "signals" / f"{as_of}__{canonical_hash(protocol)[:16]}.json"
            if existing.exists():
                forward_path = existing
                selection = json.loads(existing.read_text(encoding="utf-8"))["selection"]
            if not healthy:
                issues.append("行情/同日报价覆盖未达到95%，不新增前向推荐留证")
            if not timely:
                issues.append("已错过该日推荐的前向留证时限；本次不补造历史推荐")
        print("[3/4] Updating official benchmark and forward cohort ledger", flush=True)
        if sync:
            try:
                refresh_benchmark(directory / "benchmark.json", as_of, now)
            except Exception as error:  # noqa: BLE001 - missing benchmarks stay visibly missing
                issues.append(f"指数更新失败，超额保持待核对：{type(error).__name__}: {error}")
        validation = evaluate_daily(directory, days, protocol)
        following = datetime.fromisoformat(as_of).date() + timedelta(days=1)
        while following.weekday() >= 5:
            following += timedelta(days=1)
        fresh_until = datetime.combine(following, time(15, 30), SHANGHAI_TIMEZONE)
        state = "live" if forward_path and now < fresh_until and healthy else "historical"
        published = {
            "schema_version": 3,
            "title": "每日五选",
            "published_at": now.isoformat(),
            "as_of": as_of,
            "state": state,
            "protocol": protocol,
            "protocol_hash": canonical_hash(protocol),
            "selection": selection,
            "validation": validation,
            "issues": issues,
            "signal_file": f"signals/{forward_path.name}" if forward_path else None,
        }
        atomic_json(directory / "latest.json", published)
        print(
            f"[4/4] {as_of}: {len(selection['picks'])} picks; "
            f"state={state}; mature={validation['summary']['matured']}",
            flush=True,
        )
        return published


def public_daily_payload(data_dir: Path, source_dir: Path, output_dir: Path) -> dict:
    """Export only the daily product, frozen signals and public cohort results."""
    import shutil

    directory = data_dir / "reports/daily-v2"
    payload = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
    payload["selection"].pop("pool_symbols", None)
    payload["strategy"] = [
        {"key": k, "label": FACTOR_LABELS[k], "weight": v}
        for k, v in payload["protocol"]["weights"].items()
    ]
    with MarketStore(data_dir / "market/daily.parquet") as market:
        for pick in payload["selection"]["picks"]:
            history = market.symbol_bars(pick["symbol"], end=payload["as_of"], window=60)
            pick["history"] = [
                {"day": str(r.trade_date)[:10], "close": float(r.close)}
                for r in history.itertuples(index=False)
            ]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
    for signal in (directory / "signals").glob("*.json"):
        target = output_dir / "signals" / signal.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(signal, target)
    atomic_json(output_dir / "data.json", payload)
    # Local file opening also works; JSON is still the canonical downloadable snapshot.
    (output_dir / "daily-data.js").write_text(
        "window.DAILY_PICKS = " + json.dumps(payload, ensure_ascii=False, allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    return payload
