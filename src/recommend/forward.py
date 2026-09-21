"""前向纸盘 —— 把「每天跑一次筛选」的结果冻结留证，供日后回看。

为什么需要这个模块
----------------
本项目的数据分三档（见 README）：
* **T0/T1** 可回补 ⇒ 能回测。
* **T2** 只有当日快照、历史补不回来（题材热度、龙虎榜、两融、板块资金流）。
* **T3** 结构性不可回测（新闻文本、研报），永不进权重。

T2 意味着**没有任何历史样本**可以验证「按今天的题材热度选股是否有效」。
唯一能得到证据的办法是：从今天起，每个交易日**冻结一次**当时的输入与输出，
等若干交易日后拿实际走势对账。这件事**必须从今天开始做** —— 因为它无法补做。

三条机械保证
-----------
1. **spec 指纹不含 `as_of`**。前向测试要的是「同一个 spec 连续跑 N 天」；
   把每天都变的 as_of 算进指纹，等于每天都换了一个 spec，根本无法归组。
2. **记录里必须存 `close`**。留证只存代码是不够的：日后标的改名、退市、复权基准
   变化都会让「用现在的行情回算当时的推荐」失真。存下**当时看到的收盘价**才可对账。
3. **记录里必须存剔除日志与 note**。没有它，「那天只选出 2 只」到底是条件太严，
   还是某个因子整列缺失把池子清零了？—— 事后无法区分，整个样本作废。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .market import MarketStore
from .rank.score import explain
from .screen import Spec

JOURNAL_NAME = "journal.jsonl"
RECORD_SUFFIX = ".json"
EVALUATION_NAME = "evaluation.csv"
EVALUATION_SUMMARY_NAME = "evaluation_summary.csv"

# 对账窗口（交易日）。1 日 = 隔夜，5 日 = 一周，20 日 = 一个月（与调仓周期同量级）。
HORIZONS = (1, 5, 20)


def spec_fingerprint(spec: Spec) -> str:
    """spec 的稳定指纹，**不含 as_of**（理由见模块 docstring 第 1 条）。"""

    payload = spec.to_dict()
    payload.pop("as_of", None)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class ForwardRecord:
    """一次前向筛选的完整留证。"""

    as_of: str
    spec_hash: str
    spec: dict
    universe: int
    pool: int
    kept: int
    candidates: list[dict] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    run_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def build_record(result, *, run_at: str) -> ForwardRecord:
    """从一次 `ScreenResult` 构造留证记录。

    `candidates` 只留对账必需的因子摘要和操作参考字段 —— 完整因子明细已在同日的
    CSV 里，journal 是**索引**不是副本，塞满全部因子列会让它膨胀到读不动。
    """

    wanted = (
        "symbol",
        "name",
        "industry",
        "close",
        "score",
        "score_coverage",
        "score_industry",
        "score_growth",
        "score_quality",
        "score_growth_quality",
        "score_valuation",
        "score_momentum",
        "score_risk",
        "score_theme",
        "score_capital",
        "score_liquidity",
        "theme_tags",
        "reason",
        "ret_20",
        "relative_ret_20",
        "relative_ret_60",
        "ret_since_924",
        "volume_trend_20",
        "realized_vol_20",
        "downside_vol_20",
        "max_drawdown_60",
        "theme_heat_max",
        "theme_days_20",
        "lhb_net_buy_rel",
        "main_net_inflow",
        "turnover_value_20",
        "pe_ttm",
        "pe_industry_value",
        "eps_ttm",
        "roe_avg",
        "gross_profit_margin",
        "net_profit_margin",
        "revenue_yoy",
        "profit_growth_acceleration",
        "positive_growth_streak",
        "cfo_to_np",
        "liability_to_asset",
        "industry_score",
        "industry_leader_score",
        "operation_weekday",
        "operation_signal",
        "reference_price",
        "buy_low",
        "buy_high",
        "pullback_price",
        "no_chase_price",
        "invalidation_price",
        "valuation_anchor",
        "target_1",
        "target_2",
        "expectation_state",
        "operation_reason",
    )
    columns = [name for name in wanted if name in result.frame.columns]
    candidates: list[dict] = []
    for _, source_row in result.frame.iterrows():
        row = {name: _json_value(source_row[name]) for name in columns}
        row["symbol"] = str(row.get("symbol", "")).zfill(6)
        if "score_momentum" in result.frame.columns:
            categories = ("momentum", "theme", "capital", "liquidity")
            categories += tuple(
                category for category in result.weights if category not in categories
            )
            row["score_breakdown"] = {
                category: _json_value(source_row.get(f"score_{category}"))
                for category in categories
            }
            row["reasons"] = explain(source_row.to_dict(), result.weights)
        candidates.append(row)
    return ForwardRecord(
        as_of=result.as_of,
        spec_hash=spec_fingerprint(result.spec),
        spec=result.spec.to_dict(),
        universe=result.universe_size,
        pool=result.pool_size,
        kept=result.kept,
        candidates=candidates,
        rejections=result.rejection_lines(),
        notes=list(result.notes),
        weights={key: float(value) for key, value in result.weights.items()},
        run_at=run_at,
    )


def _json_value(value: object) -> object:
    """把 pandas / numpy 标量转成可长期保存的 JSON 原生值。"""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def append_record(directory: str | Path, record: ForwardRecord) -> tuple[Path, Path]:
    """落盘：单次记录 `<as-of>__<hash>.json` + 追加 `journal.jsonl`。

    同日同 spec 重跑**覆盖**单次记录（幂等），但 jsonl 仍追加一行 ——
    这样"同一天跑了两次"这件事本身也留痕。
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / f"{record.as_of}__{record.spec_hash}{RECORD_SUFFIX}"
    record_path.write_text(record.to_json(), encoding="utf-8")

    summary = {
        "as_of": record.as_of,
        "spec_hash": record.spec_hash,
        "universe": record.universe,
        "pool": record.pool,
        "kept": record.kept,
        "symbols": [row["symbol"] for row in record.candidates],
        "weights": record.weights,
        "run_at": record.run_at,
    }
    with (directory / JOURNAL_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return record_path, directory / JOURNAL_NAME


JOURNAL_COLUMNS = ["as_of", "spec_hash", "universe", "pool", "kept", "symbols", "run_at"]


def load_journal(directory: str | Path) -> pd.DataFrame:
    """读回 journal（一行一次运行）。

    文件不存在时返回**空表而不是抛错** —— 第一次跑之前调用它是正常情况，不是错误。
    """

    path = Path(directory) / JOURNAL_NAME
    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 对账
# --------------------------------------------------------------------------- #


def evaluate(store: MarketStore, directory: str | Path, horizons: tuple[int, ...] = HORIZONS):
    """拿**实际后续走势**给每条历史记录打分。

    返回 `(per_run, per_horizon)` 两张表：
    * `per_run` —— 每次运行一行的明细（含基准对照）。
    * `per_horizon` —— 按持有期聚合的汇总。

    ⚠️ 这里**只报描述性统计，不报显著性**。样本期一长，择时 beta、行业 beta、
    风格（大小盘 / 成长价值）都会混进"候选 vs 全市场"的差额里。
    前向纸盘的 n 太小（几天到几十天），任何 t 检验在这个样本下都没有判断力 —— 
    把它当成"发现"就是自欺。**要看的是方向一致性与差额的弥散程度，不是点估计大小。**
    """

    journal = load_journal(directory)
    if journal.empty:
        return journal, pd.DataFrame()

    dates = store.trade_dates()
    index_of = {day: position for position, day in enumerate(dates)}

    # 只把**对账真正会用到的日子**读进来。全量 pivot（450 万行 × 5300 列）会直接把内存吃光。
    wanted: set[str] = set()
    for record in journal.itertuples(index=False):
        position = index_of.get(record.as_of)
        if position is None:
            continue
        wanted.add(record.as_of)
        for horizon in horizons:
            if position + horizon < len(dates):
                wanted.add(dates[position + horizon])
    day_frames = _day_close_frames(store, sorted(wanted))

    def forward_return(symbol: str, as_of: str, horizon: int) -> float | None:
        position = index_of.get(as_of)
        if position is None or position + horizon >= len(dates):
            return None  # 持有期还没走完 —— 不是 0%，是**还不可测**。
        entry = day_frames.get(as_of)
        exit_frame = day_frames.get(dates[position + horizon])
        if entry is None or exit_frame is None:
            return None
        if symbol not in entry.index or symbol not in exit_frame.index:
            return None  # 停牌 / 退市 ⇒ 不可测，不能按 0 计入。
        entry_price = entry[symbol]
        exit_price = exit_frame[symbol]
        if not entry_price or pd.isna(entry_price) or pd.isna(exit_price):
            return None
        return float(exit_price) / float(entry_price) - 1.0

    rows: list[dict] = []
    for record in journal.itertuples(index=False):
        symbols = list(record.symbols)
        for horizon in horizons:
            picks = [forward_return(symbol, record.as_of, horizon) for symbol in symbols]
            picks = [value for value in picks if value is not None]
            # 基准：全市场等权（用当天有行情的标的作 universe）。
            universe = _universe_return(day_frames, dates, index_of, record.as_of, horizon)
            rows.append(
                {
                    "as_of": record.as_of,
                    "spec_hash": record.spec_hash,
                    "horizon": horizon,
                    "n": len(picks),
                    "n_requested": len(symbols),
                    "pick_ret": sum(picks) / len(picks) if picks else None,
                    "universe_ret": universe,
                    "excess": (sum(picks) / len(picks) - universe)
                    if picks and universe is not None
                    else None,
                }
            )
    per_run = pd.DataFrame(rows)
    if per_run.empty:
        return per_run, pd.DataFrame()
    grouped = (
        per_run.groupby("horizon")
        .agg(
            runs=("as_of", "nunique"),
            measured=("excess", "count"),
            pick_ret=("pick_ret", "mean"),
            universe_ret=("universe_ret", "mean"),
            excess_mean=("excess", "mean"),
            excess_std=("excess", "std"),
        )
        .reset_index()
    )
    return per_run, grouped


def persist_evaluation(
    directory: str | Path,
    per_run: pd.DataFrame,
    per_horizon: pd.DataFrame,
) -> tuple[Path, Path]:
    """把逐次对账与按持有期汇总写入项目目录。"""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    per_run_path = directory / EVALUATION_NAME
    summary_path = directory / EVALUATION_SUMMARY_NAME
    per_run.to_csv(per_run_path, index=False, encoding="utf-8-sig")
    per_horizon.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return per_run_path, summary_path


def evaluate_and_persist(
    store: MarketStore,
    directory: str | Path,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """重新计算当前所有前向记录，并覆盖项目内的验证快照。"""

    per_run, per_horizon = evaluate(store, directory, horizons=horizons)
    per_run_path, summary_path = persist_evaluation(directory, per_run, per_horizon)
    return per_run, per_horizon, per_run_path, summary_path


def _day_close_frames(store: MarketStore, days: list[str]) -> dict[str, pd.Series]:
    """`{日期: Series(close, index=symbol)}`，只取需要的那几天。"""

    if not days:
        return {}
    quoted = ", ".join(f"date '{day}'" for day in days)
    frame = store.conn.execute(
        f"select symbol, trade_date, close from '{store.path.as_posix()}' "
        f"where trade_date in ({quoted})"
    ).df()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    return {
        day: group.set_index("symbol")["close"]
        for day, group in frame.groupby("trade_date", sort=False)
    }


def _universe_return(day_frames, dates, index_of, as_of, horizon) -> float | None:
    """全市场等权后续收益（同一 as_of、同一 horizon）。"""

    position = index_of.get(as_of)
    if position is None or position + horizon >= len(dates):
        return None
    entry = day_frames.get(as_of)
    later = day_frames.get(dates[position + horizon])
    if entry is None or later is None:
        return None
    common = entry.index.intersection(later.index)
    if len(common) == 0:
        return None
    entry = entry.loc[common]
    later = later.loc[common]
    valid = entry.notna() & later.notna() & (entry != 0)
    if not valid.any():
        return None
    return float((later[valid] / entry[valid] - 1.0).mean())


__all__ = [
    "EVALUATION_NAME",
    "EVALUATION_SUMMARY_NAME",
    "HORIZONS",
    "JOURNAL_NAME",
    "ForwardRecord",
    "append_record",
    "build_record",
    "evaluate",
    "evaluate_and_persist",
    "load_journal",
    "persist_evaluation",
    "spec_fingerprint",
]
