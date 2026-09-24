"""本地股票推荐工作台。

启动：
    .venv\\Scripts\\streamlit.exe run app.py

页面只从当前项目的 `data/` 读取推荐输入；启动自动同步或「同步全部数据」会先把网络
结果写入本地归档，再重新计算推荐。这样整个项目文件夹复制到另一台电脑后仍能复现
同一套口径，网络只用于更新本地归档。
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommend.archive import ArchiveStore  # noqa: E402
from recommend.config import AppConfig  # noqa: E402
from recommend.context import ScreenContext  # noqa: E402
from recommend.forward import EVALUATION_SUMMARY_NAME, load_journal  # noqa: E402
from recommend.market import MarketStore  # noqa: E402
from recommend.rank.score import FACTOR_LABELS, RANK_MEMBERS  # noqa: E402
from recommend.screen import Spec, run_screen  # noqa: E402
from recommend.screen.presets import PRESETS, load_preset  # noqa: E402
from recommend.sync import SyncRunResult, coverage_snapshot  # noqa: E402
from recommend.web_sync import (  # noqa: E402
    render_status as render_background_sync_status,
)
from recommend.web_sync import (  # noqa: E402
    run_sync_with_feedback,
)
from recommend.web_sync import (  # noqa: E402
    snapshot as background_sync_snapshot,
)
from recommend.web_sync import (  # noqa: E402
    start as start_background_sync,
)

CATEGORY_LABELS = {
    "industry": "行业",
    "growth": "增长",
    "quality": "财务质量",
    "growth_quality": "增长质量（兼容）",
    "valuation": "估值",
    "momentum": "动量",
    "risk": "风险",
    "theme": "题材",
    "capital": "资金",
    "liquidity": "流动性",
}

st.set_page_config(
    page_title="推荐股票工作台",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@400;500;600;700&display=swap');
        :root {
            --bg: #020617;
            --panel: #0e1223;
            --panel-2: #11182b;
            --line: #26344d;
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        html, body, [class*="css"] { font-family: 'Fira Sans', sans-serif; }
        [data-testid="stAppViewContainer"] { background: var(--bg); }
        [data-testid="stHeader"] { background: rgba(2, 6, 23, .82); }
        .block-container { max-width: 1480px; padding-top: 2.4rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: #080c18; border-right: 1px solid var(--line); }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(14, 18, 35, .98), rgba(17, 24, 43, .92));
            border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--text); font-family: 'Fira Code', monospace; }
        .hero-kicker {
            color: var(--accent); font: 600 12px 'Fira Code', monospace;
            letter-spacing: .12em; text-transform: uppercase;
        }
        .hero-title {
            color: var(--text); font-size: 2.2rem; font-weight: 700;
            letter-spacing: -.03em; margin: .25rem 0 .35rem;
        }
        .hero-subtitle { color: var(--muted); font-size: 1rem; margin-bottom: 1.2rem; }
        .section-label {
            color: var(--muted); font: 600 12px 'Fira Code', monospace;
            letter-spacing: .08em; text-transform: uppercase; margin: 1rem 0 .45rem;
        }
        .data-note {
            color: var(--muted); font-size: .88rem;
            border-left: 2px solid var(--accent); padding: .2rem .8rem;
        }
        .stock-heading { color: var(--text); font-size: 1.05rem; font-weight: 600; }
        .stock-symbol { color: var(--accent); font-family: 'Fira Code', monospace; }
        .mono { font-family: 'Fira Code', monospace; }
        .operation-note {
            color: var(--muted); font-size: .88rem; line-height: 1.55;
            border-left: 2px solid var(--warning); padding: .25rem .8rem;
            background: rgba(245, 158, 11, .06); border-radius: 0 8px 8px 0;
        }
        button { transition: border-color .18s ease, background .18s ease, transform .18s ease; }
        button:hover { transform: translateY(-1px); }
        button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
        @media (prefers-reduced-motion: reduce) {
            /* 不要全局覆盖 Streamlit 的 st.spinner；它需要自己的旋转反馈。 */
            button { transition: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_config() -> AppConfig:
    return AppConfig.load(PROJECT_ROOT / "config" / "default.toml")


@st.cache_resource(show_spinner=False)
def auto_sync_state() -> dict[str, object]:
    """每个 Streamlit 服务进程只保留一份后台同步状态。

    这个状态必须放在 ``cache_resource`` 中，而不是 ``session_state``：同一个
    Streamlit 服务可能同时有多个浏览器会话，启动同步只能跑一份。同步本身在
    后台线程执行，主线程才能先把网页渲染出来，避免网络源慢时页面一直显示
    Streamlit 的全页 spinner。
    """

    return {
        "lock": threading.RLock(),
        "startup_started": False,
        "running": False,
        "kind": "",
        "preset": None,
        "result": None,
        "progress": None,
        "error": None,
        "last_result_id": None,
    }


def current_as_of(config: AppConfig) -> str | None:
    try:
        with MarketStore(config.market.daily_parquet) as store:
            return store.latest_complete_trade_date() if store.ready else None
    except (FileNotFoundError, ValueError):
        return None


def build_local_screen(config: AppConfig, preset_name: str):
    payload = load_preset(preset_name)
    with ScreenContext.open(config, payload.get("as_of")) as context:
        table = context.build()
        spec = Spec.from_dict({**payload, "as_of": table.as_of})
        return run_screen(table, spec, config.rank.as_dict())


def format_number(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def format_pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def format_money(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) / 1e8:.2f} 亿"


def format_price_range(low: object, high: object) -> str:
    low_text = format_number(low)
    high_text = format_number(high)
    if low_text == "—" and high_text == "—":
        return "—"
    return f"{low_text} – {high_text}"


def safe_text(value: object, fallback: str = "—") -> str:
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return fallback
    return str(value).replace("|", "、")


def contribution_table(row: pd.Series, weights: dict[str, float]) -> pd.DataFrame:
    total = sum(weights.values())
    rows: list[dict[str, object]] = []
    for category, weight in sorted(weights.items(), key=lambda item: -item[1]):
        score = row.get(f"score_{category}")
        available = score is not None and not pd.isna(score)
        rows.append(
            {
                "类别": CATEGORY_LABELS.get(category, category),
                "类别权重": f"{weight / total:.0%}",
                "类别分位": format_number(score, 3) if available else "缺失",
                "对综合分贡献": format_number(float(score) * weight / total, 3)
                if available
                else "未参与",
                "状态": "已参与" if available else "缺失，已重归一",
            }
        )
    return pd.DataFrame(rows)


def recommendation_reason(row: pd.Series, weights: dict[str, float]) -> str:
    active: list[str] = []
    if not pd.isna(row.get("pe_ttm")):
        active.append(f"PE(TTM) {format_number(row.get('pe_ttm'))}")
    if not pd.isna(row.get("pe_industry_value")):
        active.append(f"行业内PE相对分 {format_number(row.get('pe_industry_value'))}")
    if not pd.isna(row.get("roe_avg")):
        active.append(f"ROE {format_pct(row.get('roe_avg'))}")
    if not pd.isna(row.get("gross_profit_margin")):
        active.append(f"毛利率 {format_pct(row.get('gross_profit_margin'))}")
    if not pd.isna(row.get("net_profit_margin")):
        active.append(f"净利率 {format_pct(row.get('net_profit_margin'))}")
    if not pd.isna(row.get("yoy_net_income")):
        active.append(f"净利润同比 {format_pct(row.get('yoy_net_income'))}")
    if not pd.isna(row.get("revenue_yoy")):
        active.append(f"营收同比 {format_pct(row.get('revenue_yoy'))}")
    if not pd.isna(row.get("profit_growth_acceleration")):
        active.append(f"利润增速加速度 {format_pct(row.get('profit_growth_acceleration'))}")
    if not pd.isna(row.get("positive_growth_streak")):
        active.append(f"连续利润增长 {format_number(row.get('positive_growth_streak'), 0)} 期")
    if not pd.isna(row.get("cfo_to_np")):
        active.append(f"经营现金流/净利润 {format_pct(row.get('cfo_to_np'))}")
    if not pd.isna(row.get("liability_to_asset")):
        active.append(f"资产负债率 {format_pct(row.get('liability_to_asset'))}")
    if not pd.isna(row.get("industry_score")):
        active.append(f"行业景气分 {format_number(row.get('industry_score'))}")
    if not pd.isna(row.get("industry_leader_score")):
        active.append(f"行业龙头分 {format_number(row.get('industry_leader_score'))}")
    if not pd.isna(row.get("ret_20")):
        active.append(f"20日收益 {format_pct(row.get('ret_20'))}")
    if not pd.isna(row.get("relative_ret_20")):
        active.append(f"20日相对大盘 {format_pct(row.get('relative_ret_20'))}")
    if not pd.isna(row.get("relative_ret_60")):
        active.append(f"60日相对大盘 {format_pct(row.get('relative_ret_60'))}")
    if not pd.isna(row.get("ret_since_924")):
        active.append(f"924以来收益 {format_pct(row.get('ret_since_924'))}")
    if not pd.isna(row.get("volume_trend_20")):
        active.append(f"量比 {format_number(row.get('volume_trend_20'))}")
    if not pd.isna(row.get("theme_heat_max")):
        active.append(f"题材热度 {format_number(row.get('theme_heat_max'), 0)}")
    if not pd.isna(row.get("theme_days_20")):
        active.append(f"近20日题材上榜 {format_number(row.get('theme_days_20'), 0)} 天")
    if not pd.isna(row.get("lhb_net_buy_rel")):
        active.append(f"龙虎榜净买强度 {format_pct(row.get('lhb_net_buy_rel'))}")
    if not pd.isna(row.get("main_net_inflow")):
        active.append(f"当日主力净额 {format_money(row.get('main_net_inflow'))}")
    if not pd.isna(row.get("turnover_value_20")):
        active.append(f"20日均成交额 {format_money(row.get('turnover_value_20'))}")
    if not pd.isna(row.get("volatility_value")):
        active.append(f"低波动分 {format_number(row.get('volatility_value'))}")
    if not pd.isna(row.get("downside_volatility_value")):
        active.append(f"下行波动分 {format_number(row.get('downside_volatility_value'))}")

    position = []
    if not pd.isna(row.get("close_position_250")):
        position.append(f"近一年位置 {format_pct(row.get('close_position_250'))}")
    if not pd.isna(row.get("dist_to_52w_high")):
        position.append(f"距52周高点 {format_pct(row.get('dist_to_52w_high'))}")

    lines = []
    if active:
        lines.append("方向性证据：" + "；".join(active) + "。")
    if position:
        lines.append("筛选位置：" + "；".join(position) + "（位置只用于缩小范围，不参与排序）。")
    if not active:
        lines.append("当前可用的方向性数据不足，未把缺失值当作负分。")
    lines.append(f"权重覆盖 {format_pct(row.get('score_coverage'), 0)}；综合分由可用类别重新归一。")
    return "\n\n".join(lines)


def render_stock_cards(result) -> None:
    if result.frame.empty:
        st.warning("当前条件没有命中股票。请先查看‘数据健康’和筛选漏斗，确认归档是否完整。")
        return

    for rank, (_, row) in enumerate(result.frame.iterrows(), start=1):
        symbol = safe_text(row.get("symbol"), "未知代码")
        name = safe_text(row.get("name"), "未同步名称")
        score = format_number(row.get("score"), 3)
        title = f"{rank:02d}   {symbol}  {name}   ·   综合分 {score}"
        with st.expander(title, expanded=rank <= 3):
            top = st.columns([1, 1, 1, 1.35, 2.1])
            top[0].metric("收盘价", format_number(row.get("close")))
            top[1].metric("权重覆盖", format_pct(row.get("score_coverage"), 0))
            top[2].metric("20日收益", format_pct(row.get("ret_20")))
            industry = safe_text(row.get("industry"), "未同步")
            top[3].markdown(
                f"**行业（本地快照）**\n\n<span class='stock-symbol'>{industry}</span>",
                unsafe_allow_html=True,
            )
            tags = safe_text(row.get("theme_tags"), "未归因")
            top[4].markdown(
                f"**板块 / 题材（本地归因）**\n\n<span class='stock-symbol'>{tags}</span>",
                unsafe_allow_html=True,
            )

            st.markdown("**推荐理由**")
            st.markdown(recommendation_reason(row, result.weights))
            st.markdown("**今日操作建议**")
            plan_cols = st.columns([1.15, 1.15, 1.45, 1.05, 1.2])
            plan_cols[0].metric("动作", safe_text(row.get("operation_signal")))
            plan_cols[1].metric("参考价", format_number(row.get("reference_price")))
            plan_cols[2].metric(
                "分批买入区间",
                format_price_range(row.get("buy_low"), row.get("buy_high")),
            )
            plan_cols[3].metric("逻辑失效", format_number(row.get("invalidation_price")))
            plan_cols[4].metric(
                "目标一",
                format_number(row.get("target_1")),
                delta=f"目标二 {format_number(row.get('target_2'))}",
                delta_color="off",
            )
            weekday = safe_text(row.get("operation_weekday"), "未知日期")
            expectation = safe_text(row.get("expectation_state"), "暂无明确代理")
            no_chase = format_number(row.get("no_chase_price"))
            pullback = format_number(row.get("pullback_price"))
            reason = safe_text(row.get("operation_reason"), "暂无操作说明")
            st.markdown(
                f"<div class='operation-note'>{weekday} · 短期兑现：{expectation}；"
                f"回落参考 {pullback}；不追价上限 {no_chase}。{reason}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**评分拆解**")
            st.dataframe(
                contribution_table(row, result.weights),
                hide_index=True,
                width="stretch",
            )
            with st.expander("查看因子明细", expanded=False):
                detail = pd.DataFrame(
                    [
                        {
                            "因子": FACTOR_LABELS.get(column, column),
                            "数值": (
                                format_pct(row.get(column))
                                if column
                                in {
                                    "ret_since_924",
                                    "ret_20",
                                    "ret_60",
                                    "dist_to_52w_high",
                                    "ma20_dev",
                                    "ma60_dev",
                                    "lhb_net_buy_rel",
                                    "margin_rz_chg_20",
                                    "roe_avg",
                                    "gross_profit_margin",
                                    "net_profit_margin",
                                    "yoy_net_income",
                                    "yoy_eps_basic",
                                    "revenue_yoy",
                                    "profit_growth_acceleration",
                                    "eps_growth_acceleration",
                                    "relative_ret_20",
                                    "relative_ret_60",
                                    "liability_to_asset",
                                    "cfo_to_or",
                                    "cfo_to_np",
                                    "cfo_to_gr",
                                    "downside_vol_20",
                                }
                                else format_money(row.get(column))
                                if column in {"turnover_value_20", "main_net_inflow"}
                                else format_number(row.get(column), 3)
                            ),
                        }
                        for column in (
                            "ret_since_924",
                            "ret_20",
                            "ret_60",
                            "relative_ret_20",
                            "relative_ret_60",
                            "volume_trend_20",
                            "theme_heat_max",
                            "theme_days_20",
                            "lhb_net_buy_rel",
                            "main_net_inflow",
                            "turnover_value_20",
                            "close_position_250",
                            "dist_to_52w_high",
                                    "realized_vol_20",
                                    "downside_vol_20",
                                    "max_drawdown_60",
                                    "volatility_value",
                                    "drawdown_value",
                                    "downside_volatility_value",
                            "pe_ttm",
                            "pe_industry_value",
                            "eps_ttm",
                            "roe_avg",
                            "gross_profit_margin",
                            "net_profit_margin",
                            "yoy_net_income",
                            "yoy_eps_basic",
                            "revenue_yoy",
                            "profit_growth_acceleration",
                            "eps_growth_acceleration",
                            "positive_profit_streak",
                            "positive_growth_streak",
                            "current_ratio",
                            "quick_ratio",
                            "cash_ratio",
                            "liability_to_asset",
                            "asset_to_equity",
                            "ebit_to_interest",
                            "cfo_to_or",
                            "cfo_to_np",
                            "cfo_to_gr",
                            "financial_safety_value",
                            "industry_score",
                            "industry_leader_score",
                        )
                        if column in result.frame.columns
                    ]
                )
                st.dataframe(detail, hide_index=True, width="stretch")


def render_sidebar(config: AppConfig) -> str:
    st.sidebar.markdown("<div class='hero-kicker'>LOCAL / AUDITABLE</div>", unsafe_allow_html=True)
    st.sidebar.markdown("### 推荐工作台")
    st.sidebar.caption("推荐计算只读取项目目录内的 data/；启动或点击同步时才访问网络。")
    preset = st.sidebar.selectbox(
        "推荐预置",
        list(PRESETS),
        format_func=lambda value: f"{value} · {PRESETS[value].get('note', '')}",
        key="preset",
    )
    st.sidebar.divider()
    as_of = current_as_of(config)
    st.sidebar.metric("推荐可用数据截至", as_of or "未就绪")
    if st.sidebar.button("刷新本地展示", width="stretch"):
        st.rerun()
    return preset


def render_data_health(config: AppConfig) -> None:
    st.markdown("<div class='section-label'>Data health</div>", unsafe_allow_html=True)
    paths = [
        ("行情日线", config.market.daily_parquet),
        ("名称/上市日", config.market.instrument_parquet),
        ("个股行业快照", config.market.industry_csv),
        ("行情引导种子", config.market.source_parquet),
        ("归档数据库（含估值/财务）", config.archive_db),
        ("推荐报告目录", config.reports_dir),
    ]
    path_rows = []
    for label, raw_path in paths:
        path = Path(raw_path)
        resolved = path.resolve()
        try:
            inside = resolved.is_relative_to(PROJECT_ROOT.resolve())
        except AttributeError:  # pragma: no cover - Python 3.11+ 不会走这里
            inside = str(resolved).lower().startswith(str(PROJECT_ROOT.resolve()).lower())
        path_rows.append(
            {
                "数据项": label,
                "项目内": "是" if inside else "否",
                "存在": "是" if path.exists() else "否",
                "相对路径": str(path),
            }
        )
    st.dataframe(pd.DataFrame(path_rows), hide_index=True, width="stretch")
    if not all(row["项目内"] == "是" for row in path_rows):
        st.error("发现项目外路径：复制到另一台电脑后可能失效，请先修复配置。")
    else:
        st.success("推荐输入与引导种子均位于当前项目目录，可随文件夹一起搬运。")

    try:
        coverage = coverage_snapshot(config)
        st.markdown("**本地归档覆盖**")
        st.dataframe(coverage, hide_index=True, width="stretch")
    except Exception as error:  # noqa: BLE001
        st.warning(f"读取归档覆盖失败：{error}")

    try:
        with ArchiveStore(config.archive_db) as store:
            runs = store.run_log(limit=12)
        st.markdown("**最近同步记录**")
        st.dataframe(runs, hide_index=True, width="stretch")
    except Exception as error:  # noqa: BLE001
        st.warning(f"读取同步记录失败：{error}")


def render_method(result) -> None:
    st.markdown("<div class='section-label'>Scoring policy</div>", unsafe_allow_html=True)
    st.markdown("**当前评分权重**")
    weights = result.weights
    total = sum(weights.values())
    rows = []
    for category, weight in sorted(weights.items(), key=lambda item: -item[1]):
        members = "、".join(FACTOR_LABELS.get(item, item) for item in RANK_MEMBERS[category])
        rows.append(
            {
                "类别": CATEGORY_LABELS.get(category, category),
                "权重": f"{weight / total:.0%}",
                "参与因子": members,
                "说明": "缺失时该类别不参与，并按实际可用权重重归一",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "行业名称用于展示；行业景气 / 行业龙头分已经作为独立权重参与综合分。"
        " PE(TTM) 同时参考绝对 PE 与行业内 PE；亏损或缺失时按实际可用权重重归一。"
    )
    st.markdown("**筛选条件**")
    for line in result.spec.describe():
        st.markdown(f"- {line}")
    st.markdown("**筛选漏斗**")
    st.write(
        f"全市场 {result.universe_size:,} 只 → 通过条件 {result.pool_size:,} 只 "
        f"→ 展示 {result.kept:,} 只"
    )
    if result.truncated:
        st.info(f"还有 {result.truncated:,} 只通过条件但未展示，受当前 limit 限制。")
    if result.rejections:
        rejection = pd.DataFrame(
            [{"剔除原因": key, "数量": value} for key, value in result.rejections.items() if value]
        )
        st.dataframe(rejection, hide_index=True, width="stretch")
    if result.notes:
        for note in result.notes:
            st.warning(note)


def render_forward_validation(config: AppConfig) -> None:
    """展示自动留证与前向验证结果。"""

    st.markdown("<div class='section-label'>Forward validation</div>", unsafe_allow_html=True)
    directory = Path(config.reports_dir) / "forward"
    journal = load_journal(directory)
    summary_path = directory / EVALUATION_SUMMARY_NAME

    if journal.empty:
        st.info("还没有推荐留证。完成一次自动同步或点击“再次同步全部数据”后会自动记录。")
        return

    latest_as_of = str(journal["as_of"].max()) if "as_of" in journal else ""
    unique_batches = (
        journal[["as_of", "spec_hash"]].drop_duplicates().shape[0]
        if {"as_of", "spec_hash"}.issubset(journal.columns)
        else len(journal)
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("推荐运行记录", f"{len(journal):,}")
    metric_cols[1].metric("独立推荐批次", f"{unique_batches:,}")
    metric_cols[2].metric("最近推荐数据日", latest_as_of)

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        measured = int(summary["measured"].sum()) if "measured" in summary else 0
        metric_cols[3].metric("已完成验证项", f"{measured:,}")
        display = summary.rename(
            columns={
                "horizon": "持有交易日",
                "runs": "推荐批次",
                "measured": "已完成验证",
                "pick_ret": "推荐平均收益",
                "universe_ret": "全市场平均收益",
                "excess_mean": "平均超额",
                "excess_std": "超额波动",
            }
        )
        for column in ("推荐平均收益", "全市场平均收益", "平均超额", "超额波动"):
            if column in display:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):+.2%}"
                )
        st.markdown("**按持有期汇总**")
        st.dataframe(display, hide_index=True, width="stretch")
    else:
        metric_cols[3].metric("已完成验证项", "0")
        st.info("验证文件尚未生成；下一次同步会自动创建。")

    recent = journal.tail(20).copy()
    recent["推荐数"] = recent["symbols"].map(
        lambda value: len(value) if isinstance(value, list) else 0
    )
    recent = recent.rename(
        columns={
            "as_of": "数据日",
            "spec_hash": "规则指纹",
            "kept": "通过数",
            "run_at": "记录时间",
        }
    )
    columns = ["数据日", "规则指纹", "通过数", "推荐数", "记录时间"]
    st.markdown("**最近推荐留证**")
    recent_columns = [column for column in columns if column in recent]
    st.dataframe(recent.loc[:, recent_columns], hide_index=True, width="stretch")
    st.caption(
        "超额 = 推荐组合等权收益 − 同日全市场等权收益；持有期未完成时不计入统计。"
        "样本较少时只能作为观察，不能据此宣称策略已经可靠。"
    )


def auto_sync_on_start(config: AppConfig, preset: str) -> None:
    """服务进程第一次建立页面会话时同步一次，页面交互不会重复抓取。"""

    if os.environ.get("RECOMMEND_DISABLE_AUTO_SYNC") == "1":
        return None

    state = auto_sync_state()
    lock = state["lock"]
    if lock is None:
        return None
    with lock:
        if state.get("started", False):
            cached = state["result"]
            if state["preset"] == preset and isinstance(cached, SyncRunResult):
                st.caption(f"本次服务启动已完成自动同步：{cached.finished_at or '已完成'}")
                return cached
            return None
        state["started"] = True
        state["preset"] = preset
        run = run_sync_with_feedback(config, preset, startup=True)
        state["result"] = run
        return run


def sync_panel(config: AppConfig, preset: str) -> SyncRunResult | None:  # background-safe
    st.markdown("<div class='section-label'>Manual sync</div>", unsafe_allow_html=True)
    sync_col, hint_col = st.columns([1, 3])
    with sync_col:
        clicked = st.button("再次同步全部数据", type="primary", width="stretch", key="sync_button")
    with hint_col:
        st.markdown(
            "<div class='data-note'>服务启动或手动同步会先检查同步时段。工作日 09:30–15:30 "
            "暂停同步并只读最近完整交易日；每日 16:00 收盘后任务运行。只有当天收盘数据已确认时，"
            "才生成推荐和前向留证。</div>",
            unsafe_allow_html=True,
        )
    if not clicked:  # no blocking sync
        return None
    started = start_background_sync(config, preset)
    if started:
        st.info("已启动后台同步；页面不会被网络请求阻塞，同步完成后会自动刷新。")
    else:
        st.warning("已有同步任务正在执行，请等待当前任务结束。")
    return None


def main() -> None:
    inject_style()
    config = load_config()
    preset = render_sidebar(config)

    st.markdown(
        "<div class='hero-kicker'>STOCK RECOMMENDATION / LOCAL WORKBENCH</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hero-title'>推荐股票工作台</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>把全市场筛选结果变成可复核的候选清单："
        "每只股票都有数据时点、评分权重、板块/题材和可解释理由。</div>",
        unsafe_allow_html=True,
    )

    auto_sync_on_start(config, preset)
    manually_synced = sync_panel(config, preset)
    render_background_sync_status(st)
    background_snapshot = background_sync_snapshot()
    background_result = background_snapshot.get("result")
    synced = manually_synced or (
        background_result
        if background_snapshot.get("preset") == preset
        and isinstance(background_result, SyncRunResult)
        else None
    )
    if synced is not None and synced.screen is not None:
        result = synced.screen
    elif background_snapshot.get("running"):
        st.info(
            "启动同步正在后台执行；首屏已打开，推荐清单会在同步完成后自动刷新。"
        )
        return
    else:
        try:
            result = build_local_screen(config, preset)
        except Exception as error:  # noqa: BLE001
            st.error(f"当前无法生成推荐：{type(error).__name__}: {error}")
            st.info(
                "请确认 data/market/daily.parquet 和 "
                "data/reference/instrument_master.parquet 已随项目复制。"
            )
            render_data_health(config)
            return

    summary_cols = st.columns(4)
    summary_cols[0].metric("数据截至", result.as_of)
    summary_cols[1].metric("推荐数量", f"{result.kept:,}")
    summary_cols[1].caption(
        f"全市场 {result.universe_size:,} → 入选池 {result.pool_size:,} → 展示 {result.kept:,}"
    )
    coverage = result.frame["score_coverage"].median() if "score_coverage" in result.frame else None
    summary_cols[2].metric("中位权重覆盖", format_pct(coverage, 0))
    summary_cols[3].metric("推荐预置", preset)

    st.markdown(
        f"<div class='data-note'>数据时点：行情与因子截至 {result.as_of}；"
        "题材、龙虎榜、两融使用该日及之前最近 20 个交易日。"
        "行业来自当前目录的本地快照，仅用于展示；推荐输入均来自当前目录的本地归档。</div>",
        unsafe_allow_html=True,
    )

    recommendations_tab, method_tab, validation_tab, health_tab = st.tabs(
        ["推荐清单", "权重与方法", "验证记录", "数据健康"]
    )
    with recommendations_tab:
        st.markdown("<div class='section-label'>Recommended universe</div>", unsafe_allow_html=True)
        render_stock_cards(result)
    with method_tab:
        render_method(result)
    with validation_tab:
        render_forward_validation(config)
    with health_tab:
        render_data_health(config)


if __name__ == "__main__":
    main()
