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
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
    initial_sidebar_state="auto",
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
            border: 1px solid var(--line); border-radius: 10px; padding: 9px 12px;
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
        .top-summary-grid, .candidate-status-grid, .detail-stat-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 8px; margin: .55rem 0 .85rem;
        }
        .top-summary-item, .candidate-status-item, .detail-stat-item {
            min-width: 0; background: linear-gradient(135deg, rgba(14, 18, 35, .98), rgba(17, 24, 43, .92));
            border: 1px solid var(--line); border-radius: 10px; padding: 9px 12px;
        }
        .top-summary-label, .candidate-status-label, .detail-stat-label {
            color: var(--muted); display: block; font-size: .78rem; line-height: 1.25;
        }
        .top-summary-value, .candidate-status-value, .detail-stat-value {
            color: var(--text); display: block; font: 600 1.12rem 'Fira Code', monospace;
            margin-top: 4px; overflow-wrap: anywhere;
        }
        .top-summary-note { color: var(--muted); display: block; font-size: .72rem; margin-top: 3px; }
        .candidate-industry-line { color: var(--muted); font-size: .82rem; line-height: 1.8; }
        .candidate-industry-chip {
            display: inline-block; background: rgba(96, 165, 250, .10); border: 1px solid #263c5e;
            border-radius: 999px; color: #bfdbfe; margin: 0 .25rem .25rem 0; padding: 1px 9px;
        }
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


@st.cache_data(show_spinner=False)
def load_symbol_bars(path: str, symbol: str, end: str) -> pd.DataFrame:
    with MarketStore(path) as store:
        return store.symbol_bars(symbol, end=end, window=120)


def format_number(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def format_pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def format_score_100(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.1f} / 100"


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
                "类别得分/100": format_number(float(score) * 100, 1) if available else "缺失",
                "综合分贡献": format_number(float(score) * weight / total * 100, 1)
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


def render_candlestick_chart(bars: pd.DataFrame, symbol: str, name: str) -> None:
    if bars.empty:
        st.info("本地行情归档里没有这只股票的日线数据。")
        return

    data = bars.copy()
    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    if data.empty:
        st.info("这只股票的本地日线字段不完整，暂时无法绘图。")
        return

    width, height = 680, 286
    margin_left, margin_right, margin_top, margin_bottom = 62, 16, 18, 36
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    ma_periods = (5, 10, 20, 60)
    ma_colors = {5: "#f59e0b", 10: "#a78bfa", 20: "#60a5fa", 60: "#22d3ee"}
    moving_averages = {
        period: data["close"].rolling(period, min_periods=period).mean().tail(60).reset_index(drop=True)
        for period in ma_periods
    }
    visible = data.tail(60).reset_index(drop=True)
    ma_values = [
        float(value)
        for series in moving_averages.values()
        for value in series
        if not pd.isna(value)
    ]
    low = float(visible["low"].min())
    high = float(visible["high"].max())
    if ma_values:
        low = min(low, min(ma_values))
        high = max(high, max(ma_values))
    spread = max(high - low, max(abs(high), 1.0) * 0.02)
    low -= spread * 0.04
    high += spread * 0.04

    def y_pos(value: float) -> float:
        return margin_top + (high - value) / (high - low) * plot_height

    count = len(visible)
    step = plot_width / max(count, 1)
    candle_width = min(7.0, max(2.4, step * 0.62))
    svg: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(name)} '
        f'{escape(symbol)} 近60日K线及MA5、MA10、MA20、MA60" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" rx="12" fill="#0e1223"/>',
    ]
    for tick in range(4):
        value = high - (high - low) * tick / 3
        y = y_pos(value)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" '
            f'y2="{y:.1f}" stroke="#26344d" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{margin_left-8}" y="{y+4:.1f}" text-anchor="end" '
            'fill="#94a3b8" font-size="11">'
            f'{value:.2f}</text>'
        )

    ma_points: dict[int, list[str]] = {period: [] for period in ma_periods}
    for index, record in enumerate(visible.itertuples(index=False)):
        x = margin_left + (index + 0.5) * step
        top, bottom = y_pos(float(record.high)), y_pos(float(record.low))
        opening, closing = y_pos(float(record.open)), y_pos(float(record.close))
        color = "#ef4444" if record.close >= record.open else "#22c55e"
        body_y = min(opening, closing)
        body_h = max(abs(closing - opening), 1.2)
        svg.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
            f'stroke="{color}" stroke-width="1.2"/>'
        )
        svg.append(
            f'<rect x="{x-candle_width/2:.1f}" y="{body_y:.1f}" '
            f'width="{candle_width:.1f}" height="{body_h:.1f}" fill="{color}"/>'
        )
        for period in ma_periods:
            average = moving_averages[period].iloc[index]
            if not pd.isna(average):
                ma_points[period].append(f"{x:.1f},{y_pos(float(average)):.1f}")
    for period in ma_periods:
        points = ma_points[period]
        if len(points) >= 2:
            svg.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{ma_colors[period]}" '
                'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        elif points:
            x, y = points[0].split(",")
            svg.append(f'<circle cx="{x}" cy="{y}" r="2.3" fill="{ma_colors[period]}"/>')
    first_day = pd.Timestamp(visible.iloc[0]["trade_date"]).strftime("%Y-%m-%d")
    last_day = pd.Timestamp(visible.iloc[-1]["trade_date"]).strftime("%Y-%m-%d")
    for index, period in enumerate(ma_periods):
        legend_x = width - 258 + index * 59
        svg.append(
            f'<line x1="{legend_x}" y1="14" x2="{legend_x + 10}" y2="14" '
            f'stroke="{ma_colors[period]}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{legend_x + 13}" y="18" fill="#94a3b8" font-size="10">MA{period}</text>'
        )
    svg.extend(
        [
            f'<text x="{margin_left}" y="{height-10}" fill="#94a3b8" '
            f'font-size="11">{first_day}</text>',
            f'<text x="{width-margin_right}" y="{height-10}" text-anchor="end" '
            f'fill="#94a3b8" font-size="11">{last_day}</text>',
            "</svg>",
        ]
    )
    components.html("".join(svg), height=height, scrolling=False)


def render_stock_detail(row: pd.Series, result, config: AppConfig, rank: int) -> None:
    symbol = safe_text(row.get("symbol"), "未知代码")
    name = safe_text(row.get("name"), "未同步名称")
    industry = safe_text(row.get("industry"), "未同步")
    st.markdown(f"### {escape(name)} <span class='stock-symbol'>{escape(symbol)}</span>", unsafe_allow_html=True)
    tags = safe_text(row.get("theme_tags"), "未归因")
    st.caption(f"排名 {rank:02d} · 行业 {escape(industry)} · 题材归因 {escape(tags)}")

    signal = safe_text(row.get("operation_signal"), "未生成")
    reference = format_number(row.get("reference_price"))
    buy_range = format_price_range(row.get("buy_low"), row.get("buy_high"))
    invalidation = format_number(row.get("invalidation_price"))
    targets = f"{format_number(row.get('target_1'))} / {format_number(row.get('target_2'))}"
    detail_stats = [
        ("收盘价", format_number(row.get("close"))),
        ("综合得分", format_score_100(row.get("score"))),
        ("权重覆盖", format_pct(row.get("score_coverage"), 0)),
        ("20日收益", format_pct(row.get("ret_20"))),
        ("操作信号", signal),
        ("参考价", reference),
        ("分批参考区间", buy_range),
        ("逻辑失效价", invalidation),
        ("目标一 / 二", targets),
    ]
    stats_html = ["<div class='detail-stat-grid'>"]
    stats_html.extend(
        "<div class='detail-stat-item'><span class='detail-stat-label'>"
        f"{escape(label)}</span><strong class='detail-stat-value'>{escape(value)}</strong></div>"
        for label, value in detail_stats
    )
    stats_html.append("</div>")
    st.markdown("".join(stats_html), unsafe_allow_html=True)

    weekday = safe_text(row.get("operation_weekday"), "未知日期")
    expectation = safe_text(row.get("expectation_state"), "暂无明确代理")
    reason = safe_text(row.get("operation_reason"), "暂无操作说明")
    st.markdown(
        "<div class='operation-note'>"
        f"{escape(weekday)} · 短期兑现：{escape(expectation)}；回落参考 "
        f"{format_number(row.get('pullback_price'))}；不追价上限 "
        f"{format_number(row.get('no_chase_price'))}。{escape(reason)}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("推荐理由与评分拆解", expanded=False):
        st.markdown(recommendation_reason(row, result.weights))
        st.dataframe(
            contribution_table(row, result.weights),
            hide_index=True,
            width="stretch",
            height=min(280, 42 + 36 * len(result.weights)),
        )
        detail_columns = (
            "ret_since_924", "ret_20", "ret_60", "relative_ret_20", "relative_ret_60",
            "volume_trend_20", "theme_heat_max", "theme_days_20", "lhb_net_buy_rel",
            "main_net_inflow", "turnover_value_20", "close_position_250", "dist_to_52w_high",
            "realized_vol_20", "downside_vol_20", "max_drawdown_60", "volatility_value",
            "drawdown_value", "downside_volatility_value", "pe_ttm", "pe_industry_value",
            "eps_ttm", "roe_avg", "gross_profit_margin", "net_profit_margin", "yoy_net_income",
            "yoy_eps_basic", "revenue_yoy", "profit_growth_acceleration", "eps_growth_acceleration",
            "positive_profit_streak", "positive_growth_streak", "current_ratio", "quick_ratio",
            "cash_ratio", "liability_to_asset", "asset_to_equity", "ebit_to_interest", "cfo_to_or",
            "cfo_to_np", "cfo_to_gr", "financial_safety_value", "industry_score", "industry_leader_score",
        )
        detail_rows = []
        percent_columns = {
            "ret_since_924", "ret_20", "ret_60", "dist_to_52w_high", "ma20_dev", "ma60_dev",
            "lhb_net_buy_rel", "margin_rz_chg_20", "roe_avg", "gross_profit_margin", "net_profit_margin",
            "yoy_net_income", "yoy_eps_basic", "revenue_yoy", "profit_growth_acceleration",
            "eps_growth_acceleration", "relative_ret_20", "relative_ret_60", "liability_to_asset",
            "cfo_to_or", "cfo_to_np", "cfo_to_gr", "downside_vol_20",
        }
        for column in detail_columns:
            if column not in result.frame.columns:
                continue
            value = row.get(column)
            if column in percent_columns:
                formatted = format_pct(value)
            elif column in {"turnover_value_20", "main_net_inflow"}:
                formatted = format_money(value)
            else:
                formatted = format_number(value, 3)
            detail_rows.append({"因子": FACTOR_LABELS.get(column, column), "数值": formatted})
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch", height=280)

    with st.expander("查看本地 60 日 K 线", expanded=False):
        if st.button("加载行情图", key=f"load-chart-{symbol}"):
            st.session_state[f"chart-loaded-{symbol}"] = True
        if st.session_state.get(f"chart-loaded-{symbol}"):
            try:
                bars = load_symbol_bars(config.market.daily_parquet, symbol, result.as_of)
                render_candlestick_chart(bars, symbol, name)
            except Exception as error:  # noqa: BLE001 - 图表故障不影响推荐清单
                st.warning(f"本地图表暂不可用：{type(error).__name__}: {error}")


def render_stock_cards(result, config: AppConfig) -> None:
    if result.frame.empty:
        st.warning("当前条件没有命中股票。请先查看‘数据健康’和筛选漏斗，确认归档是否完整。")
        return

    frame = result.frame.reset_index(drop=True).copy()
    frame["_rank"] = range(1, len(frame) + 1)
    frame["industry"] = frame.get("industry", pd.Series(index=frame.index, dtype="object")).fillna("未同步")
    frame["operation_signal"] = frame.get(
        "operation_signal", pd.Series(index=frame.index, dtype="object")
    ).fillna("未生成")

    st.markdown("#### 候选分布")
    signal_order = ["今天可分批", "等回落再买", "暂缓，先确认"]
    signal_counts = frame["operation_signal"].value_counts()
    industry_counts = frame["industry"].replace("", "未同步").value_counts().head(4)
    overview_html = ["<div class='candidate-status-grid'>"]
    overview_html.extend(
        "<div class='candidate-status-item'><span class='candidate-status-label'>"
        f"{escape(signal)}</span><strong class='candidate-status-value'>"
        f"{int(signal_counts.get(signal, 0))} 只</strong></div>"
        for signal in signal_order
    )
    overview_html.append("</div><div class='candidate-industry-line'>候选行业 · ")
    overview_html.extend(
        f"<span class='candidate-industry-chip'>{escape(str(industry))} {int(count)} 只</span>"
        for industry, count in industry_counts.items()
    )
    overview_html.append("<span>仅统计当前推荐清单</span></div>")
    st.markdown("".join(overview_html), unsafe_allow_html=True)

    all_industries = sorted(frame["industry"].replace("", "未同步").unique().tolist())
    all_signals = sorted(frame["operation_signal"].unique().tolist())
    search = st.text_input(
        "搜索代码 / 名称", placeholder="输入代码或名称", key="candidate-search"
    ).strip().lower()
    with st.expander("更多筛选：行业与操作信号", expanded=False):
        industry_filter = st.selectbox("行业", ["全部"] + all_industries, key="candidate-industry")
        signal_filter = st.selectbox("操作信号", ["全部"] + all_signals, key="candidate-signal")
    industry_filter = st.session_state.get("candidate-industry", "全部")
    signal_filter = st.session_state.get("candidate-signal", "全部")

    visible = frame.copy()
    if industry_filter != "全部":
        visible = visible.loc[visible["industry"] == industry_filter]
    if signal_filter != "全部":
        visible = visible.loc[visible["operation_signal"] == signal_filter]
    if search:
        search_text = visible["symbol"].astype(str) + " " + visible["name"].astype(str)
        visible = visible.loc[search_text.str.lower().str.contains(search, regex=False, na=False)]
    visible = visible.reset_index(drop=True)
    if visible.empty:
        st.info("没有符合当前筛选条件的候选股。")
        return

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(f"**候选清单**　{len(visible)} 只 · 点击行查看详情")
        table = pd.DataFrame(
            {
                "候选": [
                    f"{int(row['_rank']):02d} · {safe_text(row.get('symbol'))} "
                    f"{safe_text(row.get('name'), '未同步名称')}"
                    for _, row in visible.iterrows()
                ],
                "信号": visible["operation_signal"].tolist(),
                "评分/100": (pd.to_numeric(visible["score"], errors="coerce") * 100).tolist(),
                "20日": (pd.to_numeric(visible["ret_20"], errors="coerce") * 100).tolist(),
            }
        )
        key = f"candidate-table-{industry_filter}-{signal_filter}-{search}"
        selection = st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=458,
            row_height=36,
            on_select="rerun",
            selection_mode="single-row",
            selection_default={"selection": {"rows": [0], "columns": [], "cells": []}},
            key=key,
            column_config={
                "候选": st.column_config.TextColumn("候选", width="medium"),
                "信号": st.column_config.TextColumn("信号", width="small"),
                "评分/100": st.column_config.NumberColumn("评分/100", format="%.1f", width="small"),
                "20日": st.column_config.NumberColumn("20日", format="%.1f%%", width="small"),
            },
        )

    selected = selection.selection.rows[0] if selection.selection.rows else 0
    selected = min(selected, len(visible) - 1)
    selected_row = visible.iloc[selected]
    with right:
        render_stock_detail(selected_row, result, config, int(selected_row["_rank"]))


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

    coverage = result.frame["score_coverage"].median() if "score_coverage" in result.frame else None
    top_summary = [
        ("数据截至", result.as_of, "最近完整交易日"),
        (
            "推荐数量",
            f"{result.kept:,}",
            f"全市场 {result.universe_size:,} → 入选池 {result.pool_size:,}",
        ),
        ("中位权重覆盖", format_pct(coverage, 0), "仅统计当前推荐清单"),
        ("推荐预置", preset, "可在左侧切换策略"),
    ]
    summary_html = ["<div class='top-summary-grid'>"]
    summary_html.extend(
        "<div class='top-summary-item'><span class='top-summary-label'>"
        f"{escape(label)}</span><strong class='top-summary-value'>{escape(value)}</strong>"
        f"<small class='top-summary-note'>{escape(note)}</small></div>"
        for label, value, note in top_summary
    )
    summary_html.append("</div>")
    st.markdown("".join(summary_html), unsafe_allow_html=True)

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
        render_stock_cards(result, config)
    with method_tab:
        render_method(result)
    with validation_tab:
        render_forward_validation(config)
    with health_tab:
        render_data_health(config)


if __name__ == "__main__":
    main()
