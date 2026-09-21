"""基本面、估值与行业因子。

PE(TTM) 是估值倍数，不是越大越好，所以单独构造 ``pe_ttm_value``：只保留正 PE，
并用 ``-log(PE)`` 让较低的正 PE 获得更高的排序方向。财务字段全部按公告日截断，
由上游归档层保证不把未来数据带进当前推荐。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNDAMENTAL_FACTORS = (
    "symbol",
    "pe_ttm",
    "pe_ttm_value",
    "pb_mrq",
    "ps_ttm",
    "eps_ttm",
    "roe_avg",
    "gross_profit_margin",
    "net_profit_margin",
    "yoy_net_income",
    "yoy_eps_basic",
    "yoy_profit_net_income",
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
    "market_cap",
    "pe_industry_value",
    "industry_score",
    "industry_leader_score",
)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _group_rank(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    """把每个行业的一项聚合值映射为全行业 0-1 分位。"""

    if "industry" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    valid = frame["industry"].notna() & frame["industry"].astype(str).ne("")
    grouped = values.where(valid).groupby(frame.loc[:, "industry"], dropna=True).mean()
    if grouped.dropna().empty:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    ranked = grouped.rank(pct=True)
    return frame["industry"].map(ranked)


def _within_group_rank(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    """行业内横截面分位，用于识别行业龙头代理。"""

    if "industry" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    valid = frame["industry"].notna() & frame["industry"].astype(str).ne("")
    ranked = values.where(valid).groupby(frame["industry"], dropna=True).rank(pct=True)
    return ranked


def _available_mean(parts: list[pd.Series], index: pd.Index) -> pd.Series:
    if not parts:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _history_metrics(history: pd.DataFrame | None) -> pd.DataFrame:
    """从最近若干期财报计算增长趋势，不把缺失期猜成零。"""

    columns = (
        "symbol",
        "revenue_yoy",
        "profit_growth_acceleration",
        "eps_growth_acceleration",
        "positive_profit_streak",
        "positive_growth_streak",
    )
    if history is None or history.empty:
        return pd.DataFrame(columns=list(columns))

    frame = history.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    frame["stat_date"] = pd.to_datetime(frame["stat_date"], errors="coerce")
    for column in ("revenue", "net_profit", "yoy_net_income", "yoy_eps_basic"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for symbol, group in frame.dropna(subset=["symbol", "stat_date"]).groupby("symbol"):
        group = group.sort_values("stat_date", kind="stable").drop_duplicates("stat_date")
        latest = group.iloc[-1]
        previous = group.iloc[-2] if len(group) >= 2 else None
        same_period = group.loc[
            (group["stat_date"].dt.month == latest["stat_date"].month)
            & (group["stat_date"].dt.day == latest["stat_date"].day)
            & (group["stat_date"].dt.year == latest["stat_date"].year - 1)
        ]
        prior_revenue = same_period.iloc[-1].get("revenue") if not same_period.empty else np.nan
        revenue = latest.get("revenue")
        revenue_yoy = (
            float(revenue) / float(prior_revenue) - 1.0
            if pd.notna(revenue) and pd.notna(prior_revenue) and float(prior_revenue) != 0
            else np.nan
        )
        profit_growth_acceleration = (
            float(latest.get("yoy_net_income")) - float(previous.get("yoy_net_income"))
            if previous is not None
            and pd.notna(latest.get("yoy_net_income"))
            and pd.notna(previous.get("yoy_net_income"))
            else np.nan
        )
        eps_growth_acceleration = (
            float(latest.get("yoy_eps_basic")) - float(previous.get("yoy_eps_basic"))
            if previous is not None
            and pd.notna(latest.get("yoy_eps_basic"))
            and pd.notna(previous.get("yoy_eps_basic"))
            else np.nan
        )

        positive_profit_streak = 0
        for value in group.iloc[::-1]["net_profit"]:
            if pd.isna(value):
                break
            if float(value) <= 0:
                break
            positive_profit_streak += 1
        positive_growth_streak = 0
        for value in group.iloc[::-1]["yoy_net_income"]:
            if pd.isna(value):
                break
            if float(value) <= 0:
                break
            positive_growth_streak += 1
        rows.append(
            {
                "symbol": symbol,
                "revenue_yoy": revenue_yoy,
                "profit_growth_acceleration": profit_growth_acceleration,
                "eps_growth_acceleration": eps_growth_acceleration,
                "positive_profit_streak": positive_profit_streak or np.nan,
                "positive_growth_streak": positive_growth_streak or np.nan,
            }
        )
    return pd.DataFrame(rows, columns=list(columns))


def add_fundamental_factors(
    frame: pd.DataFrame,
    valuation: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
    financial_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """把本地估值、财务快照和行业质量拼入因子宽表。"""

    out = frame.copy()
    if valuation is not None and not valuation.empty:
        valuation_columns = [
            column
            for column in ("symbol", "close", "pe_ttm", "pb_mrq", "ps_ttm")
            if column in valuation.columns
        ]
        values = valuation.loc[:, valuation_columns].drop_duplicates("symbol")
        # close 已来自行情；估值源的 close 仅作补充，不覆盖行情口径。
        values = values.rename(columns={"close": "valuation_close"})
        out = out.merge(values, on="symbol", how="left")
    if financial is not None and not financial.empty:
        financial_columns = [
            column
            for column in (
                "symbol",
                "roe_avg",
                "gross_profit_margin",
                "net_profit_margin",
                "net_profit",
                "eps_ttm",
                "revenue",
                "total_share",
                "yoy_net_income",
                "yoy_eps_basic",
                "yoy_profit_net_income",
                "current_ratio",
                "quick_ratio",
                "cash_ratio",
                "liability_to_asset",
                "asset_to_equity",
                "ebit_to_interest",
                "cfo_to_or",
                "cfo_to_np",
                "cfo_to_gr",
            )
            if column in financial.columns
        ]
        values = financial.loc[:, financial_columns].drop_duplicates("symbol")
        out = out.merge(values, on="symbol", how="left")

    history = _history_metrics(financial_history)
    if not history.empty:
        out = out.merge(history, on="symbol", how="left")

    out = _numeric(
        out,
        [
            "pe_ttm",
            "pb_mrq",
            "ps_ttm",
            "eps_ttm",
            "roe_avg",
            "gross_profit_margin",
            "net_profit_margin",
            "net_profit",
            "revenue",
            "total_share",
            "yoy_net_income",
            "yoy_eps_basic",
            "yoy_profit_net_income",
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
            "ret_20",
            "ret_60",
        ],
    )

    positive_pe = out["pe_ttm"].where(out["pe_ttm"] > 0) if "pe_ttm" in out else None
    if positive_pe is None:
        out["pe_ttm_value"] = np.nan
    else:
        # 极端 PE 对横截面排序的影响封顶；亏损 / 非正 PE 不猜，保持缺失。
        out["pe_ttm_value"] = -np.log(positive_pe.clip(lower=0.5, upper=200.0))

    if "industry" in out and positive_pe is not None:
        peer_median = positive_pe.groupby(out["industry"], dropna=True).transform("median")
        out["pe_industry_value"] = -np.log(
            positive_pe.clip(lower=0.5, upper=200.0)
            / peer_median.clip(lower=0.5, upper=200.0)
        )
    else:
        out["pe_industry_value"] = np.nan

    if {"close", "total_share"}.issubset(out.columns):
        out["market_cap"] = out["close"] * out["total_share"]
    else:
        out["market_cap"] = np.nan
    out["financial_safety_value"] = -pd.to_numeric(
        out.get("liability_to_asset", pd.Series(np.nan, index=out.index)), errors="coerce"
    )

    # 行业龙头代理：行业内总市值分位；财报缺失时不制造假排名。
    leader = out["market_cap"].where(out["market_cap"] > 0)
    out["industry_leader_score"] = _within_group_rank(out, leader)

    # 行业景气代理：行业市场强度 + 行业盈利增长。两项缺一时按实际可用项重归一。
    ret_20 = out.get("ret_20", pd.Series(np.nan, index=out.index))
    breadth = ret_20.where(ret_20.isna(), ret_20.gt(0).astype(float))
    industry_market = _available_mean(
        [
            _group_rank(out, out.get("ret_60", pd.Series(np.nan, index=out.index))),
            _group_rank(out, breadth),
        ],
        out.index,
    )
    industry_growth = _available_mean(
        [
            _group_rank(
                out,
                out.get("yoy_net_income", pd.Series(np.nan, index=out.index)),
            ),
            _group_rank(
                out,
                out.get("yoy_eps_basic", pd.Series(np.nan, index=out.index)),
            ),
            _group_rank(
                out,
                out.get("revenue_yoy", pd.Series(np.nan, index=out.index)),
            ),
        ],
        out.index,
    )
    out["industry_score"] = _available_mean([industry_market, industry_growth], out.index)

    for column in FUNDAMENTAL_FACTORS:
        if column not in out.columns:
            out[column] = pd.NA
    ordered = list(FUNDAMENTAL_FACTORS) + [
        column for column in out.columns if column not in FUNDAMENTAL_FACTORS
    ]
    return out.loc[:, ordered]


__all__ = ["FUNDAMENTAL_FACTORS", "add_fundamental_factors"]
