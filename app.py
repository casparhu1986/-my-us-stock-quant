from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from quant_model import (
    DEFAULT_WATCHLIST,
    MARKET_SYMBOLS,
    download_prices,
    market_risk_score,
    score_universe,
    suggested_position_pct,
)

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="美股量化驾驶舱",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Mobile-first visual tuning.
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 5rem;
        max-width: 1100px;
    }
    h1 { font-size: 1.65rem !important; margin-bottom: .2rem !important; }
    h2 { font-size: 1.25rem !important; margin-top: .8rem !important; }
    h3 { font-size: 1.05rem !important; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: .65rem .75rem;
        background: rgba(128,128,128,.05);
    }
    div[data-testid="stMetricLabel"] { font-size: .82rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    .mobile-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: .8rem .9rem;
        margin: .45rem 0;
        background: rgba(128,128,128,.04);
    }
    .tiny { font-size: .82rem; opacity: .78; }
    .signal-strong { font-weight: 700; }
    button[kind="primary"] { min-height: 44px; }
    @media (max-width: 640px) {
        .block-container { padding-left: .8rem; padding-right: .8rem; }
        h1 { font-size: 1.45rem !important; }
        div[data-testid="stHorizontalBlock"] { gap: .5rem; }
        div[data-testid="stDataFrame"] { font-size: .82rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return download_prices(tickers, period=period)


def load_theme_map() -> Dict[str, str]:
    df = pd.read_csv(BASE_DIR / "watchlist.csv")
    return dict(zip(df["ticker"].astype(str).str.upper(), df["theme"].astype(str)))


def load_default_portfolio() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "portfolio.csv")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def signal_label(score: float) -> str:
    if not np.isfinite(score):
        return "—"
    if score >= 80:
        return "强趋势"
    if score >= 70:
        return "候选"
    if score >= 60:
        return "观察"
    return "回避新增"


def market_bucket(score: float) -> str:
    if score >= 80:
        return "积极"
    if score >= 60:
        return "偏积极"
    if score >= 40:
        return "中性"
    if score >= 20:
        return "谨慎"
    return "防守"


def model_exposure_midpoint(text: str) -> float:
    # e.g. "60%–80%" -> 70
    nums = []
    for part in text.replace("%", "").replace("–", "-").split("-"):
        try:
            nums.append(float(part))
        except ValueError:
            pass
    return float(np.mean(nums)) if nums else np.nan


def score_change(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    left = current[["ticker", "total"]].rename(columns={"total": "score_now"})
    right = previous[["ticker", "total"]].rename(columns={"total": "score_prev"})
    out = left.merge(right, on="ticker", how="left")
    out["score_delta"] = out["score_now"] - out["score_prev"]
    return out


def portfolio_analysis(
    portfolio: pd.DataFrame,
    scores: pd.DataFrame,
    theme_map: Dict[str, str],
    cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    p = portfolio.copy()
    p["ticker"] = p["ticker"].astype(str).str.upper().str.strip()
    p["shares"] = pd.to_numeric(p["shares"], errors="coerce").fillna(0.0)
    p["avg_cost"] = pd.to_numeric(p["avg_cost"], errors="coerce").fillna(0.0)

    if "theme" not in p:
        p["theme"] = p["ticker"].map(theme_map).fillna("其他")
    else:
        p["theme"] = p["theme"].fillna("")
        missing = p["theme"].astype(str).str.strip().eq("")
        p.loc[missing, "theme"] = p.loc[missing, "ticker"].map(theme_map).fillna("其他")

    cols = ["ticker", "price", "total", "atr_pct", "建议单股仓位%"]
    s = scores[[c for c in cols if c in scores.columns]].copy()
    p = p.merge(s, on="ticker", how="left")
    p["market_value"] = p["shares"] * p["price"].fillna(0.0)
    p["cost_value"] = p["shares"] * p["avg_cost"]
    p["pnl"] = p["market_value"] - p["cost_value"]
    p["pnl_pct"] = np.where(p["cost_value"] > 0, p["pnl"] / p["cost_value"] * 100, np.nan)

    invested = float(p["market_value"].sum())
    total_account = invested + float(cash)
    p["weight_pct"] = np.where(total_account > 0, p["market_value"] / total_account * 100, 0.0)

    active = p[p["market_value"] > 0].copy()
    if active.empty:
        theme = pd.DataFrame(columns=["theme", "value", "weight_pct"])
    else:
        theme = active.groupby("theme", as_index=False)["market_value"].sum()
        theme = theme.rename(columns={"market_value": "value"})
        theme["weight_pct"] = np.where(total_account > 0, theme["value"] / total_account * 100, 0.0)
        theme = theme.sort_values("weight_pct", ascending=False)

    return p, theme, invested, total_account


theme_map = load_theme_map()
default_watchlist = list(theme_map.keys())
default_portfolio = load_default_portfolio()

if "portfolio" not in st.session_state:
    st.session_state.portfolio = default_portfolio.copy()
if "cash" not in st.session_state:
    st.session_state.cash = 0.0

st.title("📈 美股量化驾驶舱")
st.caption("手机优先版 V1.1 · 每日风险、持仓与信号一屏查看")

with st.expander("⚙️ 设置", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        benchmark = st.selectbox("相对强弱基准", ["QQQ", "SPY"], index=0)
    with c2:
        period = st.selectbox("历史数据", ["2y", "5y"], index=0)

    include_fundamentals = st.checkbox(
        "启用基本面因子（更新更慢）",
        value=False,
        help="关闭时基本面项按中性分处理，更适合手机快速刷新。",
    )
    raw_watchlist = st.text_area(
        "股票池（逗号分隔）",
        value=",".join(default_watchlist),
        height=100,
    )
    st.session_state.cash = st.number_input(
        "现金余额（美元）",
        min_value=0.0,
        value=float(st.session_state.cash),
        step=1000.0,
        format="%.2f",
    )

tickers = list(dict.fromkeys(
    [x.strip().upper() for x in raw_watchlist.split(",") if x.strip()]
    + st.session_state.portfolio["ticker"].astype(str).str.upper().tolist()
))
all_tickers = tuple(dict.fromkeys(MARKET_SYMBOLS + tickers + [benchmark]))

refresh_col, status_col = st.columns([1, 2])
with refresh_col:
    refresh = st.button("🔄 更新", type="primary", use_container_width=True)
with status_col:
    st.caption("行情缓存 15 分钟；再次点击“更新”可重新计算。")

# Always run on open; button clears the cache first.
if refresh:
    st.cache_data.clear()

try:
    with st.spinner("正在读取行情并计算量化信号…"):
        data = load_market_data(all_tickers, period)
        market = market_risk_score(data, breadth_tickers=tickers)
        scores = score_universe(
            data,
            tickers=tickers,
            benchmark=benchmark,
            include_fundamentals=include_fundamentals,
        )
        scores["建议单股仓位%"] = scores.apply(
            lambda r: suggested_position_pct(
                r.get("total", float("nan")),
                r.get("atr_pct", float("nan")),
                market.score,
            ),
            axis=1,
        )
        scores["signal"] = scores["total"].apply(signal_label)

        if len(data.index) > 3:
            prev_data = data.iloc[:-1].copy()
            prev_scores = score_universe(
                prev_data,
                tickers=tickers,
                benchmark=benchmark,
                include_fundamentals=False,
            )
            changes = score_change(scores, prev_scores)
            scores = scores.merge(changes[["ticker", "score_delta"]], on="ticker", how="left")
        else:
            scores["score_delta"] = np.nan

        ptable, theme_table, invested, account_total = portfolio_analysis(
            st.session_state.portfolio,
            scores,
            theme_map,
            st.session_state.cash,
        )

except Exception as exc:
    st.error("行情获取失败。请稍后点击“更新”，并确认部署环境可以访问市场数据。")
    st.exception(exc)
    st.stop()

# ---- HOME ----
tab_home, tab_holdings, tab_rank, tab_risk, tab_settings = st.tabs(
    ["首页", "持仓", "排名", "风险", "维护"]
)

with tab_home:
    exposure_mid = model_exposure_midpoint(market.suggested_equity_exposure)
    current_exposure = (invested / account_total * 100) if account_total > 0 else 0.0

    a, b = st.columns(2)
    a.metric("市场风险分", f"{market.score:.0f}/100", market_bucket(market.score))
    b.metric("模型总仓位", market.suggested_equity_exposure)

    c, d = st.columns(2)
    c.metric("当前股票仓位", f"{current_exposure:.1f}%")
    d.metric("账户规模", f"${account_total:,.0f}" if account_total > 0 else "待录入")

    if account_total > 0 and np.isfinite(exposure_mid):
        gap = current_exposure - exposure_mid
        if gap > 15:
            st.warning(f"当前股票仓位比模型区间中值高约 {gap:.0f} 个百分点，优先检查集中度与高波动持仓。")
        elif gap < -15:
            st.info(f"当前股票仓位比模型区间中值低约 {abs(gap):.0f} 个百分点。是否提高仓位仍需结合你的风险承受能力。")
        else:
            st.success("当前股票仓位大致位于模型风险暴露范围附近。")

    st.subheader("今日变化")
    move = scores.dropna(subset=["score_delta"]).copy()
    move["abs_delta"] = move["score_delta"].abs()
    move = move.sort_values("abs_delta", ascending=False).head(5)

    if move.empty:
        st.caption("暂无可比较的前一交易日评分。")
    else:
        for _, r in move.iterrows():
            arrow = "↑" if r["score_delta"] > 0 else ("↓" if r["score_delta"] < 0 else "→")
            st.markdown(
                f"""<div class="mobile-card">
                <b>{r['ticker']}</b> &nbsp; {r['total']:.0f}分 &nbsp; {arrow} {r['score_delta']:+.1f}
                <div class="tiny">{r['signal']} · 1个月 {r.get('r1m_pct', np.nan):+.1f}% · ATR {r.get('atr_pct', np.nan):.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.subheader("持仓提示")
    active = ptable[ptable["market_value"] > 0].copy()
    if active.empty:
        st.info("还没有录入实际持仓。到“维护”页填入股票数量和成本，就会自动计算持仓风险。")
    else:
        active = active.sort_values("weight_pct", ascending=False)
        for _, r in active.head(6).iterrows():
            score_text = f"{r['total']:.0f}" if np.isfinite(r.get("total", np.nan)) else "—"
            pnl_text = f"{r['pnl_pct']:+.1f}%" if np.isfinite(r.get("pnl_pct", np.nan)) else "—"
            st.markdown(
                f"""<div class="mobile-card">
                <b>{r['ticker']}</b> &nbsp; 评分 {score_text} &nbsp; 仓位 {r['weight_pct']:.1f}%
                <div class="tiny">盈亏 {pnl_text} · 模型单股仓位 {r.get('建议单股仓位%', 0):.1f}% · {r['theme']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    if not theme_table.empty:
        top_theme = theme_table.iloc[0]
        if top_theme["weight_pct"] >= 35:
            st.warning(f"主题集中度：{top_theme['theme']} 占账户约 {top_theme['weight_pct']:.1f}%，已进入重点检查区。")
        elif top_theme["weight_pct"] >= 25:
            st.info(f"主题集中度：{top_theme['theme']} 占账户约 {top_theme['weight_pct']:.1f}%，建议持续关注。")

    if market.notes:
        st.subheader("市场警报")
        for note in market.notes:
            st.warning(note)


with tab_holdings:
    st.subheader("我的持仓")
    active = ptable[ptable["market_value"] > 0].copy()
    if active.empty:
        st.info("暂无实际持仓。请在“维护”页录入。")
    else:
        cols = [
            "ticker", "shares", "price", "avg_cost", "pnl_pct",
            "weight_pct", "total", "建议单股仓位%", "theme"
        ]
        view = active[[c for c in cols if c in active.columns]].copy()
        view = view.rename(columns={
            "ticker": "代码",
            "shares": "股数",
            "price": "现价",
            "avg_cost": "成本",
            "pnl_pct": "盈亏%",
            "weight_pct": "账户仓位%",
            "total": "模型分",
            "建议单股仓位%": "模型单股仓位%",
            "theme": "主题",
        })
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.subheader("主题集中度")
        theme_view = theme_table[["theme", "weight_pct"]].rename(
            columns={"theme": "主题", "weight_pct": "账户占比%"}
        )
        st.bar_chart(theme_view.set_index("主题"))


with tab_rank:
    st.subheader("个股量化排名")
    compact = scores[
        ["ticker", "total", "score_delta", "price", "signal", "建议单股仓位%", "r1m_pct", "r3m_pct", "atr_pct"]
    ].copy()
    compact = compact.rename(columns={
        "ticker": "代码",
        "total": "总分",
        "score_delta": "日变化",
        "price": "现价",
        "signal": "状态",
        "建议单股仓位%": "模型仓位%",
        "r1m_pct": "1个月%",
        "r3m_pct": "3个月%",
        "atr_pct": "ATR%",
    })
    st.dataframe(compact, use_container_width=True, hide_index=True)

    with st.expander("查看详细因子"):
        detail = scores[
            ["ticker", "trend", "momentum", "relative_strength", "volume", "fundamentals", "risk"]
        ].rename(columns={
            "ticker": "代码",
            "trend": "趋势/25",
            "momentum": "动量/25",
            "relative_strength": "相对强弱/15",
            "volume": "量价/10",
            "fundamentals": "基本面/15",
            "risk": "风险/10",
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)


with tab_risk:
    st.subheader("市场风险结构")
    component_df = pd.DataFrame(
        {"因子": list(market.components.keys()), "得分": list(market.components.values())}
    )
    st.bar_chart(component_df.set_index("因子"))

    st.markdown(
        f"""
        **当前状态：{market.regime}**  
        模型股票风险暴露：**{market.suggested_equity_exposure}**
        """
    )

    st.caption(
        "风险分越高，代表环境对承担股票风险越友好；它不是涨跌预测，也不是买卖指令。"
    )


with tab_settings:
    st.subheader("维护我的持仓")
    edited = st.data_editor(
        st.session_state.portfolio,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("代码", required=True),
            "shares": st.column_config.NumberColumn("股数", min_value=0.0, step=1.0),
            "avg_cost": st.column_config.NumberColumn("平均成本", min_value=0.0, step=0.01),
            "theme": st.column_config.TextColumn("主题"),
        },
        key="portfolio_editor",
    )

    if st.button("保存本次持仓", use_container_width=True):
        edited["ticker"] = edited["ticker"].astype(str).str.upper().str.strip()
        st.session_state.portfolio = edited.copy()
        st.success("已保存到本次会话。重新计算后首页会按新持仓显示。")
        st.rerun()

    csv_data = edited.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载持仓 CSV 备份",
        data=csv_data,
        file_name="portfolio.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Streamlit Community Cloud 的本地文件不是持久数据库。长期使用时，建议把 portfolio.csv 放在私有仓库，"
        "或在 V1.2 接入 Google Sheets / 数据库保存持仓。"
    )

    st.divider()
    st.subheader("模型约束")
    st.markdown(
        """
- 60 分以下：原则上不新增仓位
- 单股模型仓位上限：10%
- ATR 越高：仓位自动下调
- 市场风险分下降：所有个股仓位统一降档
- 主题集中度达到约 25% 开始提示，35% 进入重点检查
- 当前版本不自动下单
        """
    )

st.divider()
st.caption("研究工具，不构成投资建议。行情源用于原型研究，实盘前应使用更稳定的数据源并先进行 Paper Trading。")
