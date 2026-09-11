from __future__ import annotations
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from quant_model import (MARKET_SYMBOLS, download_prices, completed_daily_data, data_quality,
                         market_risk_score, score_universe, target_weights)

ROOT = Path(__file__).parent
st.set_page_config(page_title="美股驾驶舱", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown('''<style>

:root{--ink:#172b45;--muted:#758399;--line:#e2e8f0;--blue:#2764df;--bg:#f3f6fa}
.stApp{background:var(--bg);color:var(--ink)}
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
header[data-testid="stHeader"]{background:rgba(243,246,250,.96)}
.block-container{max-width:1180px;padding:3.8rem 2rem 3rem}
h1,h2,h3{color:var(--ink);letter-spacing:-.03em}
h3{font-size:1.15rem!important;margin-top:.6rem}
[data-testid="stTabs"] [role="tablist"]{gap:1.7rem;border-bottom:1px solid var(--line);margin:8px 0 20px}
[data-testid="stTabs"] [role="tab"]{font-size:.95rem;padding:10px 0 13px}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--blue)!important;font-weight:700}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:var(--blue)}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line);border-radius:14px}
[data-testid="stDataFrame"],[data-testid="stDataEditor"]{border-radius:12px;overflow:hidden}
.stButton>button[kind="primary"]{background:var(--blue);border:0;border-radius:10px;min-height:42px;font-weight:600}
.stButton>button,.stDownloadButton>button{border-radius:10px}
.brand{display:flex;justify-content:space-between;align-items:center;margin:0 0 21px;gap:12px}
.eyebrow{font-size:10px;letter-spacing:2.5px;color:#688097;font-weight:800;margin-bottom:7px}
.brand h1{font-size:29px!important;font-weight:800;margin:0;padding:0;line-height:1.25}
.tag{font-size:11px;color:#426da5;background:#e7effd;border:1px solid #d4e3fb;padding:6px 10px;border-radius:99px;white-space:nowrap}
.data-line{font-size:12px;color:var(--muted);margin:2px 0 17px;line-height:1.7}
.hero{background:linear-gradient(115deg,#122940,#203f63);border-radius:20px;padding:24px 26px;color:#fff;display:grid;grid-template-columns:1.3fr 1fr;gap:25px;align-items:center;margin:0 0 15px}
.hero .label{font-size:12px;color:#b7c8db;letter-spacing:.05em}.score{font-size:60px;line-height:1.12;font-weight:700;letter-spacing:-3px;margin-top:6px}.score small{font-size:18px;color:#a0b8cf;letter-spacing:0;font-weight:400}.regime{font-size:13px;color:#aee0cb;margin:7px 0 0}.hero-note{font-size:12px;line-height:1.7;color:#b7c8db;margin-top:10px}.hero-side{border-left:1px solid #49617b;padding-left:24px}.hero-value{font-size:27px;font-weight:600;margin-top:9px;letter-spacing:-1px}.bar{height:5px;border-radius:5px;background:#3b5570;margin-top:16px;overflow:hidden}.bar span{height:100%;display:block;background:#75c5b1;border-radius:5px}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:21px}.tile{border:1px solid var(--line);border-radius:14px;background:#fff;padding:17px 18px}.tile-label{font-size:12px;color:var(--muted);margin-bottom:8px}.tile-value{font-size:25px;font-weight:700;letter-spacing:-.7px}.tile-note{font-size:11px;color:#8390a2;margin-top:5px}
.stock-row{display:grid;grid-template-columns:1fr auto;gap:12px;padding:14px 16px;border:1px solid var(--line);border-radius:12px;background:white;margin-bottom:8px}.symbol{font-size:15px;font-weight:700}.stock-sub{font-size:11px;color:var(--muted);margin-top:4px}.rank-score{font-size:20px;font-weight:700;text-align:right}.rank-sub{font-size:11px;text-align:right;color:var(--muted);margin-top:2px}
.foot{font-size:11px;color:#8190a3;line-height:1.8;border-top:1px solid var(--line);padding-top:17px;margin-top:24px}.section-note{font-size:12px;color:var(--muted);line-height:1.7;margin-bottom:12px}
@media(max-width:640px){.block-container{padding:3.5rem 1rem 2rem}.brand h1{font-size:24px!important}.brand{margin-bottom:16px}.eyebrow{font-size:9px}.tag{font-size:10px;padding:5px 8px}.hero{padding:19px 18px;gap:12px;border-radius:16px}.score{font-size:47px}.hero-side{padding-left:15px}.hero-value{font-size:23px}.hero-note{font-size:10px}.tiles{gap:7px}.tile{padding:13px 10px;border-radius:11px}.tile-label{font-size:10px}.tile-value{font-size:18px}.tile-note{font-size:9px}[data-testid="stTabs"] [role="tablist"]{gap:1.4rem;margin-bottom:13px}[data-testid="stTabs"] [role="tab"]{font-size:14px;white-space:nowrap}.data-line{font-size:11px}}

/* Compact, touch-friendly V1.3 layout. */
.block-container{padding-top:3.2rem}
.brand{margin-bottom:12px}.brand h1{font-size:27px!important}
.eyebrow{letter-spacing:1.6px}.data-line{margin:0 0 10px;color:#5b6d83}
.hero{padding:20px 23px;border-radius:16px;margin-bottom:12px}.score{font-size:50px}
.hero-note{font-size:12px}.hero-value{font-size:25px}
.tiles{margin-bottom:15px}.tile{padding:14px 16px}.tile-value{font-variant-numeric:tabular-nums}
.tile-label,.tile-note,.stock-sub,.rank-sub,.section-note{color:#5b6d83}
.quote-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:#dce4ed;border:1px solid #dce4ed;border-radius:12px;overflow:hidden;margin:0 0 12px}
.quote{background:#fff;padding:12px 14px;min-width:0}.quote-name{font-size:11px;color:#5b6d83}.quote-price{font-size:19px;font-weight:700;margin:4px 0;font-variant-numeric:tabular-nums}.quote-change{font-size:11px;color:#5b6d83}
.brief{border-left:3px solid #2764df;padding:9px 12px;background:#eaf0fb;font-size:12px;line-height:1.8;color:#29466c;margin:0 0 14px}
.stock-row{padding:12px 15px;margin-bottom:7px;align-items:center}.stock-sub{line-height:1.6}.rank-score{font-size:22px}
.mini-track{height:4px;background:#e4ebf4;border-radius:4px;margin-top:7px;overflow:hidden}.mini-track span{display:block;height:100%;background:#2764df}
[data-testid="stTabs"] [role="tablist"]{gap:0;background:#e8eef6;padding:4px;border-radius:12px;border:0;margin:2px 0 15px}
[data-testid="stTabs"] [role="tab"]{flex:1;justify-content:center;border-radius:8px;padding:10px 8px;min-height:44px}
[data-testid="stTabs"] [aria-selected="true"]{background:#fff;box-shadow:0 1px 4px #17345112}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],[data-testid="stTabs"] [data-baseweb="tab-border"]{display:none}
@media(max-width:640px){.block-container{padding:3.1rem .8rem 2rem}.brand h1{font-size:23px!important}.hero{padding:16px;gap:12px}.score{font-size:42px}.hero-value{font-size:21px}.hero-side{padding-left:12px}.hero-note{font-size:11px}.tiles{gap:6px}.tile{padding:12px 9px}.tile-label,.tile-note{font-size:11px}.tile-value{font-size:17px;overflow-wrap:anywhere}.quote-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.quote{padding:10px 12px}.quote-price{font-size:18px}.stock-sub,.rank-sub{font-size:11px}[data-testid="stTabs"] [role="tab"]{font-size:13px;padding:10px 4px}.eyebrow{font-size:10px}.tag{font-size:11px}}

</style>''', unsafe_allow_html=True)


def fmt(value, suffix="", digits=1):
    return f"{value:,.{digits}f}{suffix}" if np.isfinite(value) else "待确认"


def labels(score):
    return "数据不足" if not np.isfinite(score) else ("强趋势" if score >= 80 else "候选" if score >= 70 else "观察" if score >= 60 else "弱势")


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbols, period):
    return download_prices(symbols, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def historical_test(symbols, benchmark, cost):
    from backtest import run_backtest
    data, _ = completed_daily_data(download_prices(tuple(dict.fromkeys(list(symbols) + MARKET_SYMBOLS)), period="5y"))
    return run_backtest(data, list(symbols), benchmark, cost)


def portfolio_view(portfolio, data, cash, asof):
    p = portfolio.copy()
    p["ticker"] = p["ticker"].astype(str).str.upper().str.strip()
    for col in ["shares", "avg_cost"]:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p[p.shares.gt(0)].copy()
    latest = data["Close"].reindex([asof]).iloc[-1]
    p["price"] = p.ticker.map(latest)
    p["market_value"] = p.shares * p.price
    p["cost_value"] = p.shares * p.avg_cost
    p["pnl"] = p.market_value - p.cost_value
    p["pnl_pct"] = (p.pnl / p.cost_value.where(p.cost_value > 0)) * 100
    invested = float(p.market_value.sum()) if p.price.notna().all() else np.nan
    total = invested + cash
    p["weight_pct"] = p.market_value / total * 100 if total > 0 else np.nan
    return p, invested, total


def render_portfolio_editor():
    st.subheader("编辑持仓")
    st.caption("填写股数与美元成本。持仓保存在本次会话，关闭页面前请下载备份。")
    uploaded = st.file_uploader("恢复持仓 CSV 备份", type=["csv"], key="portfolio_import")
    if uploaded is not None and st.button("导入备份"):
        try:
            restored = pd.read_csv(uploaded)
            if not {"ticker", "shares", "avg_cost"}.issubset(restored.columns):
                raise ValueError("备份需要 ticker、shares、avg_cost 三列")
            if "theme" not in restored:
                restored["theme"] = "其他"
            for col in ["shares", "avg_cost"]:
                restored[col] = pd.to_numeric(restored[col], errors="raise")
                if not np.isfinite(restored[col]).all() or restored[col].lt(0).any():
                    raise ValueError("股数和成本需为非负有限数字")
            restored["ticker"] = restored.ticker.fillna("").astype(str).str.upper().str.strip()
            if restored.ticker.eq("").any() or restored.ticker.duplicated().any():
                raise ValueError("股票代码不能为空或重复")
            st.session_state.portfolio = restored[["ticker", "shares", "avg_cost", "theme"]]
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    edited = st.data_editor(st.session_state.portfolio, num_rows="dynamic", hide_index=True,
        width="stretch", key="holdings_editor", column_config={
            "ticker": st.column_config.TextColumn("代码", required=True),
            "shares": st.column_config.NumberColumn("股数", min_value=0.0),
            "avg_cost": st.column_config.NumberColumn("成本 $", min_value=0.0, format="%.2f"),
            "theme": st.column_config.TextColumn("行业 / 主题")})
    left, right = st.columns(2)
    if left.button("保存持仓", type="primary", width="stretch"):
        cleaned = edited.copy()
        cleaned["ticker"] = cleaned.ticker.fillna("").astype(str).str.upper().str.strip()
        cleaned = cleaned[cleaned.ticker.ne("")]
        for col in ["shares", "avg_cost"]:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0)
        if cleaned.ticker.duplicated().any() or (cleaned[["shares", "avg_cost"]] < 0).any().any() or not np.isfinite(cleaned[["shares", "avg_cost"]]).all().all():
            st.error("请检查重复代码、负数或无效数字。")
        else:
            st.session_state.portfolio = cleaned
            st.rerun()
    right.download_button("下载持仓备份", st.session_state.portfolio.to_csv(index=False).encode("utf-8-sig"),
                          "portfolio.csv", "text/csv", width="stretch")


watchlist = pd.read_csv(ROOT / "watchlist.csv")
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.read_csv(ROOT / "portfolio.csv")
if "cash" not in st.session_state:
    st.session_state.cash = 0.0
st.markdown('<div class="brand"><div><div class="eyebrow">US EQUITIES · DAILY</div><h1>美股驾驶舱</h1></div><span class="tag">V1.3 · 研究版</span></div>', unsafe_allow_html=True)
refresh_col, hint_col = st.columns([1, 2])
if refresh_col.button("刷新行情", type="primary", key="refresh_quotes", width="stretch"):
    st.cache_data.clear()
hint_col.caption("完整交易日日线 · 非实时行情")
with st.expander("设置 · 现金、基准与股票池", expanded=False):
    a, b = st.columns(2)
    benchmark = a.selectbox("比较基准", ["QQQ", "SPY"])
    cash = b.number_input("现金余额 · 美元", min_value=0.0, value=float(st.session_state.cash), step=100.0)
    st.session_state.cash = cash
    raw = st.text_area("股票代码 · 英文逗号分隔", ",".join(watchlist.ticker), height=90)
    include_fundamentals = st.checkbox("加入当前基本面快照", value=False,
        help="默认使用技术模式。开启后使用当前基本面快照，未提供可靠财报发布日期；暂停日分数变化，回测仍使用技术模式。")
    st.caption("技术模式：趋势25 + 动量25 + 相对强弱15 + 量价10 + 风险10，85分归一到100。新旧版本分数不能直接比较。")
tickers = list(dict.fromkeys([s.strip().upper() for s in raw.split(",") if s.strip()] + st.session_state.portfolio.ticker.astype(str).str.strip().str.upper().tolist()))
if not tickers:
    st.info("请在设置中填写至少一个股票代码。")
    st.stop()
market = None
try:
    with st.spinner("正在整理市场数据…"):
        data, expected = completed_daily_data(load_prices(tuple(dict.fromkeys(MARKET_SYMBOLS + tickers)), "2y"))
        audit = data_quality(data, tickers, expected)
        scores = score_universe(data, tickers, benchmark, include_fundamentals, expected)
        try:
            market = market_risk_score(data, tickers, expected)
        except ValueError as exc:
            market_error = str(exc)
        scores["状态"] = scores.total.map(labels)
        scores["score_delta"] = np.nan
        if len(data) > 253 and not include_fundamentals:
            previous = score_universe(data.iloc[:-1], tickers, benchmark, False)
            scores["score_delta"] = scores.total - scores.ticker.map(previous.set_index("ticker").total)
        holdings, invested, account_total = portfolio_view(st.session_state.portfolio, data, cash, expected)
except Exception as exc:
    st.error(f"暂时无法完成行情计算：{exc}")
    st.caption("可点击上方“刷新行情”重试，持仓仍可编辑和备份。")
    render_portfolio_editor()
    st.stop()

st.markdown(f'<div class="data-line">日线数据 · 实际最新 {data.index[-1].date()} · 应到 {expected.date()}（纽约） · 排除未收盘数据 · 缓存15分钟</div>', unsafe_allow_html=True)
if market is None:
    st.warning(market_error)
issues = audit[audit["检查"] != "通过"]
if not issues.empty:
    st.info(f"{len(issues)} 个标的数据不足，已暂停相关评分。详情见“市场”。")
if holdings.price.isna().any():
    st.warning("部分持仓缺少最新价格，账户估值暂不完整。")
# Every quote is tied to the same completed session; stale values are never substituted.
quote_cards = []
for symbol, name in [("SPY", "标普500 · SPY"), ("QQQ", "纳指100 · QQQ"), ("^VIX", "波动率 · VIX"), ("^TNX", "10年美债收益率")]:
    series = data["Close"].reindex(columns=[symbol])[symbol]
    current = series.reindex([expected]).iloc[0]
    prior_rows = series.loc[series.index < expected]
    prior = prior_rows.iloc[-1] if len(prior_rows) else np.nan
    suffix = "%" if symbol == "^TNX" else ""
    value = fmt(current, suffix, 2)
    if np.isfinite(current) and np.isfinite(prior) and prior > 0:
        delta = current - prior if symbol == "^TNX" else (current / prior - 1) * 100
        unit = "百分点" if symbol == "^TNX" else "%"
        change = f"{delta:+.2f}{unit} · 较前日"
    else:
        change = "日变化待确认"
    quote_cards.append(f'<div class="quote"><div class="quote-name">{name}</div><div class="quote-price">{value}</div><div class="quote-change">{change}</div></div>')
st.markdown('<div class="quote-strip">'+''.join(quote_cards)+'</div>', unsafe_allow_html=True)

nav_home, nav_rank, nav_holdings, nav_market, nav_test = st.tabs(["概览", "选股", "持仓", "市场", "回测"])

with nav_home:
    if market is not None:
        st.markdown(f'''<div class="hero"><div><div class="label">市场环境</div><div class="score">{market.score:.0f}<small> / 100</small></div><div class="regime">{escape(market.regime)}</div></div><div class="hero-side"><div class="label">试验仓位区间</div><div class="hero-value">{escape(market.suggested_equity_exposure)}</div><div class="bar"><span style="width:{market.score}%"></span></div><div class="hero-note">分数越高，模型环境判断越积极。<br>区间阈值尚未验证。</div></div></div>''', unsafe_allow_html=True)
    exposure = invested / account_total * 100 if account_total > 0 else np.nan
    exposure_text = fmt(exposure, "%") if account_total > 0 else ("待录入" if np.isfinite(account_total) else "待确认")
    account_text = "$" + fmt(account_total, digits=0) if account_total > 0 else ("待录入" if np.isfinite(account_total) else "估值不完整")
    candidates = int(scores.total.ge(70).sum())
    st.markdown(f'''<div class="tiles"><div class="tile"><div class="tile-label">账户总值</div><div class="tile-value">{account_text}</div><div class="tile-note">含现金 · 美元</div></div><div class="tile"><div class="tile-label">股票占比</div><div class="tile-value">{exposure_text}</div><div class="tile-note">实际录入持仓</div></div><div class="tile"><div class="tile-label">观察候选</div><div class="tile-value">{candidates}<span style="font-size:12px;font-weight:400"> 只</span></div><div class="tile-note">模型评分 ≥ 70</div></div></div>''', unsafe_allow_html=True)
    brief = []
    if account_total == 0 and holdings.empty:
        brief.append("账户尚未录入：到持仓页填写股数，在设置中填写现金。")
    if not issues.empty:
        brief.append(f"{len(issues)} 个标的评分已暂停，详情见市场页。")
    if market is None:
        brief.append("市场数据不足，环境评分和模拟仓位暂不展示。")
    if not brief:
        brief.append(f"已完成 {len(scores)} 个标的检查；{candidates} 个达到候选评分，因子明细见选股页。")
    st.markdown('<div class="brief">'+'<br>'.join(escape(t) for t in brief)+'</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("信号速览")
        st.markdown('<div class="section-note">按模型分数排序 · 点击“选股”查看全部因子</div>', unsafe_allow_html=True)
        valid = scores[scores.total.notna()].sort_values("total", ascending=False).head(5)
        if valid.empty:
            st.info("当前没有通过数据检查的评分。")
        for _, row in valid.iterrows():
            change = f"{row.score_delta:+.1f} 较前日" if np.isfinite(row.score_delta) else "暂无可比日变化"
            st.markdown(f'''<div class="stock-row"><div><div class="symbol">{escape(row.ticker)}</div><div class="stock-sub">${row.price:,.2f} · {escape(row['状态'])} · 动量 {row.momentum_12_1_pct:+.1f}%</div></div><div><div class="rank-score">{row.total:.0f}<small style="font-size:11px;font-weight:400"> 分</small></div><div class="rank-sub">{change}</div><div class="mini-track"><span style="width:{min(100, max(0, row.total))}%"></span></div></div></div>''', unsafe_allow_html=True)
    with right:
        st.subheader("市场轨迹")
        st.markdown('<div class="section-note">近60个交易日 · 起点归一为100 · 复权价格</div>', unsafe_allow_html=True)
        prices = data["Close"].reindex(columns=["SPY", "QQQ"]).tail(60)
        st.line_chart(prices / prices.iloc[0] * 100, color=["#2764df", "#37a58b"], height=230)
        if holdings.empty:
            st.info("先到“持仓”录入股数和成本，即可查看自己的风险分布。")
        elif np.isfinite(account_total) and account_total > 0:
            top = holdings.sort_values("market_value", ascending=False).iloc[0]
            st.caption(f"最大单一持仓：{top.ticker} · 占账户 {top.weight_pct:.1f}%")

with nav_rank:
    st.subheader("选股观察")
    st.caption("评分用于排序和研究。动量采用约12个月至1个月前的区间，排除最近21个交易日。")
    display = scores[["ticker", "total", "状态", "price", "score_delta", "momentum_12_1_pct", "atr_pct", "error"]].rename(columns={"ticker":"代码", "total":"模型分", "price":"价格 $", "score_delta":"日变化", "momentum_12_1_pct":"动量12-1月%", "atr_pct":"ATR%", "error":"数据问题"})
    search_col, score_col = st.columns([1, 1])
    query = search_col.text_input("搜索股票代码", placeholder="例如 MU、NVDA", key="stock_search").strip().upper()
    minimum = score_col.slider("最低模型评分", 0, 100, 0, key="minimum_score")
    filtered = display[display["代码"].str.contains(query, regex=False)]
    if minimum > 0:
        filtered = filtered[filtered["模型分"].ge(minimum)]
    st.caption(f"显示 {len(filtered)} / {len(display)} 个标的 · 默认包含数据不足的标的")
    if filtered.empty:
        st.info("没有匹配结果，请调整代码或最低评分。")
    else:
        st.dataframe(filtered, hide_index=True, width="stretch")
    st.download_button("导出当前选股结果", filtered.to_csv(index=False).encode("utf-8-sig"), "stock_scores.csv", "text/csv")
    choices = scores.ticker.tolist()
    chosen = st.selectbox("查看个股因子", choices)
    r = scores.set_index("ticker").loc[chosen]
    factor_names = {"trend":"趋势 / 25", "momentum":"动量 / 25", "relative_strength":"相对强弱 / 15", "volume":"量价 / 10", "risk":"风险 / 10"}
    factors = pd.DataFrame({"因子":list(factor_names.values()), "分数":[r[k] for k in factor_names]})
    st.bar_chart(factors.set_index("因子"), color="#2764df", horizontal=True, height=230)
    if include_fundamentals:
        st.caption(f"基本面快照：{fmt(r.fundamentals)} / 15；未提供可靠财报发布日期。")
    else:
        st.caption("基本面未启用：不填充固定分，其余85分归一为100分。")

with nav_holdings:
    st.subheader("我的持仓")
    if not holdings.empty:
        st.dataframe(holdings[["ticker", "shares", "price", "avg_cost", "pnl_pct", "weight_pct"]].rename(columns={"ticker":"代码", "shares":"股数", "price":"价格 $", "avg_cost":"成本 $", "pnl_pct":"盈亏%", "weight_pct":"账户占比%"}), hide_index=True, width="stretch")
        if np.isfinite(account_total) and account_total > 0:
            themes = holdings.assign(theme=holdings.theme.fillna("其他")).groupby("theme").market_value.sum() / account_total * 100
            st.caption("主题分布 · 占整个账户的比例")
            st.bar_chart(themes, color="#37a58b", horizontal=True, height=180)
    render_portfolio_editor()

with nav_market:
    st.subheader("市场与数据")
    if market is not None:
        components = pd.DataFrame({"因子":market.components.keys(), "得分":market.components.values()})
        st.bar_chart(components.set_index("因子"), color="#2764df", horizontal=True, height=260)
        for note in market.notes:
            st.warning(note)
    st.caption("股票池宽度只代表当前股票池，不能视为整个美股市场的宽度。")
    st.dataframe(audit, hide_index=True, width="stretch")
    if market is not None:
        with st.expander("查看模拟组合分配"):
            weights = target_weights(data, tickers, benchmark, expected)
            st.caption("使用默认技术模式：每周末观察信号，下一交易日收盘模拟执行。历史回测使用完全相同的分配函数。")
            st.dataframe(pd.DataFrame({"代码":weights.index, "目标占比%":weights.values*100}).query('`目标占比%` > 0'), hide_index=True, width="stretch")
            st.caption(f"剩余现金：{(1-weights.sum())*100:.1f}%")

with nav_test:
    st.subheader("策略实验室")
    st.caption("默认技术模式 · 当前股票池 · 最近5年 · SPY / QQQ 对照")
    st.info("每周最后交易日形成信号，下一交易日收盘模拟调仓；调仓间持仓随价格自然变化。")
    cost = st.number_input("手续费与滑点合计 · 单边基点（10基点=0.1%）", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    st.caption("当前股票池存在幸存者偏差，未含退市样本、历史成分调整、税费、现金利息。复权价格为总回报近似；回测不是已有效的证明。")
    if st.button("运行回测", type="primary", key="run_backtest"):
        try:
            with st.spinner("正在计算历史信号，首次运行可能需要几分钟…"):
                curves, report, trades, skipped = historical_test(tuple(tickers), benchmark, cost)
            st.caption(f"评价区间 {curves.index[0].date()} — {curves.index[-1].date()} · 前253日用于预热")
            st.dataframe(report.round(2), hide_index=True, width="stretch")
            st.line_chart(curves[["Strategy", "SPY", "QQQ"]], color=["#2764df", "#37a58b", "#a9b6c6"])
            st.download_button("下载净值", curves.to_csv().encode("utf-8-sig"), "backtest_result.csv", "text/csv")
            st.download_button("下载调仓记录", trades.to_csv(index=False).encode("utf-8-sig"), "backtest_trades.csv", "text/csv")
            if skipped:
                st.warning(f"{len(skipped)}个信号日因数据不足跳过，保留原持仓。")
                st.dataframe(pd.DataFrame(skipped), hide_index=True)
        except Exception as exc:
            st.error(f"回测未完成：{exc}")

st.markdown('<div class="foot">美股驾驶舱 · V1.3<br>研究与模拟用途；评分、候选和仓位区间不代表收益承诺。数据来自 Yahoo Finance，按已完成交易日日线计算。</div>', unsafe_allow_html=True)
