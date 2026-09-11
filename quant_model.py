from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal


DEFAULT_WATCHLIST = [
    "NVDA", "AVGO", "AMD", "MU", "MRVL", "MSFT", "GOOGL", "AMZN",
    "META", "PLTR", "TSM", "ASML", "AAPL", "NFLX", "ORCL", "CRM",
    "JPM", "V", "MA", "COST", "WMT", "LLY", "UNH"
]

MARKET_SYMBOLS = ["SPY", "QQQ", "^VIX", "^TNX", "HYG"]


@dataclass
class MarketRiskResult:
    score: float
    regime: str
    suggested_equity_exposure: str
    components: Dict[str, float]
    notes: List[str]


def download_prices(
    tickers: Iterable[str],
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted close / OHLCV data via yfinance."""
    tickers = list(dict.fromkeys(tickers))
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No market data returned. Check network access or ticker symbols.")
    return data


def _field(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """Handle both single-ticker and multi-ticker yfinance layouts."""
    if isinstance(data.columns, pd.MultiIndex):
        return data[field][ticker].dropna()
    return data[field].dropna()


def latest_completed_session(now=None) -> pd.Timestamp:
    """NYSE date whose close was at least 30 minutes ago; includes holidays/early closes."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=(now - pd.Timedelta(days=20)).date(), end_date=now.date())
    complete = schedule[schedule["market_close"] + pd.Timedelta(minutes=30) <= now]
    if complete.empty:
        raise ValueError("无法确定最近已完成的交易日，暂停计算。")
    return pd.Timestamp(complete.index[-1]).normalize()


def completed_daily_data(data: pd.DataFrame, now=None):
    cutoff = latest_completed_session(now)
    data = data.copy()
    data.index = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
    if data.index.has_duplicates:
        raise ValueError("行情出现重复日期，暂停计算。")
    data = data.sort_index().loc[:cutoff].dropna(how="all")
    if data.empty:
        raise ValueError("没有已完成交易日的数据。")
    return data, cutoff


def require_history(data, ticker, sessions, asof=None, fields=("Close",)):
    """Fail closed on incomplete or stale required observations."""
    if data.empty:
        raise ValueError("没有行情")
    target = pd.Timestamp(asof if asof is not None else data.index[-1]).normalize()
    reference = data.index[data.index <= target][-sessions:]
    if len(reference) < sessions or pd.Timestamp(reference[-1]).normalize() != target:
        raise ValueError(f"行情未更新到 {target.date()} 或历史不足{sessions}条")
    for field in fields:
        try:
            series = _field(data, field, ticker).reindex(reference)
        except (KeyError, TypeError):
            raise ValueError(f"缺少{field}数据") from None
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{field}最近{sessions}条存在缺失或无效值")
        if (values < 0).any() or (field != "Volume" and (values == 0).any()):
            raise ValueError(f"{field}包含无效价格或成交量")


def data_quality(data, tickers, asof=None):
    rows = []
    for ticker in dict.fromkeys(tickers):
        last_date = "无数据"
        try:
            close = _field(data, "Close", ticker)
            if not close.empty:
                last_date = str(pd.Timestamp(close.index[-1]).date())
            require_history(data, ticker, 253, asof)
            require_history(data, ticker, 21, asof, ("High", "Low", "Volume"))
            status = "通过"
        except (ValueError, KeyError, TypeError) as exc:
            status = str(exc)
        rows.append({"代码": ticker, "最新数据日": last_date, "检查": status})
    return pd.DataFrame(rows)


def validate_market_data(data, asof=None):
    for ticker, n in [("SPY", 200), ("QQQ", 200), ("HYG", 200), ("^VIX", 6), ("^TNX", 21)]:
        try:
            require_history(data, ticker, n, asof)
        except ValueError as exc:
            raise ValueError(f"市场因子 {ticker}：{exc}；暂停市场评分和仓位输出。") from None


def _last(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")


def _pct_change(series: pd.Series, periods: int) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-1 - periods] - 1.0)


def _safe_score(value: float, lo: float, hi: float, invert: bool = False) -> float:
    if not np.isfinite(value):
        return 50.0
    if hi == lo:
        return 50.0
    x = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    if invert:
        x = 1.0 - x
    return 100.0 * x


def market_risk_score(
    data: pd.DataFrame,
    breadth_tickers: Iterable[str] = DEFAULT_WATCHLIST,
    asof=None,
) -> MarketRiskResult:
    """
    Score 0-100. Higher = more supportive risk environment.
    Components:
      SPY trend 25
      QQQ trend 20
      VIX 15
      US 10Y (^TNX) 15
      HYG trend 15
      Breadth 10
    """
    validate_market_data(data, asof)
    components: Dict[str, float] = {}
    notes: List[str] = []

    # SPY trend: price > MA20 > MA50 > MA200
    spy = _field(data, "Close", "SPY")
    spy20, spy50, spy200 = spy.rolling(20).mean(), spy.rolling(50).mean(), spy.rolling(200).mean()
    spy_points = 0.0
    if _last(spy) > _last(spy20): spy_points += 6.0
    if _last(spy) > _last(spy50): spy_points += 6.0
    if _last(spy) > _last(spy200): spy_points += 8.0
    if _last(spy20) > _last(spy50) > _last(spy200): spy_points += 5.0
    components["SPY趋势"] = spy_points

    # QQQ trend
    qqq = _field(data, "Close", "QQQ")
    q20, q50, q200 = qqq.rolling(20).mean(), qqq.rolling(50).mean(), qqq.rolling(200).mean()
    q_points = 0.0
    if _last(qqq) > _last(q20): q_points += 5.0
    if _last(qqq) > _last(q50): q_points += 5.0
    if _last(qqq) > _last(q200): q_points += 6.0
    if _last(q20) > _last(q50) > _last(q200): q_points += 4.0
    components["QQQ趋势"] = q_points

    # VIX: lower is friendlier; include recent acceleration penalty
    vix = _field(data, "Close", "^VIX")
    v = _last(vix)
    v5 = _pct_change(vix, 5)
    if not np.isfinite(v):
        vix_points = 7.5
    elif v < 15:
        vix_points = 15.0
    elif v < 20:
        vix_points = 12.0
    elif v < 25:
        vix_points = 8.0
    elif v < 30:
        vix_points = 4.0
    else:
        vix_points = 1.0
    if np.isfinite(v5) and v5 > 0.20:
        vix_points = max(0.0, vix_points - 3.0)
        notes.append("VIX 5日涨幅超过20%，风险升温。")
    components["VIX"] = vix_points

    # 10Y: scoring primarily by 20d rate-of-change, not absolute level
    tnx = _field(data, "Close", "^TNX")
    t20 = _pct_change(tnx, 20)
    if not np.isfinite(t20):
        tnx_points = 7.5
    elif t20 <= -0.08:
        tnx_points = 15.0
    elif t20 <= -0.03:
        tnx_points = 12.0
    elif t20 <= 0.03:
        tnx_points = 9.0
    elif t20 <= 0.08:
        tnx_points = 5.0
    else:
        tnx_points = 2.0
        notes.append("10年期美债收益率近20日上行较快。")
    components["10Y利率"] = tnx_points

    # HYG trend as credit-risk proxy
    hyg = _field(data, "Close", "HYG")
    h50, h200 = hyg.rolling(50).mean(), hyg.rolling(200).mean()
    h_points = 0.0
    if _last(hyg) > _last(h50): h_points += 6.0
    if _last(hyg) > _last(h200): h_points += 6.0
    if _last(h50) > _last(h200): h_points += 3.0
    components["HYG信用"] = h_points

    # Breadth: percentage of available watchlist above 200DMA
    above, valid = 0, 0
    for t in breadth_tickers:
        try:
            require_history(data, t, 200, asof)
            s = _field(data, "Close", t)
        except Exception:
            continue
        ma200 = s.rolling(200).mean()
        if len(s.dropna()) >= 200 and np.isfinite(_last(ma200)):
            valid += 1
            above += int(_last(s) > _last(ma200))
    if not valid:
        raise ValueError("股票池没有可用的200日数据，暂停市场评分。")
    breadth_ratio = above / valid
    breadth_points = 10.0 * breadth_ratio
    components["股票池宽度（非全市场）"] = breadth_points

    total = round(sum(components.values()), 1)
    if total >= 80:
        regime, exposure = "🟢 Risk On", "80%–90%"
    elif total >= 60:
        regime, exposure = "🟢 偏积极", "60%–80%"
    elif total >= 40:
        regime, exposure = "🟡 中性", "40%–60%"
    elif total >= 20:
        regime, exposure = "🟠 谨慎", "20%–40%"
    else:
        regime, exposure = "🔴 Risk Off", "0%–20%"

    return MarketRiskResult(total, regime, exposure, components, notes)


def fundamental_score(ticker: str) -> Tuple[float, Dict[str, Optional[float]]]:
    """
    Lightweight fundamental factor using yfinance metadata.
    If unavailable, return NaN; never fabricate a neutral observation.
    This is intentionally conservative because metadata fields can be missing.
    """
    metrics = {
        "revenueGrowth": None,
        "earningsGrowth": None,
        "grossMargins": None,
        "operatingMargins": None,
        "freeCashflow": None,
    }
    try:
        info = yf.Ticker(ticker).info or {}
        for k in metrics:
            v = info.get(k)
            metrics[k] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None

        parts = []
        if metrics["revenueGrowth"] is not None:
            parts.append(_safe_score(metrics["revenueGrowth"], -0.10, 0.35))
        if metrics["earningsGrowth"] is not None:
            parts.append(_safe_score(metrics["earningsGrowth"], -0.20, 0.50))
        if metrics["grossMargins"] is not None:
            parts.append(_safe_score(metrics["grossMargins"], 0.15, 0.75))
        if metrics["operatingMargins"] is not None:
            parts.append(_safe_score(metrics["operatingMargins"], 0.05, 0.40))
        if metrics["freeCashflow"] is not None:
            parts.append(70.0 if metrics["freeCashflow"] > 0 else 25.0)

        return (float(np.mean(parts)) if parts else float("nan")), metrics
    except Exception:
        return float("nan"), metrics


def stock_score(
    data: pd.DataFrame,
    ticker: str,
    benchmark: str = "QQQ",
    include_fundamentals: bool = False,
    asof=None,
) -> Dict[str, float]:
    require_history(data, ticker, 253, asof)
    require_history(data, ticker, 21, asof, ("High", "Low", "Volume"))
    require_history(data, benchmark, 253, asof)
    close = _field(data, "Close", ticker)
    volume = _field(data, "Volume", ticker)
    bench = _field(data, "Close", benchmark)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    # 1) Trend 25
    trend = 0.0
    if _last(close) > _last(ma20): trend += 5
    if _last(close) > _last(ma50): trend += 5
    if _last(close) > _last(ma200): trend += 7
    if _last(ma20) > _last(ma50): trend += 4
    if _last(ma50) > _last(ma200): trend += 4

    # 2) Momentum 25: percentile-like transform of weighted returns
    r1 = _pct_change(close, 21)
    r3 = _pct_change(close, 63)
    r6 = _pct_change(close, 126)
    r12 = _pct_change(close, 252)
    # Approximate the research 12-to-2-month signal with 252/21 trading-day offsets.
    momentum_12_1 = float(close.iloc[-22] / close.iloc[-253] - 1.0)
    momentum = 25.0 * (_safe_score(momentum_12_1, -0.25, 0.60) / 100.0)

    # 3) Relative strength vs benchmark 15
    rs3 = r3 - _pct_change(bench, 63) if np.isfinite(r3) else float("nan")
    rs6 = r6 - _pct_change(bench, 126) if np.isfinite(r6) else float("nan")
    rs_value = np.nanmean([rs3, rs6])
    relative_strength = 15.0 * (_safe_score(rs_value, -0.20, 0.30) / 100.0)

    # 4) Volume / accumulation 10
    vol20 = volume.rolling(20).mean()
    daily_ret = close.pct_change()
    up_volume = volume.where(daily_ret > 0, 0).rolling(20).sum()
    down_volume = volume.where(daily_ret < 0, 0).rolling(20).sum()
    volume_score = 5.0
    if np.isfinite(_last(vol20)) and _last(vol20) > 0:
        ratio = _last(volume) / _last(vol20)
        if daily_ret.iloc[-1] > 0 and ratio >= 1.5:
            volume_score += 2.5
        elif daily_ret.iloc[-1] < 0 and ratio >= 1.5:
            volume_score -= 2.0
    if _last(up_volume) > _last(down_volume):
        volume_score += 2.5
    volume_score = max(0.0, min(10.0, volume_score))

    # 5) Fundamentals 15
    f_raw, _ = fundamental_score(ticker) if include_fundamentals else (float("nan"), {})
    fundamentals = 15.0 * f_raw / 100.0

    # 6) Risk 10: reward lower realized volatility and shallower 1y drawdown
    ret = close.pct_change().dropna()
    vol = float(ret.tail(63).std() * math.sqrt(252)) if len(ret) >= 20 else float("nan")
    peak = close.tail(252).cummax()
    dd = close.tail(252) / peak - 1.0
    max_dd = float(dd.min()) if not dd.empty else float("nan")
    vol_part = _safe_score(vol, 0.15, 0.70, invert=True)
    dd_part = _safe_score(abs(max_dd), 0.10, 0.60, invert=True)
    risk = 10.0 * (0.6*vol_part + 0.4*dd_part) / 100.0

    technical = trend + momentum + relative_strength + volume_score + risk
    # Disabled fundamentals are excluded, not replaced by a constant. Thresholds remain experimental.
    total = round(technical + fundamentals if include_fundamentals else technical / 85.0 * 100.0, 1)

    # ATR for position/risk sizing
    high = _field(data, "High", ticker)
    low = _field(data, "Low", ticker)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr14 = _last(tr.rolling(14).mean())
    atr_pct = atr14 / _last(close) if np.isfinite(atr14) and _last(close) else float("nan")

    return {
        "ticker": ticker,
        "data_date": str(pd.Timestamp(close.index[-1]).date()),
        "momentum_12_1_pct": round(momentum_12_1 * 100, 2),
        "error": "基本面数据缺失，暂停总分" if include_fundamentals and not np.isfinite(f_raw) else "",
        "total": total,
        "trend": round(trend, 1),
        "momentum": round(momentum, 1),
        "relative_strength": round(relative_strength, 1),
        "volume": round(volume_score, 1),
        "fundamentals": round(fundamentals, 1),
        "risk": round(risk, 1),
        "price": round(_last(close), 2),
        "atr_pct": round(atr_pct * 100, 2) if np.isfinite(atr_pct) else np.nan,
        "r1m_pct": round(r1 * 100, 2) if np.isfinite(r1) else np.nan,
        "r3m_pct": round(r3 * 100, 2) if np.isfinite(r3) else np.nan,
        "r6m_pct": round(r6 * 100, 2) if np.isfinite(r6) else np.nan,
        "r12m_pct": round(r12 * 100, 2) if np.isfinite(r12) else np.nan,
    }


def score_universe(
    data: pd.DataFrame,
    tickers: Iterable[str],
    benchmark: str = "QQQ",
    include_fundamentals: bool = False,
    asof=None,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            rows.append(stock_score(data, ticker, benchmark, include_fundamentals, asof))
        except Exception as e:
            rows.append(dict.fromkeys(["total", "price", "trend", "momentum", "relative_strength", "volume", "fundamentals", "risk", "atr_pct", "r1m_pct", "r3m_pct", "r6m_pct", "r12m_pct", "momentum_12_1_pct"], np.nan) | {"ticker": ticker, "error": str(e), "data_date": ""})
    df = pd.DataFrame(rows)
    if "total" in df:
        df = df.sort_values("total", ascending=False, na_position="last").reset_index(drop=True)
    return df


def suggested_position_pct(score: float, atr_pct: float, market_score: float) -> float:
    """
    Simple position sizing:
    - score gates
    - risk environment multiplier
    - ATR volatility cap
    - hard cap 10%
    """
    if not all(np.isfinite(v) for v in (score, atr_pct, market_score)) or score < 60:
        return 0.0

    base = 0.05
    if score >= 80:
        base = 0.08
    elif score >= 70:
        base = 0.06

    market_mult = 1.0
    if market_score < 20:
        market_mult = 0.25
    elif market_score < 40:
        market_mult = 0.50
    elif market_score < 60:
        market_mult = 0.75

    vol_mult = 1.0
    if np.isfinite(atr_pct):
        if atr_pct >= 6:
            vol_mult = 0.50
        elif atr_pct >= 4:
            vol_mult = 0.70
        elif atr_pct >= 3:
            vol_mult = 0.85

    return round(min(0.10, base * market_mult * vol_mult) * 100, 1)


def target_weights(data, tickers, benchmark="QQQ", asof=None):
    """Shared weekly paper-portfolio rules; fundamental snapshots deliberately excluded."""
    market = market_risk_score(data, breadth_tickers=tickers, asof=asof)
    scores = score_universe(data, tickers, benchmark, False, asof)
    weights = pd.Series({r["ticker"]: suggested_position_pct(r["total"], r["atr_pct"], market.score) / 100
                         for _, r in scores.iterrows()}, dtype=float)
    bands = market.suggested_equity_exposure.replace("%", "").split("–")
    cap = sum(float(x) for x in bands) / 200
    if weights.sum() > cap:
        weights *= cap / weights.sum()
    return weights
