"""Experimental close-to-close paper backtest using the website's exact technical rules.
A signal at a week's final session executes at the NEXT session's close.
Adjusted prices are total-return approximations; current stock pool has survivorship bias.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from quant_model import MARKET_SYMBOLS, download_prices, completed_daily_data, target_weights


def run_backtest(data, tickers, benchmark="QQQ", fee_bps=10.0):
    if not 0 <= fee_bps <= 100:
        raise ValueError("单边成本必须在0–100基点之间。")
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("股票池不能为空")
    data = data.sort_index().copy()
    data.index = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
    if data.index.has_duplicates:
        raise ValueError("日期重复")
    close = data["Close"].copy()
    if isinstance(close, pd.Series):
        raise ValueError("回测需要多标的行情")
    close = close.reindex(columns=list(dict.fromkeys(tickers + ["SPY", "QQQ"])))
    if len(close) < 260:
        raise ValueError("至少需要260个交易日，其中253日用于指标预热。")
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=close.index.min().date(), end_date=(close.index.max() + pd.Timedelta(days=7)).date())
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    weekly_ends = set(pd.Series(sessions, index=sessions).groupby(sessions.to_period("W-FRI")).max())
    dates = close.index[252:]
    cash = 1.0
    units = pd.Series(0.0, index=tickers)
    pending = None
    curve, trades, skipped = [], [], []
    cost_rate = fee_bps / 10000.0
    for date in dates:
        prices = close.loc[date, tickers].astype(float)
        owned = units > 0
        if ((~np.isfinite(prices) | (prices <= 0)) & owned).any():
            raise ValueError(f"{date.date()} 已持有标的行情缺失，停止回测，不能按零估值。")
        values = units * prices.fillna(0)
        nav = cash + float(values.sum())
        if pending is not None:
            signal_date, weights = pending
            missing = (weights > 0) & (~np.isfinite(prices) | (prices <= 0))
            if missing.any():
                raise ValueError(f"{date.date()} 模拟成交价格缺失，停止回测。")
            post_fee_nav = nav
            for _ in range(30):
                fee = float((weights * post_fee_nav - values).abs().sum()) * cost_rate
                updated = nav - fee
                if abs(updated - post_fee_nav) < 1e-13:
                    post_fee_nav = updated
                    break
                post_fee_nav = updated
            targets = weights * post_fee_nav
            turnover = float((targets - values).abs().sum())
            cash = post_fee_nav - float(targets.sum())
            units = (targets / prices.where(prices > 0)).fillna(0.0)
            nav = cash + float((units * prices.fillna(0)).sum())
            trades.append({"signal_date": signal_date, "execution_date": date,
                           "turnover_fraction": turnover / (nav + turnover * cost_rate),
                           "cost": turnover * cost_rate, "equity_weight": float(weights.sum())})
            pending = None
        curve.append({"date": date, "Strategy": nav, "Cash": cash})
        if date in weekly_ends:
            hist = data.loc[:date]
            try:
                weights = target_weights(hist, tickers, benchmark, asof=date).reindex(tickers).fillna(0)
                if (weights < 0).any() or weights.sum() > 1.0000001:
                    raise ValueError("模拟目标仓位无效")
                pending = (date, weights)
            except ValueError as exc:
                # Missing market data does not fabricate a sell or buy signal; retain positions.
                skipped.append({"date": str(date.date()), "reason": str(exc)})
    result = pd.DataFrame(curve).set_index("date")
    for ticker in ["SPY", "QQQ"]:
        series = close.loc[dates, ticker]
        if not np.isfinite(series).all() or (series <= 0).any():
            raise ValueError(f"{ticker} 基准数据不完整")
        result[ticker] = series / series.iloc[0] / (1 + cost_rate)
    metrics = []
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    for name in ["Strategy", "SPY", "QQQ"]:
        nav = result[name]
        ret = nav.pct_change().dropna()
        metrics.append({"组合": name, "累计收益%": (nav.iloc[-1] - 1) * 100,
                        "年化收益%": (nav.iloc[-1] ** (1 / years) - 1) * 100,
                        "最大回撤%": (nav / nav.cummax().clip(lower=1.0) - 1).min() * 100,
                        "年化波动%": ret.std() * np.sqrt(252) * 100})
    return result, pd.DataFrame(metrics), pd.DataFrame(trades), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="5y", choices=["2y", "5y", "10y"])
    parser.add_argument("--fee_bps", default=10.0, type=float)
    parser.add_argument("--benchmark", default="QQQ", choices=["SPY", "QQQ"])
    args = parser.parse_args()
    tickers = pd.read_csv(Path(__file__).with_name("watchlist.csv"))["ticker"].tolist()
    data, _ = completed_daily_data(download_prices(list(dict.fromkeys(tickers + MARKET_SYMBOLS)), period=args.period))
    result, metrics, trades, skipped = run_backtest(data, tickers, args.benchmark, args.fee_bps)
    print(metrics.to_string(index=False))
    print(f"跳过信号: {len(skipped)}；当前股票池存在幸存者偏差，结果不代表未来收益。")
    result.to_csv("backtest_result.csv")
    trades.to_csv("backtest_trades.csv", index=False)


if __name__ == "__main__":
    main()
