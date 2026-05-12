# ==============================================================
# *** RESEARCH WORKFLOW ***
#
#   Idea → Data → Features → Signal → Backtest → Metrics → Robustness
#
# ==============================================================

"""
VS Code Intraday Alpha Cheat Sheet
==================================

Beginner → Advanced reusable research template.

Starts from the SAME TradingView idea:
    VWAP + RSI mean reversion

Then upgrades it like a professional research workflow:
    one ticker → costs → slippage → exits → metrics → parameter testing → multi-ticker

    "Change one thing, test, compare, decide."

Research workflow:
    Idea → Data → Features → Signal → Backtest → Metrics → Robustness
"""

# ============================================================
#  STEP 0 — INSTALL
# ============================================================
# Run once:
# pip install yfinance pandas numpy matplotlib

# Loads all the tools we need: math, data tables, charts, and stock data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ============================================================
#  STEP 1 — IDEA
# ============================================================
# The hypothesis: when price strays too far from VWAP, it tends to snap back — we bet on that snap.
#
# "Prices overreact intraday and revert to VWAP."


# ============================================================
#  STEP 2 — GET DATA
# ============================================================
# "One ticker teaches the idea. Many tickers test robustness."

# Pulls 5-minute OHLCV bars from Yahoo Finance and returns a clean DataFrame with lowercase column names.
def download_data(ticker, period="5d", interval="5m"):
    """Download one ticker of intraday data."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    df = df.dropna()
    # yfinance >=0.2 returns MultiIndex columns like ('Close', 'AAPL') — flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


# ============================================================
#  STEP 3 — FEATURES
# ============================================================
# Features = inputs used to build the signal.
# VWAP = fair value | RSI = extreme condition
#
# "Features turn raw prices into useful signals."

# Computes the running average price weighted by volume — intraday fair value, reset each day.
def add_vwap(df):
    """VWAP = cumulative price-volume / cumulative volume."""
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    return df


# Measures average gains vs average losses over the last N bars — RSI < 30 = oversold, > 70 = overbought.
def add_rsi(df, window=14):
    """RSI detects overbought and oversold conditions."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


# Calls add_vwap, add_rsi, and pct_change in one shot — produces a fully featured DataFrame ready for signals.
def prepare_data(df):
    """Add VWAP, RSI, and returns."""
    df = add_vwap(df)
    df = add_rsi(df)
    df["return"] = df["close"].pct_change()
    return df.dropna()


# ============================================================
#  STEP 4 — SIGNAL
# ============================================================
# Python signal: +1 = long | -1 = short | 0 = flat
#
# "Entry catches the extreme."

# Outputs +1 (buy), -1 (sell), or 0 (flat) at each bar when price is away from VWAP AND RSI confirms the extreme.
def build_signal(df, long_rsi=25, short_rsi=75, distance_filter=0.0):
    """
    Build VWAP-RSI mean-reversion signal.

    distance_filter:
        Require price to be far enough from VWAP.
        0.001 = at least 0.10% away from VWAP.

        "Distance filter reduces weak trades."
    """
    df = df.copy()
    distance_from_vwap = (df["close"] - df["vwap"]) / df["vwap"]
    df["distance_from_vwap"] = distance_from_vwap

    long_cond = (
        (df["close"] < df["vwap"]) &
        (df["rsi"] < long_rsi) &
        (distance_from_vwap < -distance_filter)
    )

    short_cond = (
        (df["close"] > df["vwap"]) &
        (df["rsi"] > short_rsi) &
        (distance_from_vwap > distance_filter)
    )

    df["signal"] = 0
    df.loc[long_cond, "signal"] = 1
    df.loc[short_cond, "signal"] = -1
    return df


# ============================================================
#  STEP 5 — EXIT LOGIC
# ============================================================
# "Exit captures the reversion."

# Walks through bars one by one, holding the position until price crosses back through VWAP — that IS the reversion.
def apply_vwap_exit(df):
    """
    Long exits when close > VWAP.
    Short exits when close < VWAP.
    """
    df = df.copy()
    position = 0
    positions = []

    for _, row in df.iterrows():
        signal = row["signal"]

        if position == 0:
            if signal == 1:
                position = 1
            elif signal == -1:
                position = -1
        elif position == 1:
            if row["close"] > row["vwap"]:
                position = 0
        elif position == -1:
            if row["close"] < row["vwap"]:
                position = 0

        positions.append(position)

    df["position"] = positions
    return df


# ============================================================
#  STEP 6 — BACKTEST ENGINE WITH COSTS + SLIPPAGE
# ============================================================
# "Costs and slippage make the test more realistic."

# Simulates trading the signal bar by bar: shift(1) prevents look-ahead bias, then deducts commissions and slippage from gross P&L.
def backtest(df, use_vwap_exit=True, commission_bps=5, slippage_bps=2, max_position=1.0):
    """Backtest one ticker with cost and slippage."""
    df = df.copy()

    if use_vwap_exit:
        df = apply_vwap_exit(df)
    else:
        df["position"] = df["signal"]

    df["position"] = df["position"].clip(-max_position, max_position)
    # shift(1) means we use yesterday's signal to trade today — prevents look-ahead bias.
    df["position_lagged"] = df["position"].shift(1).fillna(0)
    df["gross_return"] = df["position_lagged"] * df["return"]

    # Turnover measures how much the position changes.
    df["turnover"] = df["position_lagged"].diff().abs().fillna(0)

    # Total cost per position change.
    total_cost_bps = commission_bps + slippage_bps
    df["cost"] = df["turnover"] * (total_cost_bps / 10000)

    df["net_return"] = df["gross_return"] - df["cost"]
    df["equity"] = (1 + df["net_return"].fillna(0)).cumprod()
    return df


# ============================================================
#  STEP 7 — PERFORMANCE METRICS
# ============================================================
# "I evaluate P&L, drawdown, trades, and Sharpe."

# Summarizes the backtest in 4 numbers: total return, worst peak-to-trough loss, trade count, and annualized Sharpe.
def compute_metrics(df, bars_per_year=252 * 78):
    """Compute Total Return, Max Drawdown, Trades, and Sharpe."""
    returns = df["net_return"].dropna()
    total_return = df["equity"].iloc[-1] - 1

    running_high = df["equity"].cummax()
    drawdown = df["equity"] / running_high - 1
    max_drawdown = drawdown.min()

    trades = int((df["turnover"] > 0).sum())

    if returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)

    return {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Trades": trades,
        "Sharpe": sharpe
    }


# ============================================================
#  STEP 8 — MULTI-TICKER BACKTEST
# ============================================================
# "A signal should work beyond one chart."

# Runs the full pipeline (data → features → signal → backtest → metrics) for a single ticker in one call.
def run_one_ticker(ticker, period="5d", interval="5m", long_rsi=25, short_rsi=75,
                   distance_filter=0.0, commission_bps=5, slippage_bps=2,
                   use_vwap_exit=True):
    """Run full pipeline for one ticker."""
    df = download_data(ticker, period, interval)
    df = prepare_data(df)
    df = build_signal(df, long_rsi, short_rsi, distance_filter)
    bt = backtest(df, use_vwap_exit, commission_bps, slippage_bps)
    metrics = compute_metrics(bt)
    return bt, metrics


# Loops run_one_ticker over a list of stocks and stacks results into a side-by-side metrics table for robustness comparison.
def run_multi_ticker(tickers, period="5d", interval="5m", long_rsi=25, short_rsi=75,
                     distance_filter=0.0, commission_bps=5, slippage_bps=2,
                     use_vwap_exit=True):
    """Run the same alpha across many tickers."""
    results = {}
    rows = []

    for ticker in tickers:
        bt, metrics = run_one_ticker(
            ticker=ticker,
            period=period,
            interval=interval,
            long_rsi=long_rsi,
            short_rsi=short_rsi,
            distance_filter=distance_filter,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            use_vwap_exit=use_vwap_exit
        )
        metrics["Ticker"] = ticker
        results[ticker] = bt
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows).set_index("Ticker")
    return results, metrics_df


# ============================================================
#  STEP 9 — PARAMETER TESTING
# ============================================================
# Warning: do not just pick the best result — that is overfitting.
# Good question: does performance improve smoothly across the range?
#
# "I improve one parameter at a time."

# Grid-searches every combination of RSI thresholds and distance filters, averaging Sharpe across all tickers for each combo.
def parameter_test(tickers, long_values=(20, 25, 30), short_values=(70, 75, 80),
                   distance_values=(0.0, 0.001), period="5d", interval="5m"):
    """Test different RSI thresholds and VWAP distance filters."""
    rows = []

    for long_rsi in long_values:
        for short_rsi in short_values:
            for distance_filter in distance_values:
                _, metrics_df = run_multi_ticker(
                    tickers=tickers,
                    period=period,
                    interval=interval,
                    long_rsi=long_rsi,
                    short_rsi=short_rsi,
                    distance_filter=distance_filter
                )
                avg = metrics_df.mean(numeric_only=True)
                rows.append({
                    "long_rsi": long_rsi,
                    "short_rsi": short_rsi,
                    "distance_filter": distance_filter,
                    "avg_total_return": avg["Total Return"],
                    "avg_max_drawdown": avg["Max Drawdown"],
                    "avg_trades": avg["Trades"],
                    "avg_sharpe": avg["Sharpe"]
                })

    return pd.DataFrame(rows)


# ============================================================
#  STEP 10 — PLOT
# ============================================================
# "The equity curve shows the story."

# Draws the growth of $1 invested over time — a rising curve = money made, sharp drops = drawdowns.
def plot_equity(bt, title="Equity Curve"):
    """Plot one strategy equity curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(bt.index, bt["equity"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()


# ============================================================
#  STEP 11 — DECIDE
# ============================================================
# Ask: did P&L improve? drawdown? trades? survived costs? worked across tickers?
#
# "I keep improvements and reject what does not work."

# Averages results across all tickers and prints a go/no-go: Sharpe > 1 AND positive return = worth pursuing further.
def research_decision(metrics_df):
    """Print a simple research decision."""
    avg = metrics_df.mean(numeric_only=True)
    print("\n=== AVERAGE RESULTS ===")
    print(avg)

    if avg["Total Return"] > 0 and avg["Sharpe"] > 1:
        print("\nDecision: Interesting. Continue testing out-of-sample.")
    else:
        print("\nDecision: Not enough yet. Improve one thing and retest.")


# ============================================================
#  MAIN EXAMPLE
# ============================================================
# This is the part you run in VS Code.
# Beginner: Run AAPL first.
# Advanced: Then run multiple tickers.

if __name__ == "__main__":

    # Beginner: one ticker first
    ticker = "AAPL"
    bt, metrics = run_one_ticker(
        ticker=ticker,
        period="60d",
        interval="5m",
        long_rsi=25,
        short_rsi=75,
        distance_filter=0.0,
        commission_bps=5,
        slippage_bps=2,
        use_vwap_exit=True
    )
    print("\n=== ONE TICKER RESULT ===")
    print(ticker, metrics)
    plot_equity(bt, title="AAPL VWAP-RSI Alpha")

    # Advanced: multiple tickers
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
    results, metrics_df = run_multi_ticker(
        tickers=tickers,
        period="60d",
        interval="5m",
        long_rsi=25,
        short_rsi=75,
        distance_filter=0.0,
        commission_bps=5,
        slippage_bps=2,
        use_vwap_exit=True
    )
    print("\n=== MULTI-TICKER RESULTS ===")
    print(metrics_df)
    research_decision(metrics_df)

    # Parameter testing
    sweep = parameter_test(
        tickers=tickers,
        long_values=(20, 25, 30),
        short_values=(70, 75, 80),
        distance_values=(0.0, 0.001),
        period="60d",
        interval="5m"
    )
    print("\n=== PARAMETER TEST RESULTS ===")
    print(sweep.sort_values("avg_sharpe", ascending=False))


# ============================================================
# RESEARCH SUMMARY
# ============================================================
# Hypothesis: intraday prices dislocate from VWAP and mean-revert.
#
# Signal: long below VWAP when RSI is oversold.
#         short above VWAP when RSI is overbought.
#
# Validation pipeline:
#   Single ticker (AAPL) first, then multi-ticker robustness.
#   Measured net of transaction costs: commission and slippage.
#   Exit at VWAP — structurally consistent with the mean-reversion thesis.
#   Parameter sweep across RSI thresholds and VWAP distance filters.
#
# Standard: positive, stable performance after costs across a universe.
