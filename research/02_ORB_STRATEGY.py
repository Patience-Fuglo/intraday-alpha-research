# ==============================================================
# *** RESEARCH WORKFLOW ***
#
#   Idea → Data → Features → Signal → Backtest → Metrics → Robustness
#
# ==============================================================

"""
Opening Range Breakout (ORB) Strategy
======================================

Hypothesis:
    The first 30 minutes of trading establish a high and low.
    A breakout above the high or below the low signals institutional
    commitment to a direction. Price continues in that direction.

Signal logic:
    Long  (+1): price breaks ABOVE the 30-min opening range high with volume
    Short (-1): price breaks BELOW the 30-min opening range low with volume
    Exit      : end of day (flatten all positions before close)
    Filter    : volume > 2.5x average (institution confirmation)
                price move > 0.2% beyond range (quality filter)
                time window 10am–11am ET only (peak momentum window)

Why this is different from VWAP+RSI:
    VWAP+RSI = mean reversion = bets AGAINST the move
    ORB      = momentum       = bets WITH the move
    ORB works in trending markets — exactly where VWAP+RSI failed

The Five Numbers — read in this order every run:
    1. Gross Return  — does the signal have edge before fees?
                       Gross negative = close hypothesis immediately
                       Gross positive = fix costs, keep researching
    2. Total Costs   — what is the fee gap to close?
    3. Total Return  — net result after costs (the accountant's number)
    4. Trades        — enough observations to trust? (minimum 50)
    5. Sharpe        — return per unit of risk (benchmark 1.0)
    + Max Drawdown   — worst losing streak (risk control)

IC Note (Information Coefficient):
    IC applies to ML signals only — measures whether a model's
    continuous predictions correlate with actual returns.
    ORB uses a binary rule (+1/-1), not a continuous score.
    IC is not applicable here. IC is introduced in 03_ML_RIDGE_SIGNAL.py.

PSR Note (Probabilistic Sharpe Ratio):
    PSR = probability the true Sharpe is above zero, accounting
    for sample size and fat tails. Requires sufficient trade count.
    With 60-day Yahoo Finance data and the 10am–11am filter,
    ORB produces ~2 trades per ticker — far below the PSR minimum.
    PSR validation for ORB requires QuantConnect (3+ years of data).

Walk-Forward Note:
    Walk-forward (train on past, test on unseen future) applies
    to ML signals with learned weights. ORB uses fixed rules with
    no training step — the signal is the same rule applied to
    any data window. Walk-forward is introduced in 03_ML_RIDGE_SIGNAL.py.

Research Workflow:
    Idea → Data → Features → Signal → Backtest → Metrics → Robustness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ============================================================
# STEP 1 — GET DATA
# ============================================================

def download_data(ticker, period="60d", interval="5m"):
    """Download intraday OHLCV data."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


# ============================================================
# STEP 2 — BUILD FEATURES
# ============================================================

def add_orb_features(df):
    """
    For each day, compute:
      - orb_high : high of the first 30 minutes (bars 0–5 of the day)
      - orb_low  : low of the first 30 minutes
      - orb_done : True after 10:00am — opening range is locked in
      - avg_vol  : 20-bar rolling average volume (for volume filter)
    """
    df = df.copy()
    df["date"] = df.index.date

    orb_high_map = {}
    orb_low_map  = {}

    for date, group in df.groupby("date"):
        # First 6 bars = 9:30, 9:35, 9:40, 9:45, 9:50, 9:55 = 30 minutes
        opening_bars = group.head(6)
        orb_high_map[date] = opening_bars["high"].max()
        orb_low_map[date]  = opening_bars["low"].min()

    df["orb_high"] = df["date"].map(orb_high_map)
    df["orb_low"]  = df["date"].map(orb_low_map)

    # Mark bars that are AFTER the opening range (bar index > 5 for that day)
    df["bar_of_day"] = df.groupby("date").cumcount()
    df["orb_done"]   = df["bar_of_day"] >= 6

    # Volume filter: is this bar's volume above the 20-bar rolling average?
    df["avg_vol"]      = df["volume"].rolling(20).mean()
    df["high_volume"]  = df["volume"] > df["avg_vol"] * 2.5

    return df


# ============================================================
# STEP 3 — BUILD SIGNAL
# ============================================================

def build_orb_signal(df, min_move=0.002):
    """
    Long  when: price closes above orb_high by at least min_move AND volume is high AND after 10am
    Short when: price closes below orb_low  by at least min_move AND volume is high AND after 10am
    Only one trade per day — first signal wins.

    min_move = 0.002 means price must break at least 0.2% beyond the opening range.
    This filters out tiny meaningless breakouts that cost fees but earn nothing.
    """
    df = df.copy()
    df["signal"] = 0

    # Time filter: only enter between 10:00am and 11:59am ET (= 14:00–15:59 UTC)
    in_time_window = (df.index.hour == 14) | (df.index.hour == 15)

    long_cond  = (
        df["orb_done"] &
        (df["close"] > df["orb_high"] * (1 + min_move)) &
        df["high_volume"] &
        in_time_window
    )

    short_cond = (
        df["orb_done"] &
        (df["close"] < df["orb_low"] * (1 - min_move)) &
        df["high_volume"] &
        in_time_window
    )

    df.loc[long_cond,  "signal"] = 1
    df.loc[short_cond, "signal"] = -1

    # One trade per day: keep only the first signal each day, zero the rest
    def first_signal_only(group):
        first_idx = (group["signal"] != 0).idxmax()
        if group.loc[first_idx, "signal"] == 0:
            group["signal"] = 0
        else:
            mask = group.index > first_idx
            group.loc[mask, "signal"] = 0
        return group

    df = df.groupby("date", group_keys=False).apply(first_signal_only)
    return df


# ============================================================
# STEP 4 — EXIT LOGIC
# ============================================================

def apply_eod_exit(df):
    """
    Hold position from entry until end of day.
    Flatten at the last bar of each day.
    """
    df = df.copy()
    position  = 0
    positions = []

    for i, (_, row) in enumerate(df.iterrows()):
        # Check if this is the last bar of the day
        is_last_bar = (row["bar_of_day"] == df[df["date"] == row["date"]]["bar_of_day"].max())

        if position == 0 and row["signal"] != 0:
            position = row["signal"]

        if is_last_bar:
            position = 0

        positions.append(position)

    df["position"] = positions
    return df


# ============================================================
# STEP 5 — BACKTEST WITH COSTS
# ============================================================

def backtest(df, commission_bps=5, slippage_bps=2):
    """Backtest ORB with realistic costs."""
    df = df.copy()
    df["return"] = df["close"].pct_change()

    # shift(1) prevents look-ahead bias
    df["position_lagged"] = df["position"].shift(1).fillna(0)
    df["gross_return"]    = df["position_lagged"] * df["return"]

    total_cost_bps    = commission_bps + slippage_bps
    df["turnover"]    = df["position_lagged"].diff().abs().fillna(0)
    df["cost"]        = df["turnover"] * (total_cost_bps / 10000)
    df["net_return"]  = df["gross_return"] - df["cost"]
    df["equity"]      = (1 + df["net_return"].fillna(0)).cumprod()
    return df


# ============================================================
# STEP 6 — METRICS
# ============================================================

def compute_metrics(df, bars_per_year=252 * 78):
    """Compute Total Return, Max Drawdown, Trades, Sharpe, Gross vs Net."""
    returns      = df["net_return"].dropna()
    total_return = df["equity"].iloc[-1] - 1
    running_high = df["equity"].cummax()
    max_drawdown = (df["equity"] / running_high - 1).min()
    trades       = int((df["turnover"] > 0).sum())

    gross_return = df["gross_return"].sum()
    net_return   = df["net_return"].sum()
    total_costs  = gross_return - net_return

    if returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)

    return {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Trades":       trades,
        "Sharpe":       sharpe,
        "Gross Return": gross_return,
        "Total Costs":  total_costs
    }


# ============================================================
# STEP 7 — FULL PIPELINE FOR ONE TICKER
# ============================================================

def run_one_ticker(ticker, period="60d", interval="5m"):
    """Run the full ORB pipeline for one ticker."""
    df = download_data(ticker, period, interval)
    df = add_orb_features(df)
    df = build_orb_signal(df)
    df = apply_eod_exit(df)
    bt = backtest(df)
    metrics = compute_metrics(bt)
    return bt, metrics


# ============================================================
# STEP 8 — MULTI-TICKER
# ============================================================

def run_multi_ticker(tickers, period="60d", interval="5m"):
    """Run ORB across multiple tickers."""
    rows = []
    results = {}

    for ticker in tickers:
        try:
            bt, metrics = run_one_ticker(ticker, period, interval)
            metrics["Ticker"] = ticker
            results[ticker]   = bt
            rows.append(metrics)
        except Exception as e:
            print(f"{ticker} failed: {e}")

    metrics_df = pd.DataFrame(rows).set_index("Ticker")
    return results, metrics_df


# ============================================================
# STEP 9 — PLOT
# ============================================================

def plot_equity(bt, title="ORB Equity Curve"):
    """Plot equity curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(bt.index, bt["equity"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 10 — RESEARCH DECISION
# ============================================================

def research_decision(metrics_df):
    """Print go/no-go decision."""
    avg = metrics_df.mean(numeric_only=True)
    print("\n=== AVERAGE RESULTS ===")
    print(avg)

    if avg["Total Return"] > 0 and avg["Sharpe"] > 1:
        print("\nDecision: Interesting. Continue testing out-of-sample.")
    else:
        print("\nDecision: Not enough yet. Improve one thing and retest.")


# ============================================================
# MAIN — RUN THIS
# ============================================================

if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]

    print("\n=== ORB — ONE TICKER (AAPL) ===")
    bt, metrics = run_one_ticker("AAPL", period="60d", interval="5m")
    print("AAPL", metrics)
    plot_equity(bt, title="AAPL ORB Strategy")

    print("\n=== ORB — MULTI-TICKER ===")
    results, metrics_df = run_multi_ticker(tickers, period="60d", interval="5m")
    print(metrics_df)
    research_decision(metrics_df)
