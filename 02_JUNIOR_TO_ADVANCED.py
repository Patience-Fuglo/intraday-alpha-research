"""
================================================================================
MASTER INTRADAY ALPHA RESEARCH CHEAT SHEET
================================================================================
PURPOSE
    Single file. Entry level to production senior quant.
    Daily practice, learning, and interview preparation.
    VWAP + RSI mean-reversion signal — intraday equities.

THE RESEARCH LOOP  (every professional quant lives by this)
    Idea --> Data --> Features --> Signal --> Backtest --> Metrics --> Robustness

    Idea       : What market inefficiency am I exploiting?
    Data       : Source and clean the price/volume history.
    Features   : Engineer measurable inputs from raw data.
    Signal     : Combine features into a tradeable entry/exit rule.
    Backtest   : Simulate historical execution bar by bar.
    Metrics    : Quantify risk-adjusted performance.
    Robustness : Prove it works across tickers, periods, and parameters.

LEARNING PATH
    SECTION 1  [ENTRY]         Single ticker, basic signal, basic backtest
    SECTION 2  [JUNIOR]        Costs, slippage, stop loss, Sortino, parameter sweep
    SECTION 3  [INTERMEDIATE]  Multi-ticker, composite alpha score, cross-sectional L/S
    SECTION 4  [ADVANCED]      Risk controls — vol scaling, sector neutrality, beta hedge
    SECTION 5  [SENIOR/PROD]   Out-of-sample validation, Kyle lambda, statistical testing
    SECTION 6  [CONCEPTS]      Momentum, pairs trading, statistics reference
    SECTION 7  [FRAMEWORK]     Decision checklist, visualization, interview talking points

CORE MEMORY LINE
    "Change one thing, test, compare, decide."

DISCLAIMER
    Research and learning framework only.
    Not financial advice. Not a live trading system.
    Production funds use institutional data, compliance, and execution infrastructure.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Run: pip install yfinance pandas numpy matplotlib")


# ================================================================================
# ================================================================================
# SECTION 1  [ENTRY LEVEL]
# SINGLE TICKER — VWAP + RSI MEAN REVERSION — AAPL
# ================================================================================
# ================================================================================
#
# THE HYPOTHESIS
# ---------------
# Intraday prices dislocate from fair value and mean-revert back toward VWAP.
# We exploit that reversion: fade the extreme, exit at fair value.
#
# WHO uses this idea:
#   Statistical arbitrage desks, intraday prop shops, market-making desks.
#
# WHEN it works:
#   Range-bound, high-liquidity names. Normal intraday sessions.
#
# WHEN it breaks:
#   Trending days driven by news, earnings, or macro shocks.
#   Price keeps moving away from VWAP — the reversion never comes.
#
# MEMORY: "Prices overreact. VWAP is fair value. Fade the extreme."


# ================================================================================
# STEP 1.1 — DOWNLOAD DATA
# ================================================================================
#
# TERM: OHLCV = Open, High, Low, Close, Volume — the five columns of a price bar.
# TERM: interval = bar width. 5m = each row covers 5 minutes of trading.
# TERM: period = how far back to download. "5d" = last 5 trading days.
#
# WHY 5-minute bars:
#   Fine enough to capture intraday patterns.
#   Coarse enough that data is clean and transaction costs are manageable.
#   1-minute bars are noisier and costs hurt more per signal.
#
# REAL FUND NOTE:
#   Professionals use Bloomberg, Refinitiv, or Databento tick data.
#   yfinance is free and sufficient for learning the mechanics.

def download_one_ticker(ticker="AAPL", period="5d", interval="5m"):
    """Download intraday OHLCV data for one ticker."""
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,   # adjusts for splits and dividends
        progress=False
    )
    df = df.dropna()
    # yfinance returns MultiIndex columns like ('Close','AAPL') — flatten to 'close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


# ================================================================================
# STEP 1.2 — VWAP FEATURE
# ================================================================================
#
# TERM: VWAP = Volume-Weighted Average Price
#   Formula: cumulative(typical_price × volume) / cumulative(volume)
#   Typical price = (high + low + close) / 3
#
# WHY volume-weighted, not a simple average?
#   Bars with heavy volume represent more meaningful price discoveries.
#   A simple average treats a thin 9am bar identically to a heavy 10am bar.
#   VWAP weights toward where the most trading actually occurred.
#
# WHY cumulative from market open?
#   VWAP is an intraday benchmark. It resets each session.
#   It answers: "relative to all trading today, is price cheap or expensive?"
#
# INTERVIEW LINE:
#   "VWAP is the institutional intraday fair-value benchmark. I use the
#    distance from VWAP to measure the magnitude of price dislocation."

def add_vwap(df):
    """Add VWAP column — cumulative from start of the data window."""
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    return df


# ================================================================================
# STEP 1.3 — RSI FEATURE
# ================================================================================
#
# TERM: RSI = Relative Strength Index (range 0 to 100)
#   RSI < 30 = oversold  — price fell sharply, may revert upward
#   RSI > 70 = overbought — price rose sharply, may revert downward
#
# CALCULATION:
#   Separate each bar's price change into gains and losses.
#   RS = rolling_avg_gain / rolling_avg_loss over N bars.
#   RSI = 100 − (100 / (1 + RS))
#
# WHY pair RSI with VWAP?
#   VWAP tells WHERE price is (cheap or expensive vs. today's average).
#   RSI tells HOW FAST it got there (extreme momentum = higher reversion probability).
#   Together they filter higher-quality entry points than either alone.
#
# COMMON MISTAKE:
#   Using RSI alone. RSI can stay oversold for many bars in a trending market.
#   VWAP provides the structural anchor that RSI lacks.

def add_rsi(df, window=14):
    """Add RSI column — standard 14-bar window."""
    df = df.copy()
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


# ================================================================================
# STEP 1.4 — ENTRY-LEVEL SIGNAL
# ================================================================================
#
# TERM: Signal = a discrete rule that outputs +1 (long), −1 (short), or 0 (flat).
#
# LOGIC:
#   LONG  (+1): close below VWAP  AND  RSI oversold  (below long_rsi)
#   SHORT (−1): close above VWAP  AND  RSI overbought (above short_rsi)
#   FLAT  ( 0): neither condition met
#
# MEMORY: "Entry catches the extreme."

def build_signal_entry(df, long_rsi=30, short_rsi=70):
    """Basic VWAP-RSI mean-reversion signal. +1 / -1 / 0."""
    df = df.copy()
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

    long_cond  = (df["close"] < df["vwap"]) & (df["rsi"] < long_rsi)
    short_cond = (df["close"] > df["vwap"]) & (df["rsi"] > short_rsi)

    df["signal"] = 0
    df.loc[long_cond,  "signal"] =  1
    df.loc[short_cond, "signal"] = -1
    return df


# ================================================================================
# STEP 1.5 — ENTRY-LEVEL BACKTEST
# ================================================================================
#
# TERM: Backtest = simulate what would have happened trading this signal historically.
#
# CRITICAL RULE — NO LOOK-AHEAD BIAS:
#   position.shift(1) means: use the signal from bar t to trade at bar t+1.
#   Without the shift, you trade on a bar that is not yet closed — impossible in reality.
#   This is the single most common mistake in beginner backtests.
#
# TERM: Equity curve = cumulative value of $1 invested from the start.
#   equity = 1.10 means you grew $1 to $1.10.
#   Computed as: cumprod(1 + net_return)
#
# TERM: Gross return = return before transaction costs.
# TERM: Net return   = return after transaction costs.
# TERM: bps = basis points. 1 bps = 0.01%. 5 bps = 0.05%.
#
# WHY costs matter so much intraday:
#   Intraday signals earn 0.05–0.30% per trade.
#   Costs of 5–10 bps per round-trip consume a large fraction of that edge.
#   A strategy with Sharpe 2.0 gross can become Sharpe 0.0 net after costs.
#   ALWAYS evaluate net return. Gross return is a fantasy.

def backtest_entry(df, commission_bps=5, slippage_bps=2):
    """Entry-level backtest — shift(1) lag, flat cost model."""
    df = df.copy()
    df["ret_1"]           = df["close"].pct_change()
    df["position_lagged"] = df["signal"].shift(1).fillna(0)
    df["gross_return"]    = df["position_lagged"] * df["ret_1"]
    df["turnover"]        = df["position_lagged"].diff().abs().fillna(0)
    df["cost"]            = df["turnover"] * ((commission_bps + slippage_bps) / 10000)
    df["net_return"]      = df["gross_return"] - df["cost"]
    df["equity"]          = (1 + df["net_return"].fillna(0)).cumprod()
    return df


# ================================================================================
# STEP 1.6 — ENTRY-LEVEL METRICS
# ================================================================================
#
# TERM: Sharpe Ratio = annualized (mean_return / std_return)
#   Measures return per unit of total risk (both up and down volatility).
#   > 1.0 = decent. > 2.0 = good. > 3.0 = exceptional for intraday strategies.
#
# TERM: Max Drawdown = worst peak-to-trough equity decline.
#   −0.10 = equity fell 10% from its prior peak at some point.
#   A portfolio manager asks: "what is my worst-case pain?"
#
# WHY annualize Sharpe with sqrt(bars_per_year)?
#   5-minute bars: ~78 bars/day × 252 days/year = 19,656 bars/year.
#   Annualizing allows comparison to daily or weekly strategies.

def compute_metrics_entry(df, bars_per_year=252 * 78):
    """Total return, Sharpe, max drawdown, and trade count."""
    returns      = df["net_return"].dropna()
    total_return = df["equity"].iloc[-1] - 1
    max_drawdown = (df["equity"] / df["equity"].cummax() - 1).min()
    vol          = returns.std()
    sharpe       = (returns.mean() / vol) * np.sqrt(bars_per_year) if vol != 0 else np.nan
    trades       = int((df["turnover"] > 0).sum())
    return {"Total Return": total_return, "Sharpe": sharpe,
            "Max Drawdown": max_drawdown, "Trades": trades}


# ================================================================================
# STEP 1.7 — RUN ENTRY PIPELINE
# ================================================================================

def run_entry_level(ticker="AAPL", period="5d", interval="5m"):
    """
    [ENTRY] Full pipeline: download -> VWAP -> RSI -> signal -> backtest -> metrics.
    Run this first. Understand every output before moving to Section 2.
    """
    df = download_one_ticker(ticker, period, interval)
    df = add_vwap(df)
    df = add_rsi(df)
    df = build_signal_entry(df)
    df = df.dropna()
    bt = backtest_entry(df)
    m  = compute_metrics_entry(bt)
    print(f"\n=== ENTRY LEVEL — {ticker} ===")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return bt, m


# ================================================================================
# ================================================================================
# SECTION 2  [JUNIOR LEVEL]
# IMPROVED SINGLE TICKER — COSTS, STOPS, SORTINO, PARAMETER SWEEP
# ================================================================================
# ================================================================================
#
# UPGRADES FROM ENTRY LEVEL:
#   - Richer feature set (z-score, volume spike, SMA distance)
#   - Distance filter on VWAP deviation (reduces noise trades)
#   - Volume filter (confirms conviction)
#   - Volatility regime filter (avoid trending markets)
#   - Stop loss and profit target
#   - VWAP exit logic (intellectually consistent with the mean-reversion thesis)
#   - Sortino ratio (downside-only risk)
#   - Win rate metric
#   - Parameter sweep across RSI thresholds and distance filters
#
# MEMORY: "One parameter at a time. Test, compare, decide."


# ================================================================================
# STEP 2.1 — FEATURE ENGINEERING
# ================================================================================
#
# TERM: Feature = any computed input to a signal or model.
#   Raw price series are not directly tradeable inputs — features make them actionable.
#
# FEATURE GLOSSARY:
#   ret_1          : 1-bar return — % change from last bar to this bar
#   ret_3          : 3-bar return — short-term directional move
#   ret_6          : 6-bar return — 30-minute momentum or reversal
#   reversal_3     : negative of ret_3 — positive = recent dip (mean-reversion input)
#   vol_20         : rolling 20-bar return std dev — how volatile is price right now?
#   ret_zscore     : z-score of ret_1 — how unusual is this bar's return vs. recent history?
#   vwap_distance  : (close - vwap) / vwap — how far is price from fair value, in %?
#   volume_spike   : current volume / 20-bar avg volume — is activity abnormally high?
#   sma_distance   : (close - 20-bar SMA) / SMA — secondary fair-value deviation measure
#
# WHY z-score the return?
#   Raw returns differ across stocks and regimes. A 0.5% move in NVDA is small;
#   in a utility stock it is extreme. Z-scoring normalizes for fair comparison.

def add_features(df):
    """Compute all features. Returns DataFrame with NaN rows dropped."""
    df = df.copy()
    df = add_vwap(df)
    df = add_rsi(df)

    df["ret_1"]    = df["close"].pct_change(1)
    df["ret_3"]    = df["close"].pct_change(3)
    df["ret_6"]    = df["close"].pct_change(6)
    df["reversal_3"] = -df["ret_3"]

    df["vol_20"]   = df["ret_1"].rolling(20).std()
    ret_mean       = df["ret_1"].rolling(20).mean()
    ret_std        = df["ret_1"].rolling(20).std()
    df["ret_zscore"] = (df["ret_1"] - ret_mean) / ret_std.replace(0, np.nan)

    df["vwap_distance"]  = (df["close"] - df["vwap"]) / df["vwap"]
    df["volume_avg_20"]  = df["volume"].rolling(20).mean()
    df["volume_spike"]   = df["volume"] / df["volume_avg_20"].replace(0, np.nan)
    df["sma_20"]         = df["close"].rolling(20).mean()
    df["sma_distance"]   = (df["close"] - df["sma_20"]) / df["sma_20"].replace(0, np.nan)

    return df.dropna()


# ================================================================================
# STEP 2.2 — JUNIOR SIGNAL WITH FILTERS
# ================================================================================
#
# DISTANCE FILTER:
#   distance_filter=0.001 means price must be at least 0.10% from VWAP.
#   Small deviations are noise; we want meaningful dislocations.
#
# VOLUME FILTER:
#   Confirms conviction. A price move on thin volume is often a fake-out.
#   High volume during a dip = real sellers participated = more likely to exhaust.
#
# VOLATILITY FILTER (max_vol):
#   During high-volatility regimes (news, earnings), mean reversion fails.
#   max_vol=0.003 means only trade when 20-bar return std < 0.3%.
#   This protects the signal from its worst environment.

def build_signal_junior(
    df,
    long_rsi=25,
    short_rsi=75,
    distance_filter=0.001,
    min_volume_spike=0.8,
    use_volume_filter=True,
    max_vol=None
):
    """VWAP-RSI signal with distance, volume, and volatility filters."""
    df = df.copy()

    long_cond  = (df["close"] < df["vwap"]) & (df["rsi"] < long_rsi) & (df["vwap_distance"] < -distance_filter)
    short_cond = (df["close"] > df["vwap"]) & (df["rsi"] > short_rsi) & (df["vwap_distance"] > distance_filter)

    if use_volume_filter:
        long_cond  = long_cond  & (df["volume_spike"] >= min_volume_spike)
        short_cond = short_cond & (df["volume_spike"] >= min_volume_spike)

    if max_vol is not None:
        low_vol    = df["vol_20"] < max_vol
        long_cond  = long_cond  & low_vol
        short_cond = short_cond & low_vol

    df["signal"] = 0
    df.loc[long_cond,  "signal"] =  1
    df.loc[short_cond, "signal"] = -1
    return df


# ================================================================================
# STEP 2.3 — VWAP EXIT LOGIC
# ================================================================================
#
# TERM: Position = what you are holding right now (+1 long, −1 short, 0 flat).
#
# LOGIC:
#   Long entry  → hold until close crosses back ABOVE VWAP (reversion complete)
#   Short entry → hold until close crosses back BELOW VWAP (reversion complete)
#
# WHY VWAP as exit target?
#   Entry is based on dislocation from VWAP. Exit at VWAP = thesis fulfilled.
#   This is intellectually consistent: enter the extreme, exit at fair value.
#
# MEMORY: "Entry catches the extreme. Exit captures the reversion."

def apply_vwap_exit(df):
    """Hold position until price crosses back through VWAP."""
    df       = df.copy()
    position = 0
    positions = []

    for _, row in df.iterrows():
        sig = row["signal"]
        if position == 0:
            if sig ==  1: position =  1
            elif sig == -1: position = -1
        elif position ==  1:
            if row["close"] > row["vwap"]: position = 0
        elif position == -1:
            if row["close"] < row["vwap"]: position = 0
        positions.append(position)

    df["position"] = positions
    return df


# ================================================================================
# STEP 2.4 — JUNIOR BACKTEST WITH STOP LOSS AND PROFIT TARGET
# ================================================================================
#
# TERM: Stop loss = maximum loss on a trade before forced exit.
#   stop_loss_pct=0.003 means exit if trade loses more than 0.3%.
#   WHY: one bad trade should not destroy weeks of gains.
#   Every professional strategy has a stop. This is non-negotiable.
#
# TERM: Profit target = maximum gain at which you take the money and exit.
#   profit_target_pct=0.005 means exit when trade gains more than 0.5%.
#   WHY: locks in profit before the reversion overshoots or reverses.
#
# TERM: Turnover = absolute change in position per bar.
#   Turnover of 1.0 = you moved from full flat to full position (or vice versa).
#   High turnover = high costs. Monitor it carefully.

def backtest_junior(
    df,
    use_vwap_exit=True,
    commission_bps=5,
    slippage_bps=2,
    stop_loss_pct=None,
    profit_target_pct=None,
    max_position=1.0
):
    """Junior backtest with optional stop loss and profit target."""
    df = df.copy()

    if use_vwap_exit:
        df = apply_vwap_exit(df)
    else:
        df["position"] = df["signal"]

    df["position"] = df["position"].clip(-max_position, max_position)

    if stop_loss_pct is not None or profit_target_pct is not None:
        pos_check = df["position"].shift(1).fillna(0)
        bar_ret   = pos_check * df["ret_1"]
        if stop_loss_pct     is not None: df.loc[bar_ret < -abs(stop_loss_pct),     "position"] = 0
        if profit_target_pct is not None: df.loc[bar_ret >  abs(profit_target_pct), "position"] = 0

    df["position_lagged"] = df["position"].shift(1).fillna(0)
    df["gross_return"]    = df["position_lagged"] * df["ret_1"]
    df["turnover"]        = df["position_lagged"].diff().abs().fillna(0)
    df["cost"]            = df["turnover"] * ((commission_bps + slippage_bps) / 10000)
    df["net_return"]      = df["gross_return"] - df["cost"]
    df["equity"]          = (1 + df["net_return"].fillna(0)).cumprod()
    return df


# ================================================================================
# STEP 2.5 — JUNIOR METRICS (SHARPE + SORTINO + WIN RATE)
# ================================================================================
#
# TERM: Sortino Ratio = annualized (mean_return / downside_std)
#   Like Sharpe but penalizes only downside volatility.
#   WHY: upside volatility is not risk — only losses are.
#   Higher Sortino vs. Sharpe = return distribution is positively skewed (good).
#
# TERM: Win Rate = fraction of active bars where net_return > 0.
#   A strategy can be profitable with win rate below 50% if winners > losers.
#   Win rate alone means nothing without average win and average loss sizes.
#
# INTERVIEW LINE:
#   "I evaluate Sharpe for total risk efficiency, Sortino for downside risk,
#    max drawdown for worst-case pain, and turnover to assess cost pressure."

def compute_metrics_junior(df, bars_per_year=252 * 78):
    """Full metrics: total return, Sharpe, Sortino, max drawdown, win rate, trades."""
    returns      = df["net_return"].dropna()
    equity       = df["equity"].dropna()
    total_return = equity.iloc[-1] - 1
    max_drawdown = (equity / equity.cummax() - 1).min()
    vol          = returns.std()
    sharpe       = (returns.mean() / vol)      * np.sqrt(bars_per_year) if vol     != 0 else np.nan
    downside     = returns[returns < 0].std()
    sortino      = (returns.mean() / downside) * np.sqrt(bars_per_year) if downside != 0 else np.nan
    active       = returns[returns != 0]
    win_rate     = float((active > 0).mean()) if len(active) > 0 else np.nan
    trades       = int((df["turnover"] > 0).sum())
    avg_turnover = df["turnover"].mean()
    return {
        "Total Return":     total_return,
        "Sharpe":           sharpe,
        "Sortino":          sortino,
        "Max Drawdown":     max_drawdown,
        "Win Rate":         win_rate,
        "Trades":           trades,
        "Avg Turnover":     avg_turnover,
    }


# ================================================================================
# STEP 2.6 — PARAMETER SWEEP
# ================================================================================
#
# TERM: Parameter sweep / grid search = test all combinations of parameter values.
#
# OVERFITTING WARNING:
#   If you test 100 combinations and pick the best, you are fitting to noise.
#   What you want to see: STABLE results across a RANGE of values.
#   If performance holds only at exactly long_rsi=25 — that is suspicious.
#   If it holds across 20, 25, 30 — that is a robust signal.
#
# HOW to read the output:
#   Sort by Sharpe. Look at the top 10 rows.
#   If many different parameter sets appear in the top 10, the signal is robust.
#   If only one specific combination dominates, be skeptical.

def parameter_sweep_single_ticker(
    ticker="AAPL",
    long_values=(20, 25, 30),
    short_values=(70, 75, 80),
    distance_values=(0.0, 0.001, 0.002),
    period="5d",
    interval="5m"
):
    """Grid search over RSI thresholds and VWAP distance filter."""
    rows = []
    base = download_one_ticker(ticker, period, interval)
    base = add_features(base)

    for long_rsi in long_values:
        for short_rsi in short_values:
            for dist in distance_values:
                df = build_signal_junior(base, long_rsi=long_rsi, short_rsi=short_rsi, distance_filter=dist)
                bt = backtest_junior(df)
                m  = compute_metrics_junior(bt)
                m.update({"long_rsi": long_rsi, "short_rsi": short_rsi, "distance": dist})
                rows.append(m)

    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False)


# ================================================================================
# STEP 2.7 — RUN JUNIOR PIPELINE
# ================================================================================

def run_junior_level(ticker="AAPL", period="5d", interval="5m"):
    """
    [JUNIOR] Single ticker with full cost model, stops, and parameter sweep.
    Run this after Entry. Notice the difference costs and stops make.
    """
    df = download_one_ticker(ticker, period, interval)
    df = add_features(df)
    df = build_signal_junior(df, long_rsi=25, short_rsi=75, distance_filter=0.001)
    bt = backtest_junior(df, stop_loss_pct=0.003, profit_target_pct=0.005)
    m  = compute_metrics_junior(bt)
    print(f"\n=== JUNIOR LEVEL — {ticker} ===")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return bt, m


# ================================================================================
# ================================================================================
# SECTION 3  [INTERMEDIATE LEVEL]
# MULTI-TICKER ROBUSTNESS + CROSS-SECTIONAL LONG/SHORT PORTFOLIO
# ================================================================================
# ================================================================================
#
# THE SHIFT IN THINKING:
#
#   Single-ticker:   "Is AAPL cheap right now?" → BUY or not.
#   Cross-sectional: "Which stocks are cheapest RELATIVE TO EACH OTHER right now?"
#                    → Long the cheapest names, short the most expensive names.
#
# WHY cross-sectional is more powerful:
#   Single-stock signals are noisy — stock-specific events dominate.
#   Cross-sectional ranking naturally hedges market-wide moves.
#   If the whole market drops, longs and shorts both fall — but you profit
#   if your longs fall LESS than your shorts. This is the core of stat arb.
#
# TERM: Alpha = returns NOT explained by market direction.
#   Being long during a bull market is not alpha — anyone can do that.
#   Earning returns while market-neutral IS alpha.
#
# MEMORY: "One ticker can lie. A universe tells the truth."


# ================================================================================
# STEP 3.1 — DOWNLOAD UNIVERSE
# ================================================================================
#
# TERM: Universe = the defined set of securities your strategy trades.
#   Professionals define their universe carefully: liquidity minimums,
#   market cap thresholds, sector constraints, exchange filters.

def download_many_tickers(tickers, period="5d", interval="5m"):
    """Download intraday data for a list of tickers. Returns {ticker: DataFrame}."""
    data = {}
    for ticker in tickers:
        try:
            df = download_one_ticker(ticker, period=period, interval=interval)
            if len(df) > 50:
                data[ticker] = df
        except Exception as e:
            print(f"  Skipped {ticker}: {e}")
    return data


# ================================================================================
# STEP 3.2 — PANEL DATA
# ================================================================================
#
# TERM: Panel data = a table where rows are (ticker, timestamp) combinations.
#   Each row contains: which stock, which bar, and all its features.
#   This structure allows cross-sectional ranking at each timestamp.
#
# WHY panel data is essential:
#   To rank stocks against each other, they must all be in the same table.
#   Separate DataFrames cannot be ranked without first merging into a panel.

def make_panel(data_dict):
    """Stack individual ticker DataFrames into a single panel with features."""
    frames = []
    for ticker, df in data_dict.items():
        x           = add_features(df)
        x["ticker"] = ticker
        frames.append(x)

    panel = pd.concat(frames).reset_index()
    for col in ["Datetime", "Date", "index"]:
        if col in panel.columns:
            panel = panel.rename(columns={col: "datetime"})
            break
    return panel.dropna()


# ================================================================================
# STEP 3.3 — COMPOSITE ALPHA SCORE
# ================================================================================
#
# WHY combine multiple signals into one composite score?
#   Individual signals are noisy. Each captures one aspect of mean reversion.
#   A composite score averages out the noise — the same logic as diversification
#   but applied to signals rather than assets.
#
# SCORE COMPONENTS:
#   score_vwap_reversal : how far below VWAP? (further below = more positive)
#   score_rsi           : how oversold? (RSI=20 → +0.6, RSI=80 → −0.6)
#   score_z_reversal    : was there a recent extreme return? (big drop = positive)
#
# MODIFIERS:
#   volume_confidence   : scale up signals confirmed by volume
#   vol_penalty         : discount high-volatility names (noisier signals)
#
# WEIGHTS (0.40, 0.35, 0.25):
#   VWAP distance gets highest weight — most structurally motivated.
#   RSI and z-score reversal are supporting evidence.
#   These are starting weights. Test variations but beware overfitting.
#
# INTERVIEW LINE:
#   "I build a composite alpha score combining VWAP dislocation, RSI momentum
#    reversal, and return z-score reversal, modulated by volume conviction and
#    a volatility penalty for noisy names."

def build_composite_score(panel):
    """Compute a composite alpha score per (ticker, datetime). Positive = expect rise."""
    p = panel.copy()

    p["score_vwap_reversal"] = -p["vwap_distance"]
    p["score_rsi"]           = (50 - p["rsi"]) / 50
    p["score_z_reversal"]    = -p["ret_zscore"]
    p["volume_confidence"]   = p["volume_spike"].clip(0, 2) / 2
    p["vol_penalty"]         = 1 / (1 + p["vol_20"].rank(pct=True))

    p["raw_score"] = (
        0.40 * p["score_vwap_reversal"] +
        0.35 * p["score_rsi"]           +
        0.25 * p["score_z_reversal"]
    )
    p["raw_score"] = p["raw_score"] * p["volume_confidence"] * p["vol_penalty"]

    return p.dropna()


# ================================================================================
# STEP 3.4 — CROSS-SECTIONAL POSITION SIZING
# ================================================================================
#
# TERM: Cross-sectional = across the universe at ONE point in time.
#   At each timestamp, rank ALL stocks by their score.
#   Top 20% = long. Bottom 20% = short. Middle 60% = flat.
#
# TERM: Quantile = percentage cutoff. 80th quantile = top 20% of scores.
#
# TERM: Volatility scaling = size positions inversely to their volatility.
#   WHY: a position in a high-vol stock carries more risk per dollar.
#   Scaling down high-vol positions equalizes risk contribution across names.
#   This is risk parity at the position level.
#
# TERM: Weight cap (max_weight) = no single stock exceeds X% of portfolio.
#   WHY: concentration risk. One surprise event should not destroy the book.
#
# TERM: Normalization = rescale weights so |long| + |short| = 1.
#   WHY: controls gross exposure regardless of how many names are active.

def build_cross_sectional_positions(
    scored_panel,
    long_quantile=0.80,
    short_quantile=0.20,
    max_weight=0.10,
    use_volatility_scaling=True
):
    """Convert composite scores to portfolio weights per (ticker, datetime)."""
    p         = scored_panel.copy()
    p["rank"] = p.groupby("datetime")["raw_score"].rank(pct=True)

    p["direction"] = 0.0
    p.loc[p["rank"] >= long_quantile,  "direction"] =  1.0
    p.loc[p["rank"] <= short_quantile, "direction"] = -1.0

    active_count = p.groupby("datetime")["direction"].transform(lambda x: (x != 0).sum())
    p["weight"]  = np.where(active_count > 0, p["direction"] / active_count, 0.0)

    if use_volatility_scaling:
        p["inv_vol"] = 1 / p["vol_20"].replace(0, np.nan)
        p["inv_vol"] = p["inv_vol"].replace([np.inf, -np.inf], np.nan)
        p["weight"]  = p["weight"] * p["inv_vol"]

    gross       = p.groupby("datetime")["weight"].transform(lambda x: x.abs().sum())
    p["weight"] = np.where(gross > 0, p["weight"] / gross, 0.0)
    p["weight"] = p["weight"].clip(-max_weight, max_weight)
    gross2      = p.groupby("datetime")["weight"].transform(lambda x: x.abs().sum())
    p["weight"] = np.where(gross2 > 0, p["weight"] / gross2, 0.0)

    return p


# ================================================================================
# STEP 3.5 — ADVANCED COST MODEL
# ================================================================================
#
# TERM: Bid-ask spread = gap between best buy price and best sell price.
#   Buying at the ask and selling at the bid — you pay the spread every round-trip.
#
# TERM: Market impact = your order moves the price against you.
#   Large orders walk up the order book — each lot costs a bit more.
#
# TERM: Round-trip cost = spread + impact + commission (both entry AND exit).
#
# REAL FUND NOTE:
#   Serious funds use the Almgren-Chriss or square-root impact model.
#   We use a flat proxy here — sufficient for learning, not for production.

def estimate_cost(turnover, spread_bps=2, impact_bps=1, commission_bps=1):
    """Three-component cost model: spread + market impact + commission."""
    return turnover * ((spread_bps + impact_bps + commission_bps) / 10000)


# ================================================================================
# STEP 3.6 — PORTFOLIO BACKTEST
# ================================================================================
#
# HOW PORTFOLIO P&L IS COMPUTED:
#   Each stock contributes: weight_lagged × that_stock's_return
#   Portfolio return at time t = sum of all contributions at time t.
#
# TERM: Contribution = one stock's share of portfolio return.
# TERM: Gross exposure = sum of |weights|. Should be ~1.0 if normalized.
# TERM: Net exposure   = sum of weights (longs minus shorts). Should be ~0 if market-neutral.
#
# TERM: Turnover cap = limit on portfolio-level trading per bar.
#   Prevents excessive churn in volatile markets.

def backtest_cross_sectional(
    positions_panel,
    spread_bps=2,
    impact_bps=1,
    commission_bps=1,
    max_turnover_per_bar=None
):
    """Aggregate stock weights into portfolio returns with cost model."""
    p = positions_panel.copy().sort_values(["ticker", "datetime"])

    p["weight_lagged"] = p.groupby("ticker")["weight"].shift(1).fillna(0)
    p["contribution"]  = p["weight_lagged"] * p["ret_1"]
    p["weight_change"] = p.groupby("ticker")["weight_lagged"].diff().abs().fillna(0)

    if max_turnover_per_bar is not None:
        total_to = p.groupby("datetime")["weight_change"].transform("sum")
        scale    = np.where(total_to > max_turnover_per_bar,
                            max_turnover_per_bar / total_to, 1.0)
        p["weight_change"] = p["weight_change"] * scale

    port = p.groupby("datetime").agg(
        gross_return   = ("contribution",  "sum"),
        turnover       = ("weight_change", "sum"),
        gross_exposure = ("weight_lagged", lambda x: x.abs().sum()),
        net_exposure   = ("weight_lagged", "sum"),
        active_names   = ("weight_lagged", lambda x: (x != 0).sum()),
    )
    port["cost"]       = estimate_cost(port["turnover"], spread_bps, impact_bps, commission_bps)
    port["net_return"] = port["gross_return"] - port["cost"]
    port["equity"]     = (1 + port["net_return"].fillna(0)).cumprod()
    return port, p


# ================================================================================
# STEP 3.7 — RUN INTERMEDIATE PIPELINE
# ================================================================================

def run_intermediate_level(period="5d", interval="5m"):
    """
    [INTERMEDIATE] Multi-ticker cross-sectional long/short portfolio.
    Run this after Junior. Notice the improvement from relative ranking.
    """
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "SPY", "QQQ"]
    data    = download_many_tickers(tickers, period=period, interval=interval)
    panel   = make_panel(data)
    scored  = build_composite_score(panel)
    pos     = build_cross_sectional_positions(scored)
    port, _ = backtest_cross_sectional(pos)
    m       = compute_metrics_junior(port)
    print("\n=== INTERMEDIATE — Cross-Sectional Portfolio ===")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return port, m


# ================================================================================
# ================================================================================
# SECTION 4  [ADVANCED LEVEL]
# RISK CONTROLS — SECTOR NEUTRALITY, BETA HEDGE, KYLE LAMBDA
# ================================================================================
# ================================================================================
#
# At this level, the question is no longer just "does the signal work?"
# The question becomes: "WHAT RISK AM I INADVERTENTLY TAKING?"
#
# Hidden risks that destroy alpha claims:
#   - Sector exposure: all your longs happen to be tech stocks — you're just long tech.
#   - Market beta:     net long high-beta names during a bull run — you're just long the market.
#   - Liquidity risk:  your cost model ignored that some names are expensive to trade.
#
# MEMORY: "Alpha is return UNEXPLAINED by known risk factors."


# ================================================================================
# STEP 4.1 — SECTOR NEUTRALIZATION
# ================================================================================
#
# TERM: Sector exposure = portfolio accidentally bets on a sector, not the signal.
#   Example: all longs are tech, tech rallies → profit. But your signal was right?
#   Or did you just get lucky on sector direction?
#
# TERM: Sector-neutral = within each sector, longs and shorts cancel.
#   You profit from RELATIVE performance within sectors, not sector direction.
#   This is the standard at stat arb desks.
#
# HOW it works:
#   Subtract the average weight within each (datetime, sector) group.
#   This centers weights so net sector exposure is approximately zero.

SECTOR_MAP = {
    "AAPL": "Technology",  "MSFT": "Technology",  "NVDA": "Technology",
    "META": "Communication", "GOOGL": "Communication",
    "AMZN": "Consumer",    "TSLA": "Consumer",
    "JPM":  "Financials",  "XOM":  "Energy",
    "SPY":  "ETF",         "QQQ":  "ETF",
}

def add_sector(panel, sector_map=SECTOR_MAP):
    """Label each ticker with its GICS sector."""
    p = panel.copy()
    p["sector"] = p["ticker"].map(sector_map).fillna("Unknown")
    return p

def sector_neutralize_weights(position_panel):
    """Remove average weight within each (datetime, sector) group."""
    p = position_panel.copy()
    p["sector_mean"] = p.groupby(["datetime", "sector"])["weight"].transform("mean")
    p["weight"]      = p["weight"] - p["sector_mean"]
    gross            = p.groupby("datetime")["weight"].transform(lambda x: x.abs().sum())
    p["weight"]      = np.where(gross > 0, p["weight"] / gross, 0.0)
    return p


# ================================================================================
# STEP 4.2 — BETA NEUTRALITY (MARKET EXPOSURE CONTROL)
# ================================================================================
#
# TERM: Beta = sensitivity of a stock's return to the broad market.
#   Beta=1.0 → moves 1:1 with the market.
#   Beta=1.5 → amplifies market moves by 50%.
#
# TERM: Beta-neutral = portfolio has near-zero net market sensitivity.
#   WHY: being net long high-beta names during a bull run is NOT alpha.
#   True alpha earns returns regardless of market direction.
#
# WHAT WE DO HERE:
#   Add SPY return as a market proxy column for visibility.
#   Full beta neutralization uses factor models (Barra, Axioma) in production.
#   You regress each position's return on factors and hedge the residuals.

def add_market_return_proxy(panel, market_ticker="SPY"):
    """Add SPY 1-bar return to every row as a market return reference."""
    p      = panel.copy()
    market = (p[p["ticker"] == market_ticker][["datetime", "ret_1"]]
              .rename(columns={"ret_1": "market_ret"}))
    return p.merge(market, on="datetime", how="left")


# ================================================================================
# STEP 4.3 — KYLE LAMBDA (MARKET MICROSTRUCTURE & PRICE IMPACT)
# ================================================================================
#
# SOURCE: Albert Kyle, "Continuous Auctions and Insider Trading" (1985)
#   One of the most cited papers in market microstructure.
#
# THE CORE EQUATION:
#   delta_price = lambda × order_flow
#
#   delta_price : how much price moved this period
#   order_flow  : net signed volume (buyer-initiated minus seller-initiated)
#   lambda      : price impact coefficient — how much each unit of order flow moves price
#
# INTUITION:
#   High lambda = illiquid stock. Your trades move the price a lot against you.
#   Low lambda  = liquid stock. You can transact large size with minimal impact.
#   Example: AAPL has low lambda — you can buy $1M and barely move it.
#            A small-cap has high lambda — $100K moves it 0.5%.
#
# CONNECTION TO POSITION SIZING:
#   In our flat-bps model, all stocks get the same impact assumption.
#   With Kyle lambda, impact = lambda × trade_size × price.
#   Scale down high-lambda positions to equalize expected impact cost.
#   This is impact-aware position sizing — standard at professional desks.
#
# WHY CUBIST ASKS ABOUT KYLE LAMBDA:
#   1. It is THE standard market impact measure in academic and industry research.
#   2. It directly connects signal generation to execution cost modeling.
#   3. It separates candidates who understand microstructure from those who don't.
#
# INTERVIEW ANSWER STRUCTURE:
#   Step 1 — Define: "Kyle lambda measures price impact per unit of signed order flow.
#             The model is: delta_price = lambda × order_flow."
#   Step 2 — Intuition: "High lambda = illiquid. Your trades hurt you more."
#   Step 3 — Apply: "I scale position sizes inversely to lambda so impact cost
#             is equalized across names."
#
# PROXY NOTE:
#   True lambda requires TAQ signed tick data. Here we use the tick rule:
#   signed_volume = volume × sign(price_change). Well-known approximation.

def estimate_kyle_lambda(df, window=20):
    """
    Estimate Kyle lambda via rolling OLS: lambda = Cov(delta_price, signed_vol) / Var(signed_vol).
    Higher = less liquid = more impact per share traded.
    """
    df = df.copy()
    df["price_change"]  = df["close"].diff()
    df["signed_volume"] = df["volume"] * np.sign(df["price_change"])

    lambdas = []
    for i in range(len(df)):
        if i < window:
            lambdas.append(np.nan)
            continue
        w  = df.iloc[i - window : i]
        pc = w["price_change"].dropna()
        sv = w["signed_volume"].dropna()
        if len(pc) < 5 or sv.std() == 0:
            lambdas.append(np.nan)
        else:
            lam = np.cov(pc, sv)[0, 1] / np.var(sv)
            lambdas.append(max(lam, 0))

    df["kyle_lambda"] = lambdas
    return df

def compare_kyle_lambda(tickers, period="5d", interval="5m"):
    """Compare Kyle lambda across tickers. Lower = more liquid = cheaper to trade."""
    data = download_many_tickers(tickers, period=period, interval=interval)
    rows = []
    for ticker, df in data.items():
        df = estimate_kyle_lambda(df)
        rows.append({"Ticker": ticker, "Mean Lambda": df["kyle_lambda"].mean(),
                     "Median Lambda": df["kyle_lambda"].median()})
    result = pd.DataFrame(rows).set_index("Ticker").sort_values("Mean Lambda")
    print("\n=== KYLE LAMBDA (lower = more liquid) ===")
    print(result.to_string())
    print("\n  Scale positions inversely to lambda to equalize impact cost.")
    return result


# ================================================================================
# STEP 4.4 — RUN ADVANCED PIPELINE
# ================================================================================

def run_advanced_level(period="5d", interval="5m"):
    """
    [ADVANCED] Cross-sectional portfolio with sector neutralization and cost model.
    Adds: sector neutralization, market return proxy, advanced cost model.
    """
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "XOM", "SPY", "QQQ"]
    data    = download_many_tickers(tickers, period=period, interval=interval)
    panel   = make_panel(data)
    panel   = add_market_return_proxy(panel)
    panel   = add_sector(panel)
    scored  = build_composite_score(panel)
    pos     = build_cross_sectional_positions(scored)
    pos     = add_sector(pos)
    pos     = sector_neutralize_weights(pos)
    port, stock_level = backtest_cross_sectional(pos)
    m       = compute_metrics_junior(port)
    print("\n=== ADVANCED — Sector-Neutral Cross-Sectional Portfolio ===")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return port, stock_level, m


# ================================================================================
# ================================================================================
# SECTION 5  [SENIOR / PRODUCTION LEVEL]
# OUT-OF-SAMPLE VALIDATION + STATISTICAL SIGNIFICANCE
# ================================================================================
# ================================================================================
#
# At production level, the only result that matters is out-of-sample performance.
# In-sample results are expected to look good — they were developed on that data.
# OOS performance tells you whether the signal has genuine predictive power.
#
# TERM: In-sample (IS) = the period where you developed and tuned the strategy.
#   IS performance is optimistic by construction. Do not trust it alone.
#
# TERM: Out-of-sample (OOS) = a period the strategy has never seen.
#   This is the only honest test of whether the signal generalizes.
#   If OOS degrades sharply vs. IS, the strategy is overfit.
#
# BENCHMARK FOR SHARPE DECAY:
#   IS Sharpe=2.5, OOS Sharpe=1.8 → modest decay, expected, signal is credible
#   IS Sharpe=3.0, OOS Sharpe=0.2 → collapsed, overfit or data-mined
#   OOS Sharpe negative             → destroyed value on unseen data — stop here
#
# MEMORY: "In-sample teaches. Out-of-sample judges."


# ================================================================================
# STEP 5.1 — CHRONOLOGICAL TRAIN/TEST SPLIT
# ================================================================================
#
# CRITICAL: Always split CHRONOLOGICALLY, never randomly.
#   Random splits leak future information into the training set.
#   This produces artificially inflated IS performance.
#   A random split on time series data is look-ahead bias at the split level.

def split_train_test(df, datetime_col=None, split_ratio=0.70):
    """70/30 chronological split. Train on past. Test on future."""
    x = df.copy()
    if datetime_col:
        times     = sorted(x[datetime_col].dropna().unique())
        split_t   = times[int(len(times) * split_ratio)]
        return x[x[datetime_col] <= split_t], x[x[datetime_col] > split_t]
    else:
        n = int(len(x) * split_ratio)
        return x.iloc[:n], x.iloc[n:]


# ================================================================================
# STEP 5.2 — STATISTICAL SIGNIFICANCE
# ================================================================================
#
# T-STATISTIC FOR SHARPE:
#   t = Sharpe × sqrt(N / bars_per_year)
#   N = total bars in backtest.
#   t > 2.0 = significant at 95% confidence.
#   t > 3.0 = strong credibility for a strategy.
#
#   Example: Sharpe=1.5, 1 year of 5-min bars (N=19,656):
#     t = 1.5 × sqrt(1) = 1.5. Weak.
#   Same Sharpe over 5 years (N=98,280):
#     t = 1.5 × sqrt(5) = 3.35. Much stronger.
#
# MULTIPLE TESTING PROBLEM:
#   Testing 100 parameter combinations → 5 will pass t > 2.0 by chance alone.
#   Apply Bonferroni correction: threshold = 0.05 / N_tests.
#   Or use the Deflated Sharpe Ratio (Lopez de Prado 2014).
#   Rule of thumb: require t > 3.0 for strategies found via parameter search.
#
# SIGNAL DECAY:
#   Compare Sharpe in first half of OOS vs. second half of OOS.
#   Accelerating decay = the signal is being arbitraged away.
#
# AUTOCORRELATION:
#   If returns have positive autocorrelation, annualized Sharpe using
#   sqrt(bars_per_year) OVERSTATES true risk-adjusted performance.
#   Fix: use Newey-West standard errors or block bootstrap.

def compute_tstat(sharpe, n_bars, bars_per_year=252 * 78):
    """t-stat for a Sharpe ratio. t > 2.0 = significant. t > 3.0 = credible."""
    return sharpe * np.sqrt(n_bars / bars_per_year)

def print_statistical_significance(df, sharpe, label="Strategy"):
    """Report t-stat and significance level for the backtest."""
    n = len(df["net_return"].dropna())
    t = compute_tstat(sharpe, n)
    sig = "*** STRONG"  if t > 3.0 else "** OK" if t > 2.0 else "* WEAK — need more data"
    print(f"\n  {label}: Sharpe={sharpe:.2f}, N={n:,} bars, t-stat={t:.2f} {sig}")


# ================================================================================
# STEP 5.3 — RUN PRODUCTION VALIDATION
# ================================================================================

def run_production_level(period="5d", interval="5m"):
    """
    [SENIOR/PRODUCTION] Full pipeline with IS/OOS split and statistical testing.
    This is the only honest way to claim a signal has predictive value.
    """
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "XOM", "SPY", "QQQ"]
    data    = download_many_tickers(tickers, period=period, interval=interval)
    panel   = make_panel(data)
    panel   = add_sector(panel)
    scored  = build_composite_score(panel)

    train_scored, test_scored = split_train_test(scored, datetime_col="datetime", split_ratio=0.70)

    def build_and_backtest(s):
        pos     = build_cross_sectional_positions(s)
        pos     = add_sector(pos)
        pos     = sector_neutralize_weights(pos)
        port, _ = backtest_cross_sectional(pos)
        return port

    train_port = build_and_backtest(train_scored)
    test_port  = build_and_backtest(test_scored)

    train_m = compute_metrics_junior(train_port)
    test_m  = compute_metrics_junior(test_port)

    print("\n=== PRODUCTION — In-Sample (Train) ===")
    for k, v in train_m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== PRODUCTION — Out-of-Sample (Test) ===")
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    is_sharpe  = train_m.get("Sharpe",  float("nan"))
    oos_sharpe = test_m.get("Sharpe",   float("nan"))

    if not np.isnan(is_sharpe) and not np.isnan(oos_sharpe) and is_sharpe != 0:
        decay = (is_sharpe - oos_sharpe) / abs(is_sharpe) * 100
        print(f"\n  IS Sharpe={is_sharpe:.2f}  OOS Sharpe={oos_sharpe:.2f}  Decay={decay:.1f}%")
        if   decay < 0:  print("  --> OOS better than IS. Unusual — check for data issues.")
        elif decay < 40: print("  --> Decay < 40%: signal appears robust. Continue.")
        else:            print("  --> Decay > 40%: likely overfit. Simplify and retest.")

    print_statistical_significance(test_port, oos_sharpe, "OOS")
    return train_port, test_port, train_m, test_m


# ================================================================================
# ================================================================================
# SECTION 6  [CONCEPTS]
# STRATEGY TYPES + STATISTICS REFERENCE — FOR INTERVIEWS
# ================================================================================
# ================================================================================
#
# You do not need to have built all of these. You need to DISCUSS them.
# Know: the hypothesis, the signal logic, the key risk, and how it differs
# from the VWAP mean-reversion strategy you built.


# ================================================================================
# CONCEPT A — CROSS-SECTIONAL MOMENTUM
# ================================================================================
#
# HYPOTHESIS:
#   Stocks that outperformed recently will continue to outperform.
#   Winners keep winning. Losers keep losing.
#   This is the OPPOSITE of mean reversion.
#
# SIGNAL LOGIC:
#   At each timestamp, rank by return over the past N bars.
#   Long the top quantile (recent winners). Short the bottom (recent losers).
#   Structure is IDENTICAL to Section 3 — only the score DIRECTION flips.
#
# KEY DIFFERENCE:
#   Mean reversion: big DOWN move = BUY (expect bounce)
#   Momentum:       big UP move   = BUY (expect continuation)
#
# WHEN EACH WORKS:
#   Mean reversion: range-bound markets, intraday, high-liquidity names.
#   Momentum:       trending markets, daily/weekly frequency, post-news.
#
# IMPLEMENTATION — ONE LINE CHANGE FROM SECTION 3:
#   Change: p["raw_score"] = -p["vwap_distance"] ...   (reversal)
#   To:     p["raw_score"] =  p["ret_6"]          ...   (momentum)
#   Everything else is unchanged.
#
# CUBIST INTERVIEW LINE:
#   "Mean reversion and momentum are structurally identical in my pipeline —
#    both use the same cross-sectional ranking. Only the score direction flips.
#    This is why building a clean scoring layer matters: you swap the signal
#    without rebuilding the portfolio construction or backtest infrastructure."

def momentum_score_concept(panel):
    """Drop-in for build_composite_score() using momentum instead of reversal."""
    p = panel.copy()
    p["raw_score"] = (
        0.40 * p["ret_6"]                           +
        0.35 * p.get("momentum_6", p["ret_6"])      +
        0.25 * p["volume_spike"].clip(0, 2) / 2
    )
    return p.dropna()


# ================================================================================
# CONCEPT B — PAIRS TRADING / STATISTICAL ARBITRAGE
# ================================================================================
#
# HYPOTHESIS:
#   Two historically correlated stocks share a long-run equilibrium.
#   When their spread deviates, it reverts. Fade the deviation.
#   Long the underperformer, short the outperformer.
#
# TERM: Spread = Price_A − hedge_ratio × Price_B
#   hedge_ratio chosen so the spread is stationary (mean-reverting).
#   Estimated via regression: Price_A = beta × Price_B + epsilon.
#
# TERM: Cointegration = two series share a long-run equilibrium.
#   Even if each price wanders, their DIFFERENCE is stable.
#   Test with Engle-Granger or Johansen before assuming mean reversion.
#
# SIGNAL LOGIC:
#   1. Find a cointegrated pair (e.g., AAPL/MSFT, XOM/CVX).
#   2. Compute spread: spread_t = Price_A − beta × Price_B.
#   3. Z-score: z = (spread − mean) / std.
#   4. Long when z < −2. Short when z > +2. Exit when z returns to 0.
#
# DIFFERENCE FROM SECTION 3:
#   Section 3: rank MANY stocks against each other. Market-neutral portfolio.
#   Pairs: hedge ONE stock against ONE specific counterpart. Stock-specific hedge.
#
# KEY RISK — PAIR DIVERGENCE:
#   The cointegration relationship can break permanently.
#   Example: two energy companies where one pivots to renewables.
#   Monitor whether the cointegration remains valid continuously.
#
# CUBIST INTERVIEW LINE:
#   "Pairs trading is mean reversion on a relative-value spread rather than
#    an absolute price. The signal logic is the same — fade the extreme,
#    exit at the mean — but the target is a statistically constructed spread
#    between two instruments. The additional step is verifying cointegration
#    before treating the spread as mean-reverting."


# ================================================================================
# CONCEPT C — STATISTICS REFERENCE
# ================================================================================
#
# HYPOTHESIS TESTING:
#   H0 (null): the signal has no predictive power (Sharpe = 0).
#   H1 (alt):  the signal has real predictive power (Sharpe > 0).
#   p-value: probability of seeing this result if H0 is true.
#   p < 0.05: reject H0 at 95% confidence.
#   BUT: testing 100 parameters → 5 pass by chance (multiple testing problem).
#   SOLUTION: Bonferroni correction or require t-stat > 3.0.
#
# SHARPE T-STAT:
#   t = Sharpe × sqrt(N / bars_per_year)
#   t > 2.0 = 95% confidence. t > 3.0 = credible for a searched strategy.
#
# DEFLATED SHARPE RATIO (Lopez de Prado 2014):
#   Adjusts Sharpe downward for the number of trials tested.
#   Rule of thumb: effective_Sharpe ≈ Sharpe_reported / sqrt(log(N_trials)).
#   If you tested N=100 combinations: divisor = sqrt(log(100)) ≈ 2.1.
#   A reported Sharpe of 2.0 becomes ~0.95 after deflation.
#
# SIGNAL DECAY:
#   Alpha signals lose edge over time as more traders discover them.
#   Measure: compare OOS Sharpe in first half vs. second half of test period.
#   Accelerating decay = the signal is being arbitraged away.
#
# AUTOCORRELATION IN RETURNS:
#   If daily P&L has positive autocorrelation (good day → good day),
#   the annualized Sharpe using sqrt(252×78) OVERSTATES true risk-adjusted return.
#   Fix: use Newey-West standard errors or block bootstrap for confidence intervals.
#
# OVERFITTING — THE MOST IMPORTANT CONCEPT:
#   More parameters tested = higher probability the best result is luck.
#   The correct question is not "what is the best Sharpe?"
#   The correct question is "is performance stable across a range of parameters?"
#
# MEMORY: "A high in-sample Sharpe after 100 tests means almost nothing.
#   A t-stat > 3.0 on OOS data means something real."


# ================================================================================
# ================================================================================
# SECTION 7  [FRAMEWORK]
# DECISION CHECKLIST + VISUALIZATION + INTERVIEW TALKING POINTS
# ================================================================================
# ================================================================================


# ================================================================================
# DECISION CHECKLIST
# ================================================================================
#
# Ask ALL of these before claiming the strategy is worth pursuing:
#
#   1. Did it make money?               Total Return > 0
#   2. Was risk-adjusted return strong? Sharpe > 1.5 for intraday
#   3. Was drawdown acceptable?         Max Drawdown < −10% for short backtests
#   4. Did it survive costs?            Net return significantly above zero
#   5. Did it work across tickers?      Not just AAPL — tested the universe
#   6. Did it survive OOS?              OOS Sharpe close to IS Sharpe (<40% decay)
#   7. Is turnover manageable?          Costs don't overwhelm the edge
#
# IF ALL 7 YES: continue. Add more data, more tickers, stress tests, paper trade.
# IF ANY NO:    diagnose ONE issue. Fix ONE thing. Retest.
#   Never fix multiple things simultaneously — you will not know which helped.
#
# COMMON MISTAKE:
#   Tweaking parameters until all metrics look good, then calling it done.
#   That is overfitting. A robust strategy survives without being tuned to the test.

def print_decision(metrics, label="Strategy"):
    """Print research decision based on standard checklist."""
    print(f"\n{'='*55}")
    print(f"DECISION CHECK: {label}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    good = (
        metrics.get("Total Return", -1)  > 0    and
        metrics.get("Sharpe",      -99)  > 1.0  and
        metrics.get("Max Drawdown", -1)  > -0.10
    )
    if good:
        print("\n  --> INTERESTING. Proceed to deeper out-of-sample validation.")
    else:
        print("\n  --> NOT ENOUGH. Diagnose one issue and retest.")


# ================================================================================
# VISUALIZATION
# ================================================================================
#
# EQUITY CURVE: how $1 invested grew over time.
#   Smooth upward curve with small dips = good.
#   Flat or jagged = strategy is inactive or losing.
#
# DRAWDOWN CURVE: at each point, how far equity is below its prior peak.
#   Always zero or negative. Deep drawdowns = high pain = hard to allocate capital.
#   A PM asks: "how long was your worst drawdown and how deep?"

def plot_equity(df, title="Equity Curve", col="equity"):
    """Plot cumulative equity curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[col])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity ($1 = start)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(df, title="Drawdown from Peak", col="equity"):
    """Plot rolling drawdown from peak."""
    eq = df[col]
    dd = eq / eq.cummax() - 1
    plt.figure(figsize=(10, 4))
    plt.fill_between(dd.index, dd, 0, alpha=0.4, color="red")
    plt.plot(dd.index, dd, color="red", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Drawdown from Peak")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_both(df, label="Strategy"):
    plot_equity(df,   title=f"{label} — Equity Curve")
    plot_drawdown(df, title=f"{label} — Drawdown")


# ================================================================================
# INTERVIEW TALKING POINTS — MEMORIZE AND ADAPT
# ================================================================================
#
# QUESTION: "Walk me through your research process."
#
# ANSWER:
#   "I start with a clearly stated hypothesis: intraday prices dislocate from
#    fair value and mean-revert back toward VWAP. I engineer features that
#    measure the magnitude of that dislocation — VWAP distance, RSI, and
#    return z-score.
#
#    For a single ticker I construct a rule-based signal, shift positions by
#    one bar to prevent look-ahead bias, and always measure performance net
#    of transaction costs: commission, slippage, spread, and market impact.
#
#    I then extend to a universe of names, build a composite alpha score,
#    rank stocks cross-sectionally, go long the strongest and short the weakest.
#    I apply volatility scaling at the position level and sector neutralization
#    to remove unintended factor exposures.
#
#    Finally I split the data 70/30 chronologically and report the out-of-sample
#    Sharpe. If OOS Sharpe decays less than 40% from in-sample, the signal has
#    credibility worth pursuing. I also check the t-statistic — I want t > 3.0
#    before I trust a result found via parameter search.
#
#    The goal is never one beautiful backtest.
#    The goal is robust, cost-aware, out-of-sample performance across a universe."
#
# QUESTION: "What is Kyle lambda?"
#
# ANSWER:
#   "Kyle lambda is the price impact coefficient from Kyle (1985). The model is:
#    delta_price = lambda × order_flow. Lambda measures how much price moves per
#    unit of net signed volume. High lambda means the stock is illiquid — your
#    trades move the price against you. Low lambda means you can trade large size
#    with minimal impact. In my framework I use a flat bps proxy, but with
#    estimated lambda per stock I would scale position sizes inversely to lambda
#    so that expected impact cost is equalized across all names."
#
# QUESTION: "What is the difference between mean reversion and momentum?"
#
# ANSWER:
#   "Both are cross-sectional strategies with identical portfolio construction
#    structure. The difference is the score direction. Mean reversion: a large
#    recent DROP generates a positive score — expect a bounce. Momentum: a large
#    recent GAIN generates a positive score — expect continuation. Mean reversion
#    works best intraday in range-bound, liquid names. Momentum works best at
#    daily/weekly frequency in trending regimes. In my pipeline I can switch
#    between them by changing the score function without touching the ranking,
#    weighting, cost, or validation layers."


# ================================================================================
# ================================================================================
# MAIN — PRACTICE ORDER (DO NOT SKIP STEPS)
# ================================================================================
# ================================================================================
#
# Run ONE function at a time. Read all output before moving to the next.
# Each level builds on the last. Skipping steps creates false confidence.

if __name__ == "__main__":

    print("""
================================================================================
MASTER INTRADAY ALPHA CHEAT SHEET
================================================================================
Practice in this order — do NOT skip:

  run_entry_level()           [ENTRY]        One ticker, basic signal and backtest
  run_junior_level()          [JUNIOR]       Costs, stops, Sortino, param sweep
  run_intermediate_level()    [INTERMEDIATE] Cross-sectional long/short portfolio
  run_advanced_level()        [ADVANCED]     Sector neutrality, beta proxy, Kyle lambda
  run_production_level()      [SENIOR/PROD]  IS/OOS split, statistical significance

Uncomment ONE at a time. Read the output. Move on only when you understand it.
================================================================================
""")

    # ── Uncomment ONE at a time ──────────────────────────────────────────────

    # bt, m = run_entry_level("AAPL")
    # plot_both(bt, "AAPL Entry")

    # bt, m = run_junior_level("AAPL")
    # plot_both(bt, "AAPL Junior")

    # sweep = parameter_sweep_single_ticker("AAPL")
    # print(sweep.head(10))

    # port, m = run_intermediate_level()
    # plot_both(port, "Intermediate Cross-Sectional")

    # port, stock_level, m = run_advanced_level()
    # plot_both(port, "Advanced Sector-Neutral")

    # train_port, test_port, tm, om = run_production_level()
    # plot_both(test_port, "Production OOS")

    # compare_kyle_lambda(["AAPL", "MSFT", "NVDA", "TSLA", "JPM"])

    pass
