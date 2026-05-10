# ==============================================================
# *** RESEARCH WORKFLOW ***
#
#   Idea → Data → Features → Signal → Backtest → Metrics → Robustness
#
# ==============================================================

"""
ML Signal: Ridge Regression on Intraday Features
=================================================

Hypothesis:
    No single feature predicts returns reliably.
    But a combination of features — VWAP distance, RSI, volume,
    ORB breakout size, and recent momentum — may contain a
    combined signal that manual rules cannot find.

    Ridge Regression learns the optimal weight for each feature
    automatically from the data. The output is a continuous score:
    high score = likely positive return, low score = likely negative.

What is new here vs ORB:
    ORB  = one rule, hand-coded threshold, binary signal (+1 / -1)
    Ridge = five features, learned weights, continuous score

What is Ridge Regression:
    A linear model that finds the best weights for each feature
    while penalizing large weights (L2 regularization).
    The penalty prevents the model from overfitting to 60 days of data.

What is Walk-Forward Validation:
    Train on the first 40 days. Test on the last 20 days.
    The model never sees the test data during training.
    This is how you test honestly — as if the future is unknown.

What is IC (Information Coefficient):
    IC = correlation between the model's predicted score and
         the actual forward return that followed.
    IC > 0.05 = useful signal
    IC > 0.10 = strong signal
    IC near 0 = model has no predictive power
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


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
#
# Features are the inputs we feed the model.
# Each feature asks a different question about the current bar.
#
# Feature 1 — VWAP distance : is price cheap or expensive right now?
# Feature 2 — RSI           : how extreme is the recent move?
# Feature 3 — Volume ratio  : is institutional money active?
# Feature 4 — ORB breakout  : how far has price broken the opening range?
# Feature 5 — Momentum      : what direction has price moved recently?
# Feature 6 — Time of day   : how far into the session are we?

def add_vwap(df):
    """VWAP resets each day — intraday fair value."""
    df = df.copy()
    df["date"] = df.index.date
    typical = (df["high"] + df["low"] + df["close"]) / 3

    vwap_vals = []
    for date, group in df.groupby("date"):
        tp = typical[group.index]
        vol = group["volume"]
        running_vwap = (tp * vol).cumsum() / vol.cumsum()
        vwap_vals.append(running_vwap)

    df["vwap"] = pd.concat(vwap_vals)
    return df


def add_rsi(df, window=14):
    """RSI — measures how extreme the recent move is."""
    df = df.copy()
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_orb_features(df):
    """Opening range high and low — set in first 30 minutes."""
    df = df.copy()
    orb_high_map = {}
    orb_low_map  = {}

    for date, group in df.groupby("date"):
        opening_bars       = group.head(6)
        orb_high_map[date] = opening_bars["high"].max()
        orb_low_map[date]  = opening_bars["low"].min()

    df["orb_high"]    = df["date"].map(orb_high_map)
    df["orb_low"]     = df["date"].map(orb_low_map)
    df["bar_of_day"]  = df.groupby("date").cumcount()
    df["orb_done"]    = df["bar_of_day"] >= 6
    return df


def build_features(df):
    """
    Combine all features into one DataFrame.
    Each row = one 5-minute bar with 6 features attached.
    """
    df = df.copy()
    df = add_vwap(df)
    df = add_rsi(df)
    df = add_orb_features(df)

    # Feature 1: VWAP distance — how far is price from fair value?
    # Positive = above VWAP (expensive), Negative = below VWAP (cheap)
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

    # Feature 2: RSI — already computed above (0–100)

    # Feature 3: Volume ratio — is this bar busier than average?
    # 1.0 = average, 2.5 = 2.5x busier than average
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Feature 4: ORB breakout size — how far has price moved beyond the range?
    # Positive = broke above high, Negative = broke below low, Zero = inside range
    orb_mid = (df["orb_high"] + df["orb_low"]) / 2
    orb_range = df["orb_high"] - df["orb_low"]
    df["orb_breakout"] = (df["close"] - orb_mid) / orb_range.replace(0, np.nan)

    # Feature 5: Momentum — what has price done in the last 5 bars?
    # Positive = price rising, Negative = price falling
    df["momentum_5"] = df["close"].pct_change(5)

    # Feature 6: Time of day — normalized position in the session (0 to 1)
    # 0 = market open, 1 = market close
    df["time_of_day"] = df["bar_of_day"] / 77

    # Target: what happens over the NEXT 30 MINUTES (6 bars)?
    # 5-minute returns are dominated by noise — too hard to predict.
    # 30-minute returns capture the momentum window ORB confirmed is real.
    # shift(-6) means 6 bars ahead — we never look into the future during training.
    df["forward_return"] = df["close"].pct_change(6).shift(-6)

    return df


# ============================================================
# STEP 3 — WALK-FORWARD VALIDATION
# ============================================================
#
# The most important step for honest ML research.
#
# Wrong way (in-sample):
#   Train on all 60 days, test on same 60 days.
#   The model memorized the data. Result is fake.
#
# Right way (walk-forward):
#   Train on first 40 days. Test on last 20 days.
#   The model has never seen the test period.
#   Result is honest — the future was unknown during training.

FEATURE_COLS = [
    "vwap_distance",
    "rsi",
    "volume_ratio",
    "orb_breakout",
    "momentum_5",
    "time_of_day"
]


def walk_forward_predict(df, train_frac=0.67, ridge_alpha=1.0):
    """
    Train Ridge Regression on first 67% of data.
    Generate predictions on the remaining 33%.

    ridge_alpha = regularization strength.
    Higher = simpler model = less overfitting.
    Lower = more complex model = higher overfit risk.
    """
    df = df.copy()
    df = df.dropna(subset=FEATURE_COLS + ["forward_return"])

    n         = len(df)
    train_end = int(n * train_frac)

    train = df.iloc[:train_end]
    test  = df.iloc[train_end:]

    X_train = train[FEATURE_COLS].values
    y_train = train["forward_return"].values
    X_test  = test[FEATURE_COLS].values

    # StandardScaler: rescales each feature to mean=0, std=1
    # Required so that large-valued features (RSI = 0–100)
    # do not dominate small-valued features (VWAP distance = 0.001)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train, y_train)

    # Predictions on test set only — out-of-sample
    predictions = model.predict(X_test)

    # Conviction threshold: only trade the top 30% strongest predictions.
    # The middle 70% stays flat (signal = 0).
    threshold = np.percentile(np.abs(predictions), 70)

    result             = test.copy()
    result["ml_score"] = predictions

    # Time filter: only trade 10am–11am ET (= UTC hour 14)
    # ORB confirmed this is the peak institutional momentum window.
    # Outside this window, stay flat regardless of model score.
    in_window = result.index.hour == 14

    result["ml_signal"] = np.where(
        in_window & (predictions >  threshold),  1,
        np.where(
        in_window & (predictions < -threshold), -1, 0))

    return result, model, scaler


# ============================================================
# STEP 4 — INFORMATION COEFFICIENT (IC)
# ============================================================
#
# IC measures whether the model's predictions have any
# relationship to what actually happened.
#
# IC = correlation between predicted score and actual return
# Range: -1 to +1
#
# IC > 0.05 = useful. IC > 0.10 = strong. IC near 0 = useless.
# This is the first metric a senior quant checks on an ML signal.

def compute_ic(result):
    """Pearson correlation between ml_score and actual forward return."""
    clean = result[["ml_score", "forward_return"]].dropna()
    if len(clean) < 10:
        return np.nan
    ic, pval = pearsonr(clean["ml_score"], clean["forward_return"])
    return ic, pval


# ============================================================
# STEP 5 — BACKTEST WITH COSTS
# ============================================================

def backtest(result, commission_bps=5, slippage_bps=2):
    """Backtest the ML signal with realistic transaction costs."""
    df = result.copy()
    df["return"] = df["close"].pct_change()

    # shift(1) prevents look-ahead bias — use yesterday's signal today
    df["position_lagged"] = df["ml_signal"].shift(1).fillna(0)
    df["gross_return"]    = df["position_lagged"] * df["return"]

    total_cost_bps = commission_bps + slippage_bps
    df["turnover"]  = df["position_lagged"].diff().abs().fillna(0)
    df["cost"]      = df["turnover"] * (total_cost_bps / 10000)
    df["net_return"] = df["gross_return"] - df["cost"]
    df["equity"]    = (1 + df["net_return"].fillna(0)).cumprod()
    return df


# ============================================================
# STEP 6 — METRICS
# ============================================================

def compute_metrics(df, bars_per_year=252 * 78):
    """Five numbers + IC."""
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

    ic_result = compute_ic(df) if "ml_score" in df.columns else (np.nan, np.nan)
    ic        = ic_result[0] if ic_result is not np.nan else np.nan

    return {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Trades":       trades,
        "Sharpe":       sharpe,
        "Gross Return": gross_return,
        "Total Costs":  total_costs,
        "IC":           ic
    }


# ============================================================
# STEP 7 — FEATURE IMPORTANCE
# ============================================================
#
# Ridge Regression assigns a coefficient (weight) to each feature.
# Larger absolute coefficient = that feature matters more to the model.
# This tells you which features are driving the predictions.

def print_feature_importance(model, scaler):
    """Print which features the model weighted most heavily."""
    coefficients = model.coef_
    print("\n=== FEATURE WEIGHTS (Ridge Coefficients) ===")
    for name, coef in sorted(zip(FEATURE_COLS, coefficients),
                              key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(coef) * 500)
        direction = "+" if coef > 0 else "-"
        print(f"  {name:<20} {direction}{abs(coef):.4f}  {bar}")


# ============================================================
# STEP 8 — FULL PIPELINE FOR ONE TICKER
# ============================================================

def run_one_ticker(ticker, period="60d", interval="5m",
                   train_frac=0.67, ridge_alpha=1.0):
    """Full ML pipeline for one ticker."""
    df             = download_data(ticker, period, interval)
    df             = build_features(df)
    result, model, scaler = walk_forward_predict(df, train_frac, ridge_alpha)
    bt             = backtest(result)
    metrics        = compute_metrics(bt)
    return bt, metrics, model, scaler


# ============================================================
# STEP 9 — MULTI-TICKER
# ============================================================

def run_multi_ticker(tickers, period="60d", interval="5m",
                     train_frac=0.67, ridge_alpha=1.0):
    """Run ML signal across multiple tickers."""
    rows    = {}
    results = {}

    for ticker in tickers:
        try:
            bt, metrics, model, scaler = run_one_ticker(
                ticker, period, interval, train_frac, ridge_alpha)
            results[ticker] = (bt, model, scaler)
            rows[ticker]    = metrics
        except Exception as e:
            print(f"{ticker} failed: {e}")

    metrics_df = pd.DataFrame(rows).T
    metrics_df.index.name = "Ticker"
    return results, metrics_df


# ============================================================
# STEP 10 — PLOT
# ============================================================

def plot_equity(bt, title="ML Ridge Signal — Equity Curve"):
    """Plot equity curve for the out-of-sample test period."""
    plt.figure(figsize=(10, 5))
    plt.plot(bt.index, bt["equity"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 11 — RESEARCH DECISION
# ============================================================

def research_decision(metrics_df):
    """Go / No-Go decision based on average results."""
    avg = metrics_df.mean(numeric_only=True)
    print("\n=== AVERAGE RESULTS ===")
    for col in ["Total Return", "Max Drawdown", "Trades", "Sharpe",
                "Gross Return", "Total Costs", "IC"]:
        if col in avg:
            print(f"  {col:<15} {avg[col]:.4f}")

    ic_ok     = avg.get("IC", 0) > 0.05
    sharpe_ok = avg.get("Sharpe", 0) > 1.0
    gross_ok  = avg.get("Gross Return", 0) > 0

    print("\n=== VERDICT ===")
    print(f"  IC > 0.05      {'✓' if ic_ok     else '✗'}  ({avg.get('IC', 0):.4f})")
    print(f"  Sharpe > 1.0   {'✓' if sharpe_ok else '✗'}  ({avg.get('Sharpe', 0):.4f})")
    print(f"  Gross > 0      {'✓' if gross_ok  else '✗'}  ({avg.get('Gross Return', 0):.4f})")

    if ic_ok and sharpe_ok and gross_ok:
        print("\nDecision: Strong signal. Continue to walk-forward expansion.")
    elif gross_ok and ic_ok:
        print("\nDecision: Signal has edge. Sharpe needs improvement — tune features.")
    elif gross_ok:
        print("\nDecision: Gross positive. IC low — model predictions not reliable yet.")
    else:
        print("\nDecision: No edge found. Review features and retrain.")


# ============================================================
# MAIN — RUN THIS
# ============================================================

if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]

    print("\n=== ML RIDGE SIGNAL — ONE TICKER (AAPL) ===")
    bt, metrics, model, scaler = run_one_ticker("AAPL", period="60d", interval="5m")
    print("AAPL Metrics:", metrics)
    print_feature_importance(model, scaler)
    plot_equity(bt, title="AAPL ML Ridge Signal")

    print("\n=== ML RIDGE SIGNAL — MULTI-TICKER ===")
    results, metrics_df = run_multi_ticker(tickers, period="60d", interval="5m")
    print(metrics_df.to_string())
    research_decision(metrics_df)
