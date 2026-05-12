"""
================================================================================
SENIOR LEVEL EXTENSIONS — INTRADAY ALPHA RESEARCH
================================================================================
PURPOSE
    Extends MASTER_INTRADAY_ALPHA_CHEATSHEET.py to production senior level.
    Each section adds one layer that separates a junior from a senior quant.

WHAT THIS FILE ADDS
    SECTION 8  [ML MODELS]           Ridge, Lasso, Random Forest, XGBoost signals
    SECTION 9  [WALK-FORWARD]        Rolling and expanding window OOS validation
    SECTION 10 [FACTOR MODELS]       Market, momentum, volatility factor exposure
    SECTION 11 [PORTFOLIO OPT]       Mean-variance, min-vol, risk parity
    SECTION 12 [TAQ PROXY]           Microstructure features from OHLCV
    SECTION 13 [EXECUTION SIM]       TWAP, VWAP, impact model, fill probability

HOW TO USE
    Run each section independently after completing the master file.
    Each section imports helpers from the master file.
    Practice order: 8 → 9 → 10 → 11 → 12 → 13

CORE MEMORY LINE
    "A senior quant does not just find alpha.
     They explain it, stress-test it, control its risks,
     and execute it without destroying the edge."

DEPENDENCIES
    pip install scikit-learn scipy yfinance pandas numpy matplotlib
    pip install xgboost          (optional — Section 8)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

from sklearn.linear_model    import Ridge, Lasso, LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_squared_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost  (optional)")

# Import helpers from the master file
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop"))
try:
    from MASTER_INTRADAY_ALPHA_CHEATSHEET import (
        download_one_ticker, download_many_tickers,
        add_features, build_signal_junior, backtest_junior,
        compute_metrics_junior, make_panel, build_composite_score,
        build_cross_sectional_positions, backtest_cross_sectional,
        add_sector, sector_neutralize_weights, plot_equity, plot_drawdown,
        SECTOR_MAP
    )
    print("Master file loaded.")
except ImportError:
    print("WARNING: Master file not found. Run from Desktop or adjust sys.path.")


# ================================================================================
# ================================================================================
# SECTION 8  [ML MODELS]
# MACHINE LEARNING ALPHA SIGNALS
# ================================================================================
# ================================================================================
#
# WHY ML OVER RULE-BASED SIGNALS?
#   Rule-based: long_rsi=25, distance>0.001 — hard thresholds, manually chosen.
#   ML: learns the OPTIMAL combination of features from data automatically.
#   ML can capture non-linear interactions that rules miss entirely.
#   Example: RSI=28 AND volume_spike=1.5 AND vwap_distance=-0.002 together
#   may be far more predictive than any single threshold rule.
#
# THE CORE ML WORKFLOW FOR ALPHA:
#   1. Build feature matrix X = all features at each bar
#   2. Build target y = forward N-bar return (what we are trying to predict)
#   3. Train model on in-sample data
#   4. Predict on out-of-sample data
#   5. Convert predictions to positions: positive pred = long, negative = short
#   6. Backtest the ML signal exactly like the rule-based signal
#
# TERM: Forward return = return earned from bar t to bar t+N.
#   y_t = (close_{t+N} - close_t) / close_t
#   This is what the model is trying to predict from features at bar t.
#   Using forward return as the target captures the PREDICTIVE relationship.
#
# CRITICAL: NEVER use future data in features.
#   All features must be computable from information available at bar t.
#   No feature should "see" what happens after bar t.
#
# MEMORY: "ML finds the optimal feature weights. Rules pick them manually."


# ================================================================================
# STEP 8.1 — BUILD ML FEATURE MATRIX AND TARGET
# ================================================================================
#
# TERM: Feature matrix X = one row per bar, one column per feature.
#   Shape: (n_bars, n_features). Each row is a trading opportunity.
#
# TERM: Target y = what we want to predict.
#   Here: forward 1-bar return. Could also be 3-bar or 6-bar for longer signals.
#
# TERM: Lookahead-safe target:
#   y_t = forward_return at bar t = return FROM bar t TO bar t+1.
#   When training at bar t, we use features from bar t and the return from t to t+1.
#   At prediction time (bar t), we only observe features — not the future return.
#
# WHY standardize features (StandardScaler)?
#   Ridge and Lasso are sensitive to feature scale.
#   vwap_distance is in % (0.001), volume_spike is in ratios (0.8 to 2.0).
#   Without scaling, high-magnitude features dominate the model unfairly.
#   Tree-based models (Random Forest, XGBoost) do NOT need scaling.

FEATURE_COLS = [
    "vwap_distance",  # how far from VWAP (primary mean-reversion signal)
    "rsi",            # momentum extreme
    "ret_zscore",     # how unusual is this bar's return
    "volume_spike",   # volume confirmation
    "reversal_3",     # short-term reversal
    "vol_20",         # current volatility regime
    "sma_distance",   # secondary fair-value deviation
]

def build_ml_dataset(df, forward_bars=1):
    """
    Build feature matrix X and target vector y from a featured DataFrame.
    Target = forward_bars-bar return (what we are predicting).
    Drops rows with NaN in either X or y.
    """
    df     = df.copy()
    cols   = [c for c in FEATURE_COLS if c in df.columns]
    df["target"] = df["ret_1"].shift(-forward_bars)   # forward return at bar t
    df = df.dropna(subset=cols + ["target"])
    X  = df[cols].values
    y  = df["target"].values
    return X, y, df


# ================================================================================
# STEP 8.2 — TIME-SERIES CROSS-VALIDATION (PURGED K-FOLD)
# ================================================================================
#
# TERM: Standard K-Fold CV — splits data into K folds, trains on K-1, tests on 1.
#   PROBLEM for time series: train set may contain future information relative
#   to the test set if folds are not strictly chronological.
#   This causes data leakage and inflated CV scores.
#
# TERM: Walk-forward CV (Purged) — strictly chronological splits.
#   Train: bars 0 to T.  Test: bars T+gap to T+gap+window.
#   Gap = embargo period. Drops bars immediately after train end to prevent
#   autocorrelation leakage between train and test returns.
#
# TERM: Embargo = a buffer of bars between train end and test start.
#   WHY: if bar T is in train and bar T+1 is in test, they share the same
#   overnight information. The embargo breaks this overlap.
#
# INTERVIEW LINE:
#   "I use walk-forward cross-validation with an embargo period. Standard
#    K-Fold is invalid for time series because it allows future information
#    into the training set via autocorrelated returns."

def walk_forward_splits(n, n_splits=5, embargo=5):
    """
    Yield (train_idx, test_idx) tuples for walk-forward CV.
    Each split expands the training window chronologically.
    embargo = bars dropped between train end and test start.
    """
    fold_size = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end  = i * fold_size
        test_start = train_end + embargo
        test_end   = min(test_start + fold_size, n)
        if test_start >= n:
            break
        yield np.arange(train_end), np.arange(test_start, test_end)


# ================================================================================
# STEP 8.3 — TRAIN AND EVALUATE ML MODELS
# ================================================================================
#
# MODELS IN ORDER OF COMPLEXITY:
#
#   Ridge Regression (L2 penalty):
#     Linear model. Shrinks all coefficients toward zero equally.
#     Best for: many correlated features, fast training, interpretable.
#     alpha parameter controls regularization strength.
#     High alpha = more shrinkage = simpler model = less overfitting.
#
#   Lasso Regression (L1 penalty):
#     Linear model. Drives some coefficients exactly to zero (feature selection).
#     Best for: sparse signal — only a few features really matter.
#     Useful when you suspect most features are noise.
#
#   Random Forest:
#     Ensemble of decision trees. Captures non-linear interactions.
#     Does NOT need feature scaling. Robust to outliers.
#     Feature importance tells you which inputs drive the signal.
#     Best for: discovering non-linear patterns in feature combinations.
#
#   XGBoost (Gradient Boosting):
#     Sequentially builds trees, each correcting prior errors.
#     State of the art for tabular data. Used widely in quant research.
#     More prone to overfitting than RF — requires careful tuning.
#     Best for: final production signal after feature selection.
#
# HOW TO INTERPRET IC (Information Coefficient):
#   IC = Spearman rank correlation between predictions and actual returns.
#   IC > 0.05 = weak but tradeable signal.
#   IC > 0.10 = good signal. IC > 0.15 = strong signal.
#   ICIR = IC mean / IC std = signal consistency (like Sharpe but for predictions).
#   ICIR > 0.5 = stable. ICIR > 1.0 = very consistent.

def information_coefficient(y_pred, y_true):
    """Spearman rank IC between predictions and realized returns."""
    from scipy.stats import spearmanr
    ic, _ = spearmanr(y_pred, y_true)
    return ic

def train_ml_models(ticker="AAPL", period="60d", interval="5m", forward_bars=1):
    """
    Train Ridge, Lasso, Random Forest, and (optionally) XGBoost on one ticker.
    Evaluate each using walk-forward IC and compare to baseline rule-based signal.
    Returns dict of {model_name: {"IC_mean", "IC_std", "ICIR", "model"}}.
    """
    print(f"\n=== ML MODEL TRAINING — {ticker} ===")
    df = download_one_ticker(ticker, period=period, interval=interval)
    df = add_features(df)
    X, y, df_clean = build_ml_dataset(df, forward_bars=forward_bars)
    n  = len(X)
    print(f"  Dataset: {n} bars, {X.shape[1]} features, forward_bars={forward_bars}")

    models = {
        "Ridge":         Ridge(alpha=1.0),
        "Lasso":         Lasso(alpha=0.0001, max_iter=5000),
        "RandomForest":  RandomForestRegressor(n_estimators=100, max_depth=4,
                                               min_samples_leaf=20, random_state=42),
        "GradBoost":     GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                   learning_rate=0.05, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.05, verbosity=0, random_state=42)

    results = {}
    scaler  = StandardScaler()

    for name, model in models.items():
        ic_scores = []
        needs_scale = name in ("Ridge", "Lasso")

        for train_idx, test_idx in walk_forward_splits(n, n_splits=5, embargo=5):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test,  y_test  = X[test_idx],  y[test_idx]

            if needs_scale:
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            ic     = information_coefficient(y_pred, y_test)
            ic_scores.append(ic)

        ic_arr  = np.array(ic_scores)
        ic_mean = ic_arr.mean()
        ic_std  = ic_arr.std()
        icir    = ic_mean / ic_std if ic_std != 0 else np.nan
        results[name] = {"IC_mean": ic_mean, "IC_std": ic_std, "ICIR": icir, "model": model}
        print(f"  {name:15s}: IC={ic_mean:.4f}  IC_std={ic_std:.4f}  ICIR={icir:.3f}")

    # Feature importance from Random Forest
    rf = results["RandomForest"]["model"]
    cols = [c for c in FEATURE_COLS if c in df_clean.columns]
    importance = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    print(f"\n  Random Forest Feature Importance:")
    for feat, imp in importance.items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:20s}: {imp:.4f}  {bar}")

    return results, df_clean


# ================================================================================
# STEP 8.4 — ML SIGNAL BACKTEST
# ================================================================================
#
# HOW TO CONVERT ML PREDICTIONS TO POSITIONS:
#   positive prediction = expect price to rise = go long   (+1)
#   negative prediction = expect price to fall = go short  (-1)
#   magnitude of prediction = confidence in the signal
#
# POSITION SIZING FROM PREDICTIONS:
#   Option 1 — Binary: sign(prediction) → +1 or -1. Simple, ignores confidence.
#   Option 2 — Scaled: clip(prediction * scale, -1, 1). Larger prediction = larger position.
#   Option 3 — Rank:   rank predictions cross-sectionally, long top 20%, short bottom 20%.
#   Professional desks typically use Option 3 in a portfolio context.
#
# WHY retrain on expanding window for production?
#   You want the model to learn from ALL available data, not just a fixed window.
#   As time passes, you add new data and retrain — this is called expanding window.
#   Walk-forward expanding = retrain at each step, test on next period.

def backtest_ml_signal(ticker="AAPL", period="60d", interval="5m",
                        forward_bars=1, model_name="RandomForest"):
    """
    Full ML signal backtest: train on first 70%, predict on last 30%.
    Compare ML signal performance to rule-based signal on same data.
    """
    print(f"\n=== ML SIGNAL BACKTEST — {ticker} / {model_name} ===")

    df = download_one_ticker(ticker, period=period, interval=interval)
    df = add_features(df)
    X, y, df_clean = build_ml_dataset(df, forward_bars=forward_bars)
    n  = len(X)
    split = int(n * 0.70)

    # Train on first 70%
    scaler  = StandardScaler()
    models  = {
        "Ridge":        Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=4,
                                              min_samples_leaf=20, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, verbosity=0, random_state=42)

    chosen = models.get(model_name, models["RandomForest"])
    X_train, y_train = X[:split], y[:split]
    X_test           = X[split:]

    if model_name == "Ridge":
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    chosen.fit(X_train, y_train)
    y_pred = chosen.predict(X_test)

    # Build OOS DataFrame with ML position
    df_oos        = df_clean.iloc[split:].copy()
    df_oos["ml_pred"]   = y_pred
    df_oos["ml_signal"] = np.sign(y_pred)

    # Backtest ML signal
    df_oos["position_lagged"] = df_oos["ml_signal"].shift(1).fillna(0)
    df_oos["gross_return"]    = df_oos["position_lagged"] * df_oos["ret_1"]
    df_oos["turnover"]        = df_oos["position_lagged"].diff().abs().fillna(0)
    df_oos["cost"]            = df_oos["turnover"] * (7 / 10000)
    df_oos["net_return"]      = df_oos["gross_return"] - df_oos["cost"]
    df_oos["equity"]          = (1 + df_oos["net_return"].fillna(0)).cumprod()

    m = compute_metrics_junior(df_oos)
    print(f"  ML Signal OOS Results ({model_name}):")
    for k, v in m.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    ic_oos = information_coefficient(y_pred, y[:split*0])  # placeholder
    ic_oos = information_coefficient(y_pred, df_oos["ret_1"].shift(-1).dropna().values[:len(y_pred)])
    print(f"    OOS IC: {ic_oos:.4f}")

    return df_oos, m


# ================================================================================
# ================================================================================
# SECTION 9  [WALK-FORWARD TESTING]
# ROLLING AND EXPANDING WINDOW VALIDATION
# ================================================================================
# ================================================================================
#
# WHY A SINGLE 70/30 SPLIT IS NOT ENOUGH:
#   One split gives one estimate of OOS performance.
#   That estimate could be good or bad purely by luck of WHICH period you tested.
#   Walk-forward runs MANY consecutive IS/OOS windows across the full history.
#   If the signal consistently earns positive Sharpe across all windows,
#   the evidence for a real alpha is much stronger.
#
# TWO WALK-FORWARD METHODS:
#
#   Rolling window:
#     Train on fixed N bars. Test on M bars. Slide the whole window forward.
#     Example: train bars 0-200, test bars 200-250. Then train 50-250, test 250-300.
#     WHY: prevents the model from being dominated by very old data.
#     Appropriate when you believe market dynamics change over time.
#
#   Expanding window:
#     Train on ALL bars from 0 to T. Test on T to T+M. Grow T forward.
#     Example: train 0-200, test 200-250. Then train 0-250, test 250-300.
#     WHY: uses maximum available data at each step.
#     Appropriate when you believe older data is still informative.
#
# WHAT TO LOOK FOR IN RESULTS:
#   Good: Sharpe > 0 in most windows. Consistency > one-time luck.
#   Bad: Sharpe positive in 2 out of 10 windows — probably noise.
#   Warning: very high Sharpe in early windows, declining later — signal decay.
#   Red flag: Sharpe improves as you add more parameters — overfitting.
#
# TERM: Sharpe stability = std of per-window Sharpe / mean of per-window Sharpe.
#   Low stability = Sharpe varies wildly across windows = unreliable signal.
#   High stability = Sharpe is consistently positive = trustworthy signal.
#
# MEMORY: "One OOS window is a data point. Ten windows are evidence."

def walk_forward_backtest(
    ticker="AAPL",
    period="60d",
    interval="5m",
    train_bars=200,
    test_bars=50,
    mode="rolling",           # "rolling" or "expanding"
    long_rsi=25,
    short_rsi=75,
    distance_filter=0.001
):
    """
    Walk-forward backtest over many IS/OOS windows.
    Returns DataFrame of per-window metrics and a full OOS equity curve.

    mode="rolling"   : fixed-size training window slides forward
    mode="expanding" : training window grows from start each step
    """
    print(f"\n=== WALK-FORWARD BACKTEST — {ticker} / mode={mode} ===")

    df = download_one_ticker(ticker, period=period, interval=interval)
    df = add_features(df)
    df = build_signal_junior(df, long_rsi=long_rsi, short_rsi=short_rsi,
                              distance_filter=distance_filter)
    df = df.reset_index(drop=True)
    n  = len(df)

    rows       = []
    oos_pieces = []
    step       = 0
    start      = 0

    while True:
        if mode == "rolling":
            train_start = start
            train_end   = start + train_bars
        else:
            train_start = 0
            train_end   = (step + 1) * test_bars + train_bars

        test_start = train_end
        test_end   = test_start + test_bars

        if test_end > n:
            break

        train_df = df.iloc[train_start:train_end]
        test_df  = df.iloc[test_start:test_end].copy()

        # Use signal already built — evaluate on test window
        bt = backtest_junior(test_df)
        m  = compute_metrics_junior(bt)
        m["window"]      = step
        m["test_start"]  = test_start
        m["test_end"]    = test_end
        m["train_size"]  = train_end - train_start
        rows.append(m)
        oos_pieces.append(bt[["net_return"]].copy())

        start += test_bars
        step  += 1

    if not rows:
        print("  Not enough data for walk-forward. Try longer period.")
        return pd.DataFrame(), pd.DataFrame()

    results  = pd.DataFrame(rows)
    oos_full = pd.concat(oos_pieces)
    oos_full["equity"] = (1 + oos_full["net_return"].fillna(0)).cumprod()

    sharpe_vals = results["Sharpe"].dropna()
    pct_pos     = (sharpe_vals > 0).mean() * 100

    print(f"\n  Windows tested  : {len(results)}")
    print(f"  Mean Sharpe     : {sharpe_vals.mean():.3f}")
    print(f"  Sharpe Std Dev  : {sharpe_vals.std():.3f}")
    print(f"  % Windows Sharpe>0: {pct_pos:.1f}%")
    print(f"\n  Per-Window Sharpe:")
    for _, row in results.iterrows():
        bar    = "█" * max(0, int(row["Sharpe"] * 5)) if not np.isnan(row["Sharpe"]) else ""
        marker = "✓" if row["Sharpe"] > 0 else "✗"
        print(f"    Window {int(row['window']):2d}: Sharpe={row['Sharpe']:6.3f}  {marker}  {bar}")

    if pct_pos >= 70:
        print("\n  --> ROBUST: signal positive in 70%+ of windows.")
    elif pct_pos >= 50:
        print("\n  --> MODERATE: positive in majority of windows. Continue testing.")
    else:
        print("\n  --> WEAK: positive in fewer than half of windows. Revisit signal.")

    return results, oos_full

def plot_walk_forward(results, oos_equity):
    """Plot per-window Sharpe bar chart and concatenated OOS equity curve."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    sharpes = results["Sharpe"].fillna(0)
    colors  = ["green" if s > 0 else "red" for s in sharpes]
    axes[0].bar(range(len(sharpes)), sharpes, color=colors, alpha=0.75)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].axhline(sharpes.mean(), color="blue", linewidth=1.5, linestyle="--",
                    label=f"Mean Sharpe = {sharpes.mean():.2f}")
    axes[0].set_title("Walk-Forward: Per-Window Sharpe Ratio")
    axes[0].set_xlabel("Window")
    axes[0].set_ylabel("Sharpe")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "equity" in oos_equity.columns:
        axes[1].plot(oos_equity.index, oos_equity["equity"], color="steelblue")
        axes[1].set_title("Concatenated Out-of-Sample Equity Curve")
        axes[1].set_xlabel("Bar")
        axes[1].set_ylabel("Equity")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ================================================================================
# ================================================================================
# SECTION 10  [FACTOR MODELS]
# MARKET, MOMENTUM, AND VOLATILITY FACTOR EXPOSURE
# ================================================================================
# ================================================================================
#
# WHY FACTOR MODELS MATTER:
#   A strategy that earns 15% annualized return sounds good.
#   But if the market also returned 15%, your strategy added zero value.
#   Factor models decompose your return into:
#     - EXPLAINED: return from known risk factors (market, size, momentum, vol)
#     - UNEXPLAINED: true alpha — the part you should be credited for
#
# THE CORE EQUATION:
#   r_strategy = alpha + beta_mkt × r_market + beta_mom × r_momentum + epsilon
#
#   alpha     : the intercept — return with all factors at zero (true skill)
#   beta_mkt  : market sensitivity. If positive, you are implicitly long the market.
#   beta_mom  : momentum sensitivity. If positive, you benefit from trending markets.
#   epsilon   : residual — unexplained return (idiosyncratic alpha)
#
# TERM: Factor loading / beta = how much your strategy moves with a given factor.
#   beta_mkt > 1.0 = you amplify market moves — not alpha, just leverage.
#   beta_mkt ≈ 0.0 = market-neutral — your returns are independent of market.
#
# TERM: Alpha (intercept) = annualized return unexplained by any factor.
#   This is the ONLY number that matters for attributing skill.
#   A strategy with 15% return but beta_mkt=1.0 and alpha=0 is just a market bet.
#   A strategy with 8% return and alpha=8% is genuine skill.
#
# FACTORS WE BUILD HERE (proxies from available data):
#   Market factor   : SPY 5-minute returns (broad equity market)
#   Momentum factor : return over past 78 bars (last 6.5 hours = one full day)
#   Volatility factor: VIX proxy from realized vol of SPY (risk-off/risk-on)
#
# REAL FUND NOTE:
#   Barra and Axioma provide hundreds of pre-built factor models.
#   Fama-French 3-factor and 5-factor models are the academic standard.
#   Here we build simplified proxies from intraday data.
#
# INTERVIEW LINE:
#   "I decompose strategy returns using a factor regression to isolate alpha.
#    I check market beta, momentum loading, and volatility exposure.
#    If alpha > 0 after factor adjustment, the signal has genuine predictive power."

def build_factor_returns(tickers, period="60d", interval="5m"):
    """
    Build factor return series from available data.
    Returns DataFrame with: market_ret, momentum_ret, vol_factor.
    """
    data = {}
    for t in (["SPY", "QQQ"] + [x for x in tickers if x not in ["SPY","QQQ"]])[:5]:
        try:
            df = download_one_ticker(t, period=period, interval=interval)
            df = add_features(df)
            data[t] = df
        except Exception:
            pass

    if "SPY" not in data:
        print("  SPY not available — cannot build factor model.")
        return pd.DataFrame()

    spy = data["SPY"][["ret_1", "vol_20"]].copy()
    spy.columns = ["market_ret", "spy_vol"]

    # Momentum factor: SPY return over past 78 bars (1 full trading session)
    spy["momentum_ret"] = spy["market_ret"].rolling(78).mean()

    # Volatility factor: above/below median realized vol (risk-on vs risk-off)
    spy["vol_factor"] = (spy["spy_vol"] - spy["spy_vol"].rolling(200).mean()) / \
                         spy["spy_vol"].rolling(200).std()

    return spy.dropna()

def run_factor_regression(strategy_returns, factor_df, label="Strategy"):
    """
    Regress strategy net returns on factor returns.
    Prints: alpha, beta per factor, R-squared, residual Sharpe.

    strategy_returns : pd.Series of net_return from backtest
    factor_df        : DataFrame from build_factor_returns()
    """
    print(f"\n=== FACTOR REGRESSION — {label} ===")

    aligned = pd.concat([strategy_returns.rename("strat"), factor_df], axis=1).dropna()
    if len(aligned) < 30:
        print("  Insufficient overlapping data for regression.")
        return {}

    Y = aligned["strat"].values
    factors = [c for c in ["market_ret", "momentum_ret", "vol_factor"] if c in aligned.columns]
    X = np.column_stack([np.ones(len(Y))] + [aligned[f].values for f in factors])

    # OLS: beta = (X'X)^{-1} X'Y
    try:
        betas, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        print("  Regression failed — singular matrix.")
        return {}

    alpha_per_bar = betas[0]
    bars_per_year = 252 * 78
    alpha_annual  = alpha_per_bar * bars_per_year * 100  # in %

    y_pred  = X @ betas
    ss_res  = np.sum((Y - y_pred) ** 2)
    ss_tot  = np.sum((Y - Y.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    resid      = Y - y_pred
    resid_sr   = resid.std()
    resid_sharpe = (resid.mean() / resid_sr) * np.sqrt(bars_per_year) if resid_sr != 0 else np.nan

    print(f"  Alpha (annualized)   : {alpha_annual:.2f}%")
    for i, f in enumerate(factors):
        print(f"  Beta ({f:15s}): {betas[i+1]:.4f}")
    print(f"  R-squared            : {r2:.4f}")
    print(f"  Residual Sharpe      : {resid_sharpe:.3f}  (alpha unexplained by factors)")

    if alpha_annual > 0 and resid_sharpe > 1.0:
        print("\n  --> REAL ALPHA: positive alpha and residual Sharpe after factor adjustment.")
    elif alpha_annual > 0:
        print("\n  --> WEAK ALPHA: positive alpha but low residual Sharpe. Needs more data.")
    else:
        print("\n  --> NO ALPHA: negative alpha after factor adjustment. Signal is factor beta.")

    return {
        "alpha_annual": alpha_annual,
        "betas": dict(zip(factors, betas[1:])),
        "r2": r2,
        "residual_sharpe": resid_sharpe
    }


# ================================================================================
# ================================================================================
# SECTION 11  [PORTFOLIO OPTIMIZATION]
# MEAN-VARIANCE, MINIMUM VOLATILITY, RISK PARITY
# ================================================================================
# ================================================================================
#
# WHY PORTFOLIO OPTIMIZATION BEYOND EQUAL WEIGHT?
#   In Section 3 we used equal weights or inverse-vol weights — naive approaches.
#   Portfolio optimization finds the MATHEMATICALLY OPTIMAL weights given:
#     - Expected returns (our alpha scores or ML predictions)
#     - Covariance matrix of returns (how stocks move together)
#     - Constraints (max weight, long-only, turnover limits)
#
# THREE APPROACHES:
#
#   1. MEAN-VARIANCE (Markowitz 1952):
#      Maximize expected return for a given level of portfolio variance.
#      Equivalently: maximize Sharpe ratio.
#      PROBLEM: extremely sensitive to estimation error in expected returns.
#      Small changes in expected returns → wildly different weights.
#      WHY it's still taught: foundational theory, all other methods extend it.
#
#   2. MINIMUM VARIANCE:
#      Ignore expected returns entirely. Find weights that minimize portfolio vol.
#      More robust than mean-variance because it only needs the covariance matrix.
#      Often outperforms mean-variance in practice because expected returns are hard to estimate.
#      Professional alternative: Black-Litterman (blends model views with market weights).
#
#   3. RISK PARITY (Equal Risk Contribution):
#      Size each position so every stock contributes EQUALLY to total portfolio risk.
#      A stock with high vol gets a smaller weight. Low vol gets larger weight.
#      WHY: prevents portfolio from being dominated by one high-vol name.
#      Used by: Bridgewater (All Weather), AQR, many macro funds.
#
# TERM: Covariance matrix = measures how stocks move together.
#   Diagonal = variance of each stock.
#   Off-diagonal = covariance (positive = move together, negative = hedge each other).
#   In practice estimated from rolling historical returns.
#
# TERM: Efficient frontier = the set of all portfolios with maximum return
#   for each level of risk. Mean-variance optimization finds the frontier.
#
# INTERVIEW LINE:
#   "I use minimum variance weighting rather than Markowitz because it avoids
#    unstable expected return estimates. For intraday, risk parity is often
#    preferred — it equalizes risk contribution across names regardless of
#    how volatile markets are that day."

def estimate_covariance(panel, method="sample", lookback=100):
    """
    Estimate return covariance matrix from panel data.
    Returns (cov_matrix, tickers) for the most recent lookback bars.
    """
    recent   = panel.groupby("datetime").last().reset_index() if "datetime" in panel.columns else panel
    pivot    = panel.pivot_table(index="datetime", columns="ticker", values="ret_1")
    pivot    = pivot.dropna(axis=0).tail(lookback)
    tickers  = pivot.columns.tolist()

    if method == "sample":
        cov = pivot.cov().values
    elif method == "shrinkage":
        # Ledoit-Wolf shrinkage: blend sample cov with scaled identity
        # Reduces estimation error by pulling extreme correlations toward zero
        sample_cov = pivot.cov().values
        n, p = pivot.shape
        mu   = np.trace(sample_cov) / p
        delta = np.linalg.norm(sample_cov - mu * np.eye(p), 'fro') ** 2
        beta  = min(delta / (n * np.linalg.norm(sample_cov, 'fro') ** 2), 1.0)
        cov   = (1 - beta) * sample_cov + beta * mu * np.eye(p)
    else:
        cov = pivot.cov().values

    return cov, tickers

def optimize_minimum_variance(cov, max_weight=0.20):
    """
    Minimum variance portfolio: minimize w'Σw subject to sum(w)=1, w >= 0.
    Returns optimal weights array.
    """
    n = cov.shape[0]

    def portfolio_variance(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0.0, max_weight)] * n
    w0          = np.ones(n) / n

    result = optimize.minimize(
        portfolio_variance, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000}
    )
    return result.x if result.success else w0

def optimize_max_sharpe(cov, expected_returns, max_weight=0.20, rf_rate=0.0):
    """
    Maximum Sharpe portfolio: maximize (w'mu - rf) / sqrt(w'Σw).
    Returns optimal weights array.
    expected_returns: array of per-bar expected returns (from alpha scores or ML).
    """
    n = len(expected_returns)

    def neg_sharpe(w):
        ret = w @ expected_returns - rf_rate
        vol = np.sqrt(w @ cov @ w)
        return -ret / vol if vol > 1e-10 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0.0, max_weight)] * n
    w0          = np.ones(n) / n

    result = optimize.minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000}
    )
    return result.x if result.success else w0

def optimize_risk_parity(cov, max_iter=500, tol=1e-8):
    """
    Risk parity (equal risk contribution) portfolio.
    Iterative algorithm: each asset contributes equally to total portfolio variance.
    No closed-form solution — solved numerically.
    """
    n = cov.shape[0]
    w = np.ones(n) / n

    for _ in range(max_iter):
        marginal_risk = cov @ w                              # ∂(portfolio vol)/∂w_i
        risk_contrib  = w * marginal_risk                    # w_i × marginal risk
        total_risk    = risk_contrib.sum()
        target        = total_risk / n                       # equal contribution target
        w_new         = w * (target / risk_contrib.clip(1e-10))
        w_new        /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new

    return w

def run_portfolio_optimization(panel, label="Portfolio"):
    """
    Compare equal weight, min-variance, max-Sharpe, and risk parity allocations.
    Prints weight tables and expected contribution metrics.
    """
    print(f"\n=== PORTFOLIO OPTIMIZATION — {label} ===")

    cov, tickers = estimate_covariance(panel, method="shrinkage")
    n = len(tickers)

    if n < 2:
        print("  Need at least 2 tickers.")
        return {}

    # Expected returns: use mean recent return per ticker as simple proxy
    pivot = panel.pivot_table(index="datetime", columns="ticker", values="ret_1")
    exp_ret = pivot.tail(100).mean().reindex(tickers).fillna(0).values

    w_equal  = np.ones(n) / n
    w_minvol = optimize_minimum_variance(cov)
    w_sharpe = optimize_max_sharpe(cov, exp_ret)
    w_rp     = optimize_risk_parity(cov)

    results = pd.DataFrame({
        "Ticker":       tickers,
        "Equal":        np.round(w_equal,  4),
        "Min-Vol":      np.round(w_minvol, 4),
        "Max-Sharpe":   np.round(w_sharpe, 4),
        "Risk-Parity":  np.round(w_rp,     4),
    }).set_index("Ticker")

    print(results.to_string())

    for name, w in [("Equal", w_equal), ("Min-Vol", w_minvol),
                    ("Max-Sharpe", w_sharpe), ("Risk-Parity", w_rp)]:
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252 * 78) * 100
        port_ret = (w @ exp_ret) * 252 * 78 * 100
        print(f"  {name:12s}: Ann.Vol={port_vol:.2f}%  Ann.ExpRet={port_ret:.4f}%")

    return results


# ================================================================================
# ================================================================================
# SECTION 12  [TAQ PROXY]
# MICROSTRUCTURE FEATURES FROM OHLCV
# ================================================================================
# ================================================================================
#
# WHAT IS TAQ DATA?
#   TAQ = Trades and Quotes. Contains every individual trade and quote tick.
#   Columns: timestamp (nanosecond), price, size, bid, ask, bid_size, ask_size.
#   Used by professional intraday researchers for execution modeling.
#   Access: NYSE TAQ (academic), Bloomberg TICK, Refinitiv Tick History.
#
# WHY WE USE OHLCV AS A PROXY:
#   yfinance provides OHLCV — no bid/ask, no individual trades.
#   We estimate microstructure features from OHLC using well-known formulas.
#   These are APPROXIMATIONS. Production requires real TAQ.
#
# MICROSTRUCTURE FEATURES WE BUILD:
#
#   Roll's Spread Estimator:
#     Bid-ask spread = 2 × sqrt(-Cov(delta_price_t, delta_price_{t-1}))
#     WHY: if bid-ask spread exists, consecutive price changes negatively correlate.
#     Buying at ask → price up. Next tick sell at bid → price down.
#     This negative autocorrelation reveals the spread.
#
#   Order Flow Imbalance (OFI):
#     Proxy: volume × sign(close - open) for each bar.
#     Positive OFI = more buyer-initiated volume = price pressure upward.
#     Predicts short-term price direction better than price alone.
#     Real OFI from TAQ: sum of (buy_vol - sell_vol) per bar.
#
#   Amihud Illiquidity Ratio:
#     illiq = |return| / volume (in dollar terms).
#     Higher ratio = more price impact per dollar traded = less liquid.
#     Useful for scaling position sizes: trade less in high-illiquidity bars.
#
#   High-Low Spread (Corwin-Schultz):
#     Estimates effective spread from high/low ratio across adjacent bars.
#     More accurate than Roll's estimator for intraday data.
#
# INTERVIEW LINE:
#   "Without TAQ data, I estimate microstructure features using OHLCV proxies —
#    Roll's spread, order flow imbalance, and Amihud illiquidity. These are
#    approximations. In production I would replace them with TAQ-derived
#    bid-ask spreads and signed order flow."

def add_microstructure_features(df):
    """
    Add microstructure proxy features to an OHLCV DataFrame.
    All features use only information available at bar close.
    """
    df = df.copy()

    # ── Order Flow Imbalance (OFI) ────────────────────────────────────────────
    # Positive = up bar with volume (buying pressure)
    # Negative = down bar with volume (selling pressure)
    df["bar_direction"]  = np.sign(df["close"] - df["open"])
    df["ofi"]            = df["volume"] * df["bar_direction"]
    df["ofi_zscore"]     = (df["ofi"] - df["ofi"].rolling(20).mean()) / \
                            df["ofi"].rolling(20).std().replace(0, np.nan)

    # ── Amihud Illiquidity Ratio ──────────────────────────────────────────────
    # |return| / dollar_volume. High = illiquid = large price impact per dollar.
    dollar_volume        = df["close"] * df["volume"]
    df["amihud"]         = df["close"].pct_change().abs() / dollar_volume.replace(0, np.nan)
    df["amihud_rank"]    = df["amihud"].rolling(50).rank(pct=True)

    # ── Roll's Bid-Ask Spread Estimator ──────────────────────────────────────
    # spread = 2 × sqrt(max(-cov(Δp_t, Δp_{t-1}), 0))
    delta_p          = df["close"].diff()
    cov_roll         = delta_p.rolling(20).cov(delta_p.shift(1))
    df["roll_spread"] = 2 * np.sqrt((-cov_roll).clip(lower=0))

    # ── Corwin-Schultz High-Low Spread Estimator ─────────────────────────────
    # More accurate spread estimate using adjacent bar high-low ratios.
    alpha     = (np.sqrt(2) - 1) / (3 - 2 * np.sqrt(2))
    beta_num  = (np.log(df["high"] / df["low"])) ** 2
    beta2_num = (np.log(df["high"].rolling(2).max() / df["low"].rolling(2).min())) ** 2
    beta_cs   = beta_num + beta_num.shift(1)
    gamma_cs  = beta2_num
    cs_spread = 2 * (np.exp(alpha * np.sqrt(beta_cs)) - 1) / (1 + np.exp(alpha * np.sqrt(beta_cs)))
    df["cs_spread"] = cs_spread.clip(lower=0)

    # ── Intrabar Volatility (High-Low Range) ─────────────────────────────────
    # Normalized range: how much did price move WITHIN this bar?
    # High intrabar vol = noisy bar = less reliable signal.
    df["hl_range"]    = (df["high"] - df["low"]) / df["close"]
    df["hl_range_z"]  = (df["hl_range"] - df["hl_range"].rolling(20).mean()) / \
                         df["hl_range"].rolling(20).std().replace(0, np.nan)

    # ── Volume Clock (Time-of-Day Effect) ─────────────────────────────────────
    # Volume relative to same time-of-day average over past N days.
    # Captures the U-shaped intraday volume pattern (high at open/close).
    if hasattr(df.index, 'time'):
        df["tod_vol_ratio"] = df["volume"] / (df.groupby(df.index.time)["volume"]
                                               .transform("mean").replace(0, np.nan))
    else:
        df["tod_vol_ratio"] = df["volume_spike"] if "volume_spike" in df.columns else 1.0

    return df.dropna()

def microstructure_signal_filter(df, max_spread_bps=10, min_ofi_z=0.5):
    """
    Filter signal entries using microstructure conditions.
    Only enter when spread is reasonable AND order flow confirms direction.

    max_spread_bps : skip bars where estimated spread > this (too expensive to trade)
    min_ofi_z      : only enter longs when OFI z-score > threshold (buying pressure)
    """
    df = df.copy()
    if "roll_spread" not in df.columns:
        df = add_microstructure_features(df)

    spread_ok    = df["roll_spread"] < (max_spread_bps / 10000)
    ofi_long_ok  = df["ofi_zscore"] > -min_ofi_z   # don't fight heavy selling
    ofi_short_ok = df["ofi_zscore"] <  min_ofi_z   # don't fight heavy buying

    if "signal" in df.columns:
        df.loc[(df["signal"] == 1)  & ~(spread_ok & ofi_long_ok),  "signal"] = 0
        df.loc[(df["signal"] == -1) & ~(spread_ok & ofi_short_ok), "signal"] = 0

    return df

def print_microstructure_summary(df, ticker=""):
    """Print summary of microstructure conditions for a ticker."""
    print(f"\n=== MICROSTRUCTURE SUMMARY — {ticker} ===")
    if "roll_spread" not in df.columns:
        df = add_microstructure_features(df)
    print(f"  Median Roll Spread  : {df['roll_spread'].median()*10000:.2f} bps")
    print(f"  Median CS Spread    : {df['cs_spread'].median()*10000:.2f} bps")
    print(f"  Median Amihud       : {df['amihud'].median():.2e}")
    print(f"  Mean OFI Z-Score    : {df['ofi_zscore'].mean():.3f}")
    print(f"  Mean HL Range       : {df['hl_range'].mean()*100:.3f}%")


# ================================================================================
# ================================================================================
# SECTION 13  [EXECUTION SIMULATION]
# TWAP, VWAP, ALMGREN-CHRISS IMPACT, LIMIT ORDER FILL PROBABILITY
# ================================================================================
# ================================================================================
#
# WHY EXECUTION MODELING MATTERS AT SENIOR LEVEL:
#   A signal might show Sharpe 2.0 in backtest.
#   But if execution is poor — bad fills, excessive impact — the live Sharpe
#   could be 0.5 or negative. The gap between backtest and live is often execution.
#
# THE FOUR EXECUTION PROBLEMS:
#   1. Timing:   When exactly do you send the order?
#   2. Sizing:   How do you break a large order into smaller pieces?
#   3. Impact:   How does your trading move the price against you?
#   4. Fill:     At what price does your order actually execute?
#
# EXECUTION STRATEGIES:
#
#   MARKET ORDER:
#     Execute immediately at current market price (ask for buys, bid for sells).
#     Fastest. Highest cost — you pay the full spread plus impact.
#     Used when: urgency is high, position is small relative to volume.
#
#   LIMIT ORDER:
#     Specify the max price you will pay (buy) or min price you will accept (sell).
#     Cheaper if filled — you provide liquidity, others pay the spread.
#     Risk: order may NOT fill if price moves away. Adverse selection.
#
#   TWAP (Time-Weighted Average Price):
#     Split total order into N equal pieces. Execute one piece per time interval.
#     Minimizes timing risk. Works well for predictable intraday volume patterns.
#     Does NOT adapt to volume — trades equally in low and high volume periods.
#
#   VWAP (Volume-Weighted Average Price):
#     Split total order proportionally to expected volume per time interval.
#     Targets the VWAP benchmark — "I want to pay the volume-weighted average."
#     Adapts to volume — trades more when volume is high (lower impact).
#     Industry standard execution benchmark for institutional orders.
#
# ALMGREN-CHRISS IMPACT MODEL (simplified):
#   Total cost = permanent impact + temporary impact + spread
#   Permanent impact = lambda_perm × total_shares (affects all future prices)
#   Temporary impact = lambda_temp × sqrt(shares_per_bar / avg_volume)
#   Optimal execution: trade faster when alpha decay rate is high,
#                      trade slower when impact is high.
#
# TERM: Adverse selection = you get filled on a limit order only when
#   the market moves against you. Why? Because others know something you don't.
#   If the price is falling and you bid, you get filled — and the price keeps falling.
#
# TERM: Arrival price = the mid-price at the time you decide to trade.
#   Implementation shortfall = actual cost vs. arrival price.
#   This is the standard metric for execution quality at institutional desks.
#
# MEMORY: "Finding alpha is 40% of the job. Executing it without losing the edge is 60%."

def simulate_twap(total_shares, n_slices, avg_price_per_slice, volatility_per_slice,
                  spread_bps=2):
    """
    Simulate TWAP execution cost.
    Splits total_shares equally across n_slices time intervals.
    Returns estimated total cost in bps.
    """
    shares_per_slice = total_shares / n_slices
    timing_risk_bps  = volatility_per_slice * np.sqrt(n_slices) * 10000
    spread_cost_bps  = spread_bps
    impact_bps       = 0.5 * spread_bps * (shares_per_slice / (avg_price_per_slice * 1000))

    total_cost_bps   = spread_cost_bps + impact_bps
    return {
        "strategy":       "TWAP",
        "slices":         n_slices,
        "per_slice":      shares_per_slice,
        "spread_bps":     spread_cost_bps,
        "impact_bps":     impact_bps,
        "timing_risk_bps": timing_risk_bps,
        "total_cost_bps": total_cost_bps
    }

def simulate_vwap(total_shares, volume_profile, avg_price, spread_bps=2):
    """
    Simulate VWAP execution cost using a volume profile.
    volume_profile: array of fraction of daily volume per time slice.
    Trades more in high-volume periods → lower impact.
    Returns estimated total cost in bps.
    """
    n_slices      = len(volume_profile)
    shares_slices = total_shares * np.array(volume_profile)

    impact_per_slice = []
    for s in shares_slices:
        daily_vol = avg_price * sum(volume_profile) * 1e6
        pct       = s / max(daily_vol, 1)
        impact_per_slice.append(0.5 * np.sqrt(pct) * 10000)

    avg_impact   = np.mean(impact_per_slice)
    total_cost   = spread_bps + avg_impact
    return {
        "strategy":       "VWAP",
        "slices":         n_slices,
        "avg_impact_bps": avg_impact,
        "spread_bps":     spread_bps,
        "total_cost_bps": total_cost
    }

def almgren_chriss_cost(total_shares, avg_volume, price, sigma,
                        lambda_perm=1e-6, lambda_temp=2e-6, T_bars=78):
    """
    Almgren-Chriss simplified execution cost model.

    Permanent impact  = lambda_perm × (total_shares / avg_volume) × price
    Temporary impact  = lambda_temp × sqrt(total_shares / (avg_volume × T_bars)) × price
    Optimal execution: balances permanent impact vs. timing risk over T_bars.

    Returns cost components in bps.
    """
    x   = total_shares / max(avg_volume, 1)
    perm_impact_bps = lambda_perm * x * price * 10000
    temp_impact_bps = lambda_temp * np.sqrt(x / T_bars) * price * 10000
    sigma_bps       = sigma * 10000

    total_bps = perm_impact_bps + temp_impact_bps
    return {
        "permanent_impact_bps": perm_impact_bps,
        "temporary_impact_bps": temp_impact_bps,
        "volatility_bps":       sigma_bps,
        "total_cost_bps":       total_bps,
        "optimal_T_bars":       int(np.sqrt(lambda_perm / lambda_temp) * total_shares / avg_volume)
    }

def limit_order_fill_probability(limit_offset_bps, volatility_bps, T_bars=6):
    """
    Estimate fill probability for a limit order placed at limit_offset_bps
    away from the current mid-price, over the next T_bars bars.

    Uses a simple Gaussian approximation: price needs to touch the limit level.
    P(fill) ≈ 2 × Φ(-|offset| / (vol × sqrt(T))) for a two-sided process.

    limit_offset_bps : distance from mid to limit in bps (positive = better price)
    volatility_bps   : per-bar price volatility in bps
    T_bars           : how many bars until order expires

    REAL NOTE:
      True fill probability depends on queue position, order book depth,
      and the adverse selection component. This is a first-order approximation.
    """
    from scipy.stats import norm
    sigma_T = volatility_bps * np.sqrt(T_bars)
    if sigma_T <= 0:
        return 0.0
    p_fill = 2 * norm.cdf(-abs(limit_offset_bps) / sigma_T)
    return min(p_fill, 1.0)

def print_execution_analysis(ticker="AAPL", period="5d", interval="5m",
                              order_shares=1000):
    """
    Full execution analysis for one ticker:
    TWAP vs VWAP cost comparison, Almgren-Chriss impact, limit fill probabilities.
    """
    print(f"\n=== EXECUTION ANALYSIS — {ticker} (order_shares={order_shares:,}) ===")

    df = download_one_ticker(ticker, period=period, interval=interval)
    df = add_features(df)

    avg_price  = df["close"].mean()
    avg_volume = df["volume"].mean()
    sigma      = df["ret_1"].std() if "ret_1" in df else df["close"].pct_change().std()

    # Typical intraday volume profile (U-shaped: high open/close, low midday)
    n_slices = 13   # approx 78 bars / 6 = 13 30-min slices
    raw_vol  = np.array([3, 2, 1.5, 1.2, 1.0, 0.9, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.5])
    vol_prof = raw_vol / raw_vol.sum()

    twap = simulate_twap(order_shares, n_slices, avg_price, sigma)
    vwap = simulate_vwap(order_shares, vol_prof, avg_price)
    ac   = almgren_chriss_cost(order_shares, avg_volume, avg_price, sigma)

    print(f"\n  ── TWAP ────────────────────────────────")
    print(f"    Spread cost   : {twap['spread_bps']:.2f} bps")
    print(f"    Impact cost   : {twap['impact_bps']:.2f} bps")
    print(f"    Total cost    : {twap['total_cost_bps']:.2f} bps")

    print(f"\n  ── VWAP ────────────────────────────────")
    print(f"    Spread cost   : {vwap['spread_bps']:.2f} bps")
    print(f"    Avg impact    : {vwap['avg_impact_bps']:.2f} bps")
    print(f"    Total cost    : {vwap['total_cost_bps']:.2f} bps")

    print(f"\n  ── Almgren-Chriss ──────────────────────")
    print(f"    Permanent impact: {ac['permanent_impact_bps']:.2f} bps")
    print(f"    Temporary impact: {ac['temporary_impact_bps']:.2f} bps")
    print(f"    Total impact    : {ac['total_cost_bps']:.2f} bps")
    print(f"    Optimal horizon : {ac['optimal_T_bars']} bars")

    print(f"\n  ── Limit Order Fill Probabilities ──────")
    vol_bps = sigma * 10000
    for offset in [2, 5, 10]:
        p = limit_order_fill_probability(offset, vol_bps, T_bars=6)
        print(f"    Limit {offset:2d} bps from mid, T=6 bars: P(fill)={p:.1%}")

    if twap["total_cost_bps"] > vwap["total_cost_bps"]:
        print(f"\n  --> VWAP preferred: lower estimated total cost.")
    else:
        print(f"\n  --> TWAP preferred: simpler, similar cost.")

    return {"twap": twap, "vwap": vwap, "almgren_chriss": ac}


# ================================================================================
# ================================================================================
# FULL SENIOR PIPELINE — COMBINE ALL SECTIONS
# ================================================================================
# ================================================================================

def run_senior_full_pipeline(ticker="AAPL", period="60d", interval="5m"):
    """
    [SENIOR] Full end-to-end senior pipeline:
    ML signal → walk-forward → factor decomposition → portfolio optimization
    → microstructure features → execution analysis.

    Run after completing the master file.
    This is the workflow a senior quant runs before allocating real capital.
    """
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "JPM", "SPY", "QQQ"]

    print("\n" + "="*70)
    print("SENIOR PIPELINE — STEP 1: ML MODEL TRAINING")
    print("="*70)
    ml_results, df_ml = train_ml_models(ticker, period=period, interval=interval)

    print("\n" + "="*70)
    print("SENIOR PIPELINE — STEP 2: WALK-FORWARD VALIDATION")
    print("="*70)
    wf_results, wf_equity = walk_forward_backtest(
        ticker, period=period, interval=interval,
        train_bars=200, test_bars=50, mode="expanding"
    )
    if len(wf_results) > 0:
        plot_walk_forward(wf_results, wf_equity)

    print("\n" + "="*70)
    print("SENIOR PIPELINE — STEP 3: FACTOR DECOMPOSITION")
    print("="*70)
    factor_df = build_factor_returns(tickers, period=period, interval=interval)
    if len(factor_df) > 0 and len(wf_equity) > 0:
        strat_ret = wf_equity["net_return"].dropna()
        run_factor_regression(strat_ret, factor_df, label=f"{ticker} Walk-Forward")

    print("\n" + "="*70)
    print("SENIOR PIPELINE — STEP 4: MICROSTRUCTURE ANALYSIS")
    print("="*70)
    df_micro = download_one_ticker(ticker, period="5d", interval=interval)
    df_micro = add_features(df_micro)
    df_micro = add_microstructure_features(df_micro)
    print_microstructure_summary(df_micro, ticker=ticker)

    print("\n" + "="*70)
    print("SENIOR PIPELINE — STEP 5: EXECUTION ANALYSIS")
    print("="*70)
    exec_results = print_execution_analysis(ticker, period="5d", interval=interval,
                                            order_shares=1000)

    print("\n" + "="*70)
    print("SENIOR PIPELINE — STEP 6: PORTFOLIO OPTIMIZATION")
    print("="*70)
    data  = download_many_tickers(tickers[:6], period="5d", interval=interval)
    panel = make_panel(data)
    opt_weights = run_portfolio_optimization(panel, label="6-Ticker Universe")

    print("\n" + "="*70)
    print("SENIOR PIPELINE COMPLETE")
    print("="*70)
    print("""
    What you have now:
      ✓ ML signal trained and evaluated (IC, ICIR)
      ✓ Walk-forward Sharpe stability across windows
      ✓ Factor decomposition — alpha vs market/momentum beta
      ✓ Microstructure features — spread, OFI, illiquidity
      ✓ Execution cost comparison — TWAP vs VWAP vs Almgren-Chriss
      ✓ Portfolio weights — min-vol, max-Sharpe, risk parity

    What still separates you from a live desk:
      ✗ Institutional data (Bloomberg, TAQ, Refinitiv)
      ✗ Live paper trading with real fill simulation
      ✗ Risk limits tied to real P&L (stop-out rules)
      ✗ Production infrastructure (C++, low-latency, monitoring)
      ✗ Compliance and regulatory review
    """)

    return ml_results, wf_results, opt_weights


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    print("""
================================================================================
SENIOR LEVEL EXTENSIONS — READY
================================================================================
Run in this order (each builds on the previous):

  Section 8  — train_ml_models("AAPL")
  Section 9  — walk_forward_backtest("AAPL", mode="expanding")
  Section 10 — run_factor_regression(returns, factor_df)
  Section 11 — run_portfolio_optimization(panel)
  Section 12 — print_microstructure_summary(df, "AAPL")
  Section 13 — print_execution_analysis("AAPL")

  OR run everything at once:
  run_senior_full_pipeline("AAPL", period="60d")

Uncomment below to start.
================================================================================
""")

    # Uncomment ONE at a time:

    # ml_results, df_ml = train_ml_models("AAPL", period="60d")

    # wf_results, wf_equity = walk_forward_backtest("AAPL", period="60d", mode="expanding")
    # plot_walk_forward(wf_results, wf_equity)

    # factor_df = build_factor_returns(["AAPL","MSFT","SPY"], period="60d")

    # df_micro = download_one_ticker("AAPL", period="5d")
    # df_micro = add_features(df_micro)
    # df_micro = add_microstructure_features(df_micro)
    # print_microstructure_summary(df_micro, "AAPL")

    # print_execution_analysis("AAPL", order_shares=1000)

    # data = download_many_tickers(["AAPL","MSFT","NVDA","SPY"], period="5d")
    # panel = make_panel(data)
    # run_portfolio_optimization(panel)

    # run_senior_full_pipeline("AAPL", period="60d")

    pass
