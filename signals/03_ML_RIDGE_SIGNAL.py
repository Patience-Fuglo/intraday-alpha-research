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

What is PSR (Probabilistic Sharpe Ratio):
    PSR = probability that the true Sharpe ratio is above zero,
          accounting for sample size and non-normality of returns.
    PSR > 95% = strong — result is almost certainly real
    PSR > 50% = some evidence — better than random
    PSR < 50% = noise — sample too small to confirm
    PSR <  5% = garbage — result is statistical noise

What is DSR (Deflated Sharpe Ratio):
    PSR asks: is the Sharpe above zero?
    DSR asks: is the Sharpe above SR* — the maximum Sharpe that
              chance produces when you test k strategies?

    When you test 15 strategies, by pure luck the best one
    will show Sharpe ~1.77 even with no real edge.
    DSR penalises the reported Sharpe by how many strategies
    you searched through (multiple testing correction).

    SR* formula (expected max Sharpe from k trials):
        SR* = (1-gamma) * Phi_inv(1 - 1/k) + gamma * Phi_inv(1 - 1/(k*e))
        gamma = 0.5772 (Euler-Mascheroni constant)
        k = number of independent trials tested

    DSR = PSR(SR*) — same PSR formula, benchmark = SR* not 0.
    DSR > 95% = signal beats multiple-testing benchmark — real
    DSR < 95% = result may be selection bias / data mining

    The gut-punch: our 5 tickers x 3 folds = 15 trials.
    Expected max Sharpe from 15 trials by chance: SR* ~ 1.77.
    NVDA Fold 2 best Sharpe = +1.44 < 1.77 -> DSR < 50%.
    Explained by selection bias — QuantConnect needed to beat SR*.

The Five Numbers — read in this order every run:
    1. Gross Return  — does the signal have edge before fees?
    2. Total Costs   — what is the fee gap to close?
    3. Total Return  — net result (gross minus costs)
    4. IC            — do model predictions track actual returns?
    5. PSR           — is the result statistically real?
    + DSR            — does Sharpe survive multiple testing correction?
    + Max Drawdown   — worst losing streak
    + Trades         — enough observations to trust the result?
    + Sharpe         — return per unit of risk (annualised)

Research Workflow:
    Idea → Data → Features → Signal → Backtest → Metrics → Robustness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, norm


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
    Each row = one 5-minute bar with 10 features attached.

    Original 6 features:
      vwap_distance, rsi, volume_ratio, orb_breakout, momentum_5, time_of_day

    New features added (feature engineering round 2):
      atr_ratio    — regime detector: is volatility high or low right now?
      dollar_vol   — institutional activity: volume × price (size of money moving)
      gap_size     — opening gap: how much did price jump from yesterday's close?
      return_zscore — how unusual is the current move vs recent history?
    """
    df = df.copy()
    df = add_vwap(df)
    df = add_rsi(df)
    df = add_orb_features(df)

    # ── Original features ──────────────────────────────────────────

    # Feature 1: VWAP distance — how far is price from fair value?
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

    # Feature 2: RSI — already computed above (0–100)

    # Feature 3: Volume ratio — is this bar busier than average?
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Feature 4: ORB breakout size — how far beyond the opening range?
    orb_mid            = (df["orb_high"] + df["orb_low"]) / 2
    orb_range          = df["orb_high"] - df["orb_low"]
    df["orb_breakout"] = (df["close"] - orb_mid) / orb_range.replace(0, np.nan)

    # Feature 5: Momentum — price change over last 5 bars (25 minutes)
    df["momentum_5"] = df["close"].pct_change(5)

    # Feature 6: Time of day — where in the session are we? (0 = open, 1 = close)
    df["time_of_day"] = df["bar_of_day"] / 77

    # ── New features (feature engineering round 2) ─────────────────

    # Feature 7: ATR ratio — regime detector
    # ATR = Average True Range = average size of recent bar movements
    # High ATR = volatile regime (institutions moving aggressively)
    # Low ATR  = calm regime (slow, thin market)
    # We normalize by price so AAPL and NVDA are comparable
    true_range      = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_ratio"] = true_range / df["close"]

    # Feature 8: Dollar volume — institutional activity measure
    # Volume alone is misleading: 1M shares of a $5 stock = $5M moved
    #                             1M shares of a $200 stock = $200M moved
    # Dollar volume = volume × price = actual money flowing through the bar
    # Normalized by 20-bar average so it is relative, not absolute
    dollar_vol         = df["volume"] * df["close"]
    df["dollar_vol"]   = dollar_vol / dollar_vol.rolling(20).mean()

    # Feature 9: Gap size — opening jump from yesterday's close
    # A large gap up means institutions acted overnight on news or positioning
    # Gaps often set the tone for the morning momentum window
    prev_close      = df.groupby("date")["close"].transform("first").shift(78)
    day_open        = df.groupby("date")["open"].transform("first")
    df["gap_size"]  = (day_open - prev_close) / prev_close.replace(0, np.nan)

    # Feature 10: Return z-score — how unusual is this bar's move?
    # Z-score = (current return - average return) / std of returns
    # High z-score = this bar moved much more than usual = potential signal
    # Low z-score  = normal bar, nothing unusual happening
    ret              = df["close"].pct_change()
    roll_mean        = ret.rolling(20).mean()
    roll_std         = ret.rolling(20).std().replace(0, np.nan)
    df["ret_zscore"] = (ret - roll_mean) / roll_std

    # ── Target ─────────────────────────────────────────────────────

    # Predict the next 30-minute return (6 bars ahead)
    # shift(-6) ensures no look-ahead bias during training
    df["forward_return"] = df["close"].pct_change(6).shift(-6)

    return df


# ============================================================
# STEP 3 — WALK-FORWARD VALIDATION (MULTIPLE FOLDS)
# ============================================================
#
# Single fold (what we had before):
#   Train on days 1–40. Test on days 41–60.
#   One result. Could be lucky or unlucky.
#
# Multiple folds (what we build now):
#   Fold 1: Train days 1–20,  test days 21–40
#   Fold 2: Train days 1–40,  test days 41–60
#   Fold 3: Train days 1–30,  test days 31–50
#
#   Each fold = one honest out-of-sample result.
#   Average across all folds = consistent or not.
#
# Why this matters:
#   A signal that works in one window might be coincidence.
#   A signal that works across three different windows
#   under different market conditions is showing real edge.

FEATURE_COLS = [
    "vwap_distance",
    "rsi",
    "volume_ratio",
    "orb_breakout",
    "momentum_5",
    "time_of_day",
    "atr_ratio",
    "dollar_vol",
    "gap_size",
    "ret_zscore"
]


def walk_forward_single_fold(df, train_end_frac, test_start_frac,
                              test_end_frac, ridge_alpha=1.0):
    """
    Run one walk-forward fold.

    train_end_frac  : fraction of data where training ends
    test_start_frac : fraction of data where test begins
    test_end_frac   : fraction of data where test ends

    Example: train_end=0.33, test_start=0.33, test_end=0.67
    means train on first third, test on second third.
    """
    df = df.copy()
    df = df.dropna(subset=FEATURE_COLS + ["forward_return"])

    n          = len(df)
    train_end  = int(n * train_end_frac)
    test_start = int(n * test_start_frac)
    test_end   = int(n * test_end_frac)

    train = df.iloc[:train_end]
    test  = df.iloc[test_start:test_end]

    if len(train) < 50 or len(test) < 20:
        return None

    X_train = train[FEATURE_COLS].values
    y_train = train["forward_return"].values
    X_test  = test[FEATURE_COLS].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    threshold   = np.percentile(np.abs(predictions), 70)
    in_window   = test.index.hour == 14

    result             = test.copy()
    result["ml_score"] = predictions
    result["ml_signal"] = np.where(
        in_window & (predictions >  threshold),  1,
        np.where(
        in_window & (predictions < -threshold), -1, 0))

    return result, model


def walk_forward_multi_fold(df, ridge_alpha=1.0):
    """
    Run three walk-forward folds on one ticker's data.

    Three folds test whether the signal is consistent across
    different market periods — not just one lucky window.

    Fold 1: Train days 1–20  (frac 0.00–0.33), test days 21–40 (frac 0.33–0.67)
    Fold 2: Train days 1–40  (frac 0.00–0.67), test days 41–60 (frac 0.67–1.00)
    Fold 3: Train days 1–30  (frac 0.00–0.50), test days 31–50 (frac 0.50–0.83)

    Returns a list of (fold_number, metrics_dict) tuples.
    """
    FOLDS = [
        (1, 0.00, 0.33, 0.33, 0.67),   # (fold, train_start, train_end, test_start, test_end)
        (2, 0.00, 0.67, 0.67, 1.00),
        (3, 0.00, 0.50, 0.50, 0.83),
    ]

    fold_results = []

    for fold_num, _, train_end_frac, test_start_frac, test_end_frac in FOLDS:
        out = walk_forward_single_fold(
            df, train_end_frac, test_start_frac, test_end_frac, ridge_alpha)

        if out is None:
            print(f"  Fold {fold_num}: insufficient data — skipped")
            continue

        result, model = out
        bt      = backtest(result)
        metrics = compute_metrics(bt)
        fold_results.append((fold_num, metrics))

    return fold_results


# ============================================================
# STEP 4A — PURGED WALK-FORWARD WITH EMBARGO
# ============================================================
#
# Standard walk-forward has a data leakage problem that most
# researchers miss:
#
#   The forward_return label for a training bar at time T extends
#   to time T + 6 bars. If the test period starts at T + 1, that
#   label OVERLAPS the test period. The model is trained on
#   information from the future. This is label leakage.
#
# Two fixes, applied together:
#
#   PURGE — remove the last h training bars before the test start.
#           These bars have labels that extend into the test period.
#           h = forward_return horizon in bars (6 bars = 30 min).
#
#   EMBARGO — skip the first h bars of the test set.
#             Even after purging labels, adjacent bars share
#             correlated features (VWAP, rolling windows).
#             The embargo creates a clean gap — a no-man's-land —
#             between training and test.
#
# For 5-min bars with 6-bar (30-min) forward returns:
#   embargo_bars = 6 (one forward-return horizon)
#
# Visual:
#
#   Standard:  [── TRAIN ──|─── TEST ───]
#   Purged:    [── TRAIN ──XXX|gap|── TEST ──]
#                          ^^^     ^^^
#                         purged  embargo
#
# The cost: you lose 2 × embargo_bars rows of data per fold.
# On 60 days of 5-min data (~3,000 bars), losing 12 rows is
# trivial. The integrity gain is not trivial.
#
# This is the production standard. Any fund using ML on time
# series that does NOT purge and embargo has latent leakage.

def walk_forward_single_fold_purged(df, train_end_frac, test_start_frac,
                                     test_end_frac, embargo_bars=6, ridge_alpha=1.0):
    """
    One walk-forward fold with purge and embargo applied.

    Parameters
    ----------
    embargo_bars : int
        Number of bars to purge from train end and skip at test start.
        Must equal the forward-return horizon (default 6 = 30 min).
    """
    df = df.copy().dropna(subset=FEATURE_COLS + ["forward_return"])
    n = len(df)

    train_end  = int(n * train_end_frac)
    test_start = int(n * test_start_frac)
    test_end   = int(n * test_end_frac)

    # PURGE: last embargo_bars of training have labels that overlap test
    purged_train_end = max(0, train_end - embargo_bars)
    train = df.iloc[:purged_train_end]

    # EMBARGO: skip first embargo_bars of test to break feature correlation
    embargoed_test_start = min(test_start + embargo_bars, test_end)
    test = df.iloc[embargoed_test_start:test_end]

    if len(train) < 50 or len(test) < 20:
        return None

    X_train = train[FEATURE_COLS].values
    y_train = train["forward_return"].values
    X_test  = test[FEATURE_COLS].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    threshold   = np.percentile(np.abs(predictions), 70)
    in_window   = test.index.hour == 14

    result             = test.copy()
    result["ml_score"] = predictions
    result["ml_signal"] = np.where(
        in_window & (predictions >  threshold),  1,
        np.where(
        in_window & (predictions < -threshold), -1, 0))

    return result, model


def walk_forward_purged(df, embargo_bars=6, ridge_alpha=1.0):
    """
    Three-fold walk-forward with purge and embargo.

    Drop-in replacement for walk_forward_multi_fold.
    Same fold structure; adds leakage prevention.
    Returns list of (fold_number, metrics_dict).
    """
    FOLDS = [
        (1, 0.00, 0.33, 0.33, 0.67),
        (2, 0.00, 0.67, 0.67, 1.00),
        (3, 0.00, 0.50, 0.50, 0.83),
    ]

    fold_results = []
    for fold_num, _, train_end_frac, test_start_frac, test_end_frac in FOLDS:
        out = walk_forward_single_fold_purged(
            df, train_end_frac, test_start_frac, test_end_frac,
            embargo_bars, ridge_alpha)

        if out is None:
            print(f"  Fold {fold_num}: insufficient data after purge/embargo — skipped")
            continue

        result, _ = out
        bt        = backtest(result)
        metrics   = compute_metrics(bt)
        fold_results.append((fold_num, metrics))

    return fold_results


def print_fold_results(fold_results, ticker=""):
    """Print five numbers for each fold then average — the full walk-forward picture."""
    header = f"=== WALK-FORWARD RESULTS — {ticker} ===" if ticker else "=== WALK-FORWARD RESULTS ==="
    print(f"\n{header}")

    if not fold_results:
        print("  No folds completed.")
        return

    all_metrics = []

    for fold_num, m in fold_results:
        tr   = m["Total Return"]
        dd   = m["Max Drawdown"]
        n    = m["Trades"]
        sh   = m["Sharpe"]
        gr   = m["Gross Return"]
        cost = m["Total Costs"]
        ic   = m["IC"]
        psr  = m.get("PSR", np.nan)

        tr_verdict  = "GAIN  " if tr   > 0      else "LOSS  "
        dd_verdict  = "GOOD  " if dd   > -0.10  else "DANGER"
        n_verdict   = "OK    " if n    >= 50     else "WARN  "
        sh_verdict  = "STRONG" if sh   > 1.0     else "WEAK  "
        gr_verdict  = "EDGE  " if gr   > 0       else "NO EDGE"
        ic_verdict  = "USEFUL" if ic   > 0.05    else "LOW   "
        psr_verdict = "REAL  " if (not np.isnan(psr) and psr > 0.95) else \
                      "SOME  " if (not np.isnan(psr) and psr > 0.50) else "NOISE "

        dsr     = m.get("DSR",  np.nan)
        sr_star = m.get("SR*",  np.nan)
        dsr_verdict = "REAL  " if (not np.isnan(dsr) and dsr > 0.95) else \
                      "SOME  " if (not np.isnan(dsr) and dsr > 0.50) else "NOISE "

        psr_str = f"{psr:.1%}" if not np.isnan(psr) else "  n/a "
        dsr_str = f"{dsr:.1%}" if not np.isnan(dsr) else "  n/a "
        sr_star_str = f"{sr_star:.2f}" if not np.isnan(sr_star) else "n/a"

        print(f"\n  Fold {fold_num}")
        print(f"  {'':─<56}")
        print(f"  Total Return    {tr:+.2%}     {tr_verdict}  {'✓' if tr > 0 else '✗'}")
        print(f"  Max Drawdown    {dd:+.2%}     {dd_verdict}  {'✓' if dd > -0.10 else '✗'}")
        print(f"  Trades          {n:<6.0f}       {n_verdict}  {'✓' if n >= 50 else '✗'}")
        print(f"  Sharpe          {sh:+.2f}       {sh_verdict}  {'✓' if sh > 1.0 else '✗'}")
        print(f"  Gross Return    {gr:+.2%}     {gr_verdict}  {'✓' if gr > 0 else '✗'}")
        print(f"  Total Costs     {cost:.2%}      costs       ")
        print(f"  IC              {ic:+.4f}      {ic_verdict}  {'✓' if ic > 0.05 else '✗'}")
        print(f"  PSR             {psr_str}       {psr_verdict}  {'✓' if (not np.isnan(psr) and psr > 0.95) else '✗'}")
        print(f"  DSR             {dsr_str}       {dsr_verdict}  {'✓' if (not np.isnan(dsr) and dsr > 0.95) else '✗'}  (SR*={sr_star_str})")

        all_metrics.append(m)

    if len(all_metrics) > 1:
        keys = ["Total Return", "Max Drawdown", "Trades", "Sharpe",
                "Gross Return", "Total Costs", "IC", "PSR", "DSR", "SR*"]
        avg = {k: np.mean([m[k] for m in all_metrics
                           if k in m and not np.isnan(m[k])]) for k in keys}

        folds_positive_gross = sum(1 for m in all_metrics if m["Gross Return"] > 0)
        folds_positive_ic    = sum(1 for m in all_metrics if m["IC"] > 0)
        folds_psr_real       = sum(1 for m in all_metrics
                                   if not np.isnan(m.get("PSR", np.nan)) and m["PSR"] > 0.95)
        folds_dsr_real       = sum(1 for m in all_metrics
                                   if not np.isnan(m.get("DSR", np.nan)) and m["DSR"] > 0.95)

        print(f"\n  {'':─<56}")
        print(f"  AVERAGE ACROSS {len(all_metrics)} FOLDS")
        print(f"  {'':─<56}")
        print(f"  Total Return    {avg['Total Return']:+.2%}")
        print(f"  Max Drawdown    {avg['Max Drawdown']:+.2%}")
        print(f"  Trades          {avg['Trades']:.1f}")
        print(f"  Sharpe          {avg['Sharpe']:+.2f}")
        print(f"  Gross Return    {avg['Gross Return']:+.2%}")
        print(f"  Total Costs     {avg['Total Costs']:.2%}")
        print(f"  IC              {avg['IC']:+.4f}")
        print(f"  PSR             {avg['PSR']:.1%}")
        print(f"  DSR             {avg['DSR']:.1%}    (SR* = {avg['SR*']:.2f} annualised)")
        print(f"\n  Folds gross > 0  : {folds_positive_gross} / {len(all_metrics)}")
        print(f"  Folds IC > 0     : {folds_positive_ic} / {len(all_metrics)}")
        print(f"  Folds PSR > 95%  : {folds_psr_real} / {len(all_metrics)}")
        print(f"  Folds DSR > 95%  : {folds_dsr_real} / {len(all_metrics)}")

        consistent = folds_positive_gross >= 2 and avg["IC"] > 0
        psr_avg    = avg.get("PSR", 0)
        dsr_avg    = avg.get("DSR", 0)
        sr_star    = avg.get("SR*", np.nan)
        print(f"\n  Consistency : {'CONSISTENT — edge in multiple windows' if consistent else 'INCONSISTENT — results vary by period'}")
        print(f"  PSR verdict : {'REAL — high confidence' if psr_avg > 0.95 else 'SOME EVIDENCE' if psr_avg > 0.50 else 'NOISE — sample too small, need more data'}")
        print(f"  DSR verdict : {'BEATS MULTIPLE-TESTING THRESHOLD' if dsr_avg > 0.95 else 'BELOW SR*=' + f'{sr_star:.2f}' + ' — more data needed (QuantConnect)'}")

    return all_metrics


def walk_forward_predict(df, train_frac=0.67, ridge_alpha=1.0):
    """
    Original single-fold walk-forward kept for compatibility.
    Train on first 67% of data, test on remaining 33%.
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
# STEP 4B — PROBABILISTIC SHARPE RATIO (PSR)
# ============================================================
#
# Sharpe tells you the return-per-unit-of-risk from your backtest.
# It does not tell you whether that result is real or lucky.
#
# PSR answers: given this Sharpe, this sample size, and the
# shape of the return distribution — what is the probability
# the true Sharpe ratio is above zero?
#
# Two things reduce PSR below what the Sharpe suggests:
#
#   Small sample  →  fewer observations = less confidence
#                    a Sharpe of 2.0 from 5 trades is meaningless
#                    a Sharpe of 2.0 from 500 trades is real
#
#   Fat tails     →  intraday returns are not normal
#                    more extreme moves than a bell curve predicts
#                    this inflates the Sharpe estimate artificially
#                    PSR corrects for it via kurtosis adjustment
#
# Formula (Lopez de Prado, 2014):
#
#   PSR = Φ [ (SR̂ - SR*) × √(T-1) / √(1 - γ₃×SR̂ + ((γ₄+2)/4)×SR̂²) ]
#
#   SR̂  = per-period Sharpe (mean/std, NOT annualized)
#   SR*  = benchmark Sharpe per period (0 = just beat cash)
#   T    = number of return observations
#   γ₃   = skewness of returns
#   γ₄   = excess kurtosis (pandas .kurt())
#   Φ    = standard normal CDF
#
# Thresholds:
#   PSR > 95%  = strong — result is almost certainly real
#   PSR > 50%  = better than random — some evidence
#   PSR < 50%  = noise — more likely luck than skill
#   PSR <  5%  = garbage — QuantConnect VWAP+RSI was here

def compute_psr(returns, sr_benchmark=0.0):
    """
    Probabilistic Sharpe Ratio — Lopez de Prado (2014).

    returns       : pd.Series of net returns (one observation per bar)
    sr_benchmark  : annualized benchmark Sharpe (default 0 = beat cash)

    Returns PSR as a float between 0 and 1.
    """
    returns = returns.dropna()
    n = len(returns)

    if n < 5:
        return np.nan

    sr_hat    = returns.mean() / returns.std()   # per-period (not annualized)
    skew      = returns.skew()
    exc_kurt  = returns.kurt()                   # excess kurtosis (pandas default)

    # Convert annualized benchmark to per-period
    # If benchmark = 0, this is also 0
    sr_star = sr_benchmark / np.sqrt(252 * 78)

    # Denominator: non-normality correction
    # When returns are normal: skew=0, exc_kurt=0 → denominator term = 1
    # Fat tails (exc_kurt > 0) increase the denominator → lower PSR
    denom_sq = 1 - skew * sr_hat + ((exc_kurt + 2) / 4) * sr_hat ** 2

    if denom_sq <= 0:
        return np.nan

    z   = (sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    psr = norm.cdf(z)

    return psr


# ============================================================
# STEP 4C — DEFLATED SHARPE RATIO (DSR)
# ============================================================
#
# PSR tests: is the Sharpe above zero?
# DSR tests: is the Sharpe above SR* — the expected maximum Sharpe
#            produced by chance when you run k independent backtests?
#
# This is the multiple testing correction. Every strategy you try
# that doesn't work still inflates the expected best Sharpe.
# The researcher who runs 100 strategies and picks the best will
# find a winner — even if all 100 strategies had zero true edge.
#
# SR* — expected maximum Sharpe from k trials:
#
#   SR* = (1-gamma) * Phi_inv(1 - 1/k) + gamma * Phi_inv(1 - 1/(k*e))
#
#   gamma = 0.5772 (Euler-Mascheroni constant)
#   k     = number of independent strategy trials
#   e     = 2.71828 (Euler's number)
#
# Trial count in this research:
#   5 tickers x 3 folds = 15 independent walk-forward tests
#   SR* at k=15  ~ 1.77 annualised
#   SR* at k=30  ~ 2.11 annualised (if you count ORB trials too)
#
# DSR = PSR(SR*)  — PSR formula with SR* as benchmark instead of 0.
#
# DSR > 95% = signal beats the multiple-testing threshold — real
# DSR < 50% = result is within what chance alone would produce
#
# The correct interpretation when DSR < 95%:
#   Do not discard the signal.
#   Get more data. With 500+ trades (QuantConnect), SR* becomes
#   a much easier hurdle because the Sharpe estimate is more precise.

def compute_dsr(returns, n_trials=15):
    """
    Deflated Sharpe Ratio — Lopez de Prado (2014).

    Parameters
    ----------
    returns   : pd.Series of per-bar net returns
    n_trials  : number of independent strategies / folds tested

    Returns
    -------
    dsr       : float in [0, 1] — probability Sharpe beats SR*
    sr_star   : float — expected max annualised Sharpe from n_trials
    """
    returns = returns.dropna()
    n = len(returns)
    if n < 5:
        return np.nan, np.nan

    sr_hat   = returns.mean() / returns.std()   # per-period Sharpe
    skew     = returns.skew()
    exc_kurt = returns.kurt()

    # Expected max annualised Sharpe from k independent trials
    gamma = 0.5772156649   # Euler-Mascheroni constant
    if n_trials > 1:
        sr_star_annual = (
            (1 - gamma) * norm.ppf(1 - 1 / n_trials) +
            gamma        * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        sr_star_annual = 0.0

    # Convert annualised SR* to per-period units (same as sr_hat)
    sr_star_per_period = sr_star_annual / np.sqrt(252 * 78)

    denom_sq = 1 - skew * sr_hat + ((exc_kurt + 2) / 4) * sr_hat ** 2
    if denom_sq <= 0:
        return np.nan, sr_star_annual

    z   = (sr_hat - sr_star_per_period) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    dsr = norm.cdf(z)

    return float(dsr), float(sr_star_annual)


# ============================================================
# STEP 4E — MONTE CARLO P&L SIMULATION
# ============================================================
#
# A single backtest gives you one P&L path.
# Monte Carlo asks: across thousands of possible orderings of
# your trade returns, what does the distribution of outcomes
# look like?
#
# Why this matters:
#   Your backtest Sharpe was 1.03.
#   But maybe that sequence of wins and losses happened to fall
#   in a lucky order. What if the same trades happened in a
#   different order? Would you still be up?
#
# Method — bootstrap resampling:
#   Take your N actual trade returns.
#   Sample them with replacement to build M alternative paths.
#   Each path has the same returns, in a random order.
#   Plot the distribution of final outcomes.
#
# Key outputs:
#   Median outcome     — the typical result
#   5th percentile     — what a bad-luck run looks like
#   95th percentile    — what a good-luck run looks like
#   P(ruin)            — fraction of paths that go below -20%
#
# If the 5th percentile is still positive, the edge is robust
# to bad luck. If the 5th percentile is deeply negative, the
# strategy depends on lucky trade sequencing to survive.
#
# This is also used for pre-trade sizing:
#   Size down if P(ruin) > 5% at current position size.
#
# Interview line:
#   "I bootstrapped 1,000 equity paths from the trade-level
#    returns. The 5th percentile outcome at the 60-day horizon
#    was −8.2%. That set my max position size — I would not take
#    a bet where the bad-luck outcome exceeded my drawdown limit."

def monte_carlo_pnl(returns, n_paths=1000, ruin_threshold=-0.20, seed=42):
    """
    Bootstrap P&L simulation from trade returns.

    Parameters
    ----------
    returns         : pd.Series of net returns (one per bar or per trade)
    n_paths         : number of simulated equity paths
    ruin_threshold  : drawdown level defined as ruin (default -20%)
    seed            : random seed for reproducibility

    Returns
    -------
    dict with path stats: median, p5, p95, p_ruin, paths array
    """
    rng = np.random.default_rng(seed)
    r   = np.array(returns.dropna())
    n   = len(r)

    if n < 10:
        return None

    # Bootstrap: n_paths × n matrix of randomly sampled returns
    sampled = rng.choice(r, size=(n_paths, n), replace=True)

    # Cumulative equity curves — starting at 1.0
    equity_paths = np.cumprod(1 + sampled, axis=1)

    final_equity = equity_paths[:, -1]    # terminal value of each path
    min_equity   = equity_paths.min(axis=1)  # worst point of each path

    p_ruin = np.mean(min_equity < (1 + ruin_threshold))

    return {
        "median":    float(np.median(final_equity)),
        "p5":        float(np.percentile(final_equity, 5)),
        "p95":       float(np.percentile(final_equity, 95)),
        "p_ruin":    float(p_ruin),
        "paths":     equity_paths,
        "n_paths":   n_paths,
        "n_returns": n,
    }


def print_monte_carlo(mc, ticker=""):
    """Print Monte Carlo P&L summary."""
    label = f" — {ticker}" if ticker else ""
    print(f"\n=== MONTE CARLO P&L{label} ({mc['n_paths']} paths, {mc['n_returns']} returns) ===")
    print(f"  Median outcome   : {mc['median'] - 1:+.2%}")
    print(f"  5th percentile   : {mc['p5']    - 1:+.2%}  (bad-luck scenario)")
    print(f"  95th percentile  : {mc['p95']   - 1:+.2%}  (good-luck scenario)")
    print(f"  P(ruin < -20%)   : {mc['p_ruin']:.1%}")

    width = mc['p95'] - mc['p5']
    if mc['p5'] > 1.0:
        verdict = "ROBUST — positive even in bad-luck scenario"
    elif mc['p5'] > 0.90:
        verdict = "ACCEPTABLE — small loss in worst case"
    else:
        verdict = "FRAGILE — depends on trade sequencing"

    ruin_verdict = "OK" if mc['p_ruin'] < 0.05 else "HIGH — reduce position size"
    print(f"  Range (p5–p95)   : {width:.2%}")
    print(f"  Verdict          : {verdict}")
    print(f"  Ruin risk        : {ruin_verdict}")


# ============================================================
# STEP 4F — PRE-TRADE RISK CHECKLIST
# ============================================================
#
# Before a signal goes live, it must pass a structured gate.
# Not a gut feel. Not "the backtest looked good."
# A checklist — every item must be checked against a threshold.
#
# The checklist maps directly to the five numbers:
#
#   1. Gross Return > 0         — signal has edge before costs
#   2. Net Return > 0           — edge survives all costs
#   3. IC > 0.05                — model predictions are directionally useful
#   4. PSR > 95%                — Sharpe is statistically real
#   5. DSR > 95%                — Sharpe survives multiple testing
#   6. Max Drawdown > -20%      — drawdown is within risk limit
#   7. Trades >= 50             — sample large enough to measure
#   8. P(ruin) < 5%             — Monte Carlo path risk acceptable
#
# Items 1–4 are CRITICAL — fail any of these and the signal
# does NOT go live. Items 5–8 are ADVISORY — fail means investigate
# further, not automatic rejection.
#
# Why a checklist?
#   Because the human brain pattern-matches. It will find reasons
#   to approve a signal you are emotionally attached to.
#   A checklist removes the bias. Every signal faces the same gate.
#
# Interview line:
#   "Every signal I develop passes through a ten-item pre-trade
#    checklist before I would take it to a PM. The checklist
#    includes IC, PSR, DSR, drawdown, position sizing from
#    Monte Carlo, and a minimum trade count. If the signal fails
#    any critical item, it goes back to research — not to production."

def pretrade_checklist(metrics, monte_carlo=None, label=""):
    """
    Structured pre-trade approval gate.

    Parameters
    ----------
    metrics      : dict from compute_metrics()
    monte_carlo  : dict from monte_carlo_pnl() (optional)
    label        : ticker or strategy name for display

    Returns
    -------
    go           : bool — True if all critical items pass
    """
    header = f"=== PRE-TRADE RISK CHECKLIST{' — ' + label if label else ''} ==="
    print(f"\n{header}")
    print(f"  {'':─<58}")

    checks = [
        # (label, condition, is_critical, actual_value_str)
        ("Gross Return > 0",
         metrics.get("Gross Return", 0) > 0,
         True,
         f"{metrics.get('Gross Return', 0):+.2%}"),

        ("Net Return > 0",
         metrics.get("Total Return", 0) > 0,
         True,
         f"{metrics.get('Total Return', 0):+.2%}"),

        ("IC > 0.05",
         metrics.get("IC", 0) > 0.05,
         True,
         f"{metrics.get('IC', 0):+.4f}"),

        ("PSR > 95%",
         metrics.get("PSR", 0) > 0.95,
         True,
         f"{metrics.get('PSR', 0):.1%}"),

        ("Max DD > -20%",
         metrics.get("Max Drawdown", -1) > -0.20,
         True,
         f"{metrics.get('Max Drawdown', 0):+.2%}"),

        ("DSR > 95%",
         metrics.get("DSR", 0) > 0.95,
         False,
         f"{metrics.get('DSR', 0):.1%}"),

        ("Trades >= 50",
         metrics.get("Trades", 0) >= 50,
         False,
         f"{int(metrics.get('Trades', 0))}"),
    ]

    if monte_carlo is not None:
        checks.append((
            "P(ruin) < 5%",
            monte_carlo.get("p_ruin", 1) < 0.05,
            False,
            f"{monte_carlo.get('p_ruin', 1):.1%}",
        ))

    critical_fails = 0
    advisory_fails = 0

    for name, passed, is_critical, val in checks:
        marker    = "CRITICAL" if is_critical else "advisory"
        tick      = "✓" if passed else "✗"
        flag      = "" if passed else f"  ← {'FAIL — BLOCKED' if is_critical else 'investigate'}"
        print(f"  {tick}  {name:<22} {val:>10}   [{marker}]{flag}")
        if not passed and is_critical:
            critical_fails += 1
        elif not passed:
            advisory_fails += 1

    print(f"  {'':─<58}")

    go = critical_fails == 0
    if go and advisory_fails == 0:
        verdict = "GO — all items pass. Signal approved for QuantConnect validation."
    elif go:
        verdict = f"CONDITIONAL GO — {advisory_fails} advisory item(s) need investigation."
    else:
        verdict = f"NO-GO — {critical_fails} critical item(s) failed. Return to research."

    print(f"\n  Verdict: {verdict}")
    return go


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

def compute_metrics(df, bars_per_year=252 * 78, n_trials=15):
    """Five numbers + IC + PSR + DSR."""
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

    psr            = compute_psr(returns)
    dsr, sr_star   = compute_dsr(returns, n_trials=n_trials)

    return {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Trades":       trades,
        "Sharpe":       sharpe,
        "Gross Return": gross_return,
        "Total Costs":  total_costs,
        "IC":           ic,
        "PSR":          psr,
        "DSR":          dsr,
        "SR*":          sr_star,
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
    psr_ok    = avg.get("PSR", 0) > 0.95
    psr_some  = avg.get("PSR", 0) > 0.50

    print("\n=== VERDICT ===")
    print(f"  IC > 0.05      {'✓' if ic_ok     else '✗'}  ({avg.get('IC', 0):.4f})")
    print(f"  Sharpe > 1.0   {'✓' if sharpe_ok else '✗'}  ({avg.get('Sharpe', 0):.4f})")
    print(f"  Gross > 0      {'✓' if gross_ok  else '✗'}  ({avg.get('Gross Return', 0):.4f})")
    print(f"  PSR > 95%      {'✓' if psr_ok    else '✗'}  ({avg.get('PSR', 0):.1%})")

    if ic_ok and sharpe_ok and gross_ok and psr_ok:
        print("\nDecision: Strong signal — statistically confirmed. Proceed to QuantConnect.")
    elif gross_ok and ic_ok and psr_some:
        print("\nDecision: Edge present, PSR borderline — need more observations. QuantConnect next.")
    elif gross_ok and ic_ok:
        print("\nDecision: Direction correct, sample too small to confirm. QuantConnect will resolve.")
    elif gross_ok:
        print("\nDecision: Gross positive. IC and PSR low — model predictions not reliable yet.")
    else:
        print("\nDecision: No edge found. Review features and retrain.")


# ============================================================
# MAIN — RUN THIS
# ============================================================

if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]

    # ── Walk-forward: multiple folds on AAPL ─────────────────────────
    # Three folds across different market periods.
    # This tests consistency — does the signal hold in different windows?

    print("\n=== WALK-FORWARD MULTI-FOLD — AAPL ===")
    df_aapl = download_data("AAPL", period="60d", interval="5m")
    df_aapl = build_features(df_aapl)
    fold_results_aapl = walk_forward_multi_fold(df_aapl)
    print_fold_results(fold_results_aapl, ticker="AAPL")

    # ── Walk-forward: multiple folds across all tickers ───────────────
    print("\n\n=== WALK-FORWARD MULTI-FOLD — ALL TICKERS ===")
    all_fold_averages = {}

    for ticker in tickers:
        print(f"\n  Downloading {ticker}...")
        try:
            df_t = download_data(ticker, period="60d", interval="5m")
            df_t = build_features(df_t)
            fold_res = walk_forward_multi_fold(df_t)
            fold_metrics = print_fold_results(fold_res, ticker=ticker)
            if fold_metrics:
                keys = ["Total Return", "Max Drawdown", "Trades", "Sharpe",
                        "Gross Return", "Total Costs", "IC", "PSR", "DSR", "SR*"]
                avg = {k: np.mean([m[k] for m in fold_metrics
                                   if k in m and not np.isnan(m[k])]) for k in keys}
                all_fold_averages[ticker] = avg
        except Exception as e:
            print(f"  {ticker} failed: {e}")

    if all_fold_averages:
        avg_df = pd.DataFrame(all_fold_averages).T
        avg_df.index.name = "Ticker"
        print("\n\n=== CROSS-TICKER WALK-FORWARD AVERAGE ===")
        print(avg_df[["Gross Return", "Total Costs", "Total Return",
                       "IC", "PSR", "DSR", "SR*", "Sharpe",
                       "Max Drawdown", "Trades"]].to_string())
        research_decision(avg_df)

    # ── Run 7: DSR — Deflated Sharpe Ratio ───────────────────────────
    # DSR = PSR corrected for multiple testing.
    # After 5 tickers x 3 folds = 15 trials, the expected max Sharpe
    # by chance alone is SR* ~ 1.77 (annualised).
    # DSR asks: does our best result beat that threshold?
    #
    # Read these numbers in order:
    #   Sharpe  — what did we observe?
    #   SR*     — what would chance produce across 15 trials?
    #   PSR     — is Sharpe above zero? (lenient test)
    #   DSR     — is Sharpe above SR*?  (strict test, corrects for search)
    #
    # DSR > 95% = signal is real beyond multiple testing doubt
    # DSR < 95% = result is within the range of lucky selection bias
    # The honest answer is almost always DSR < 95% on 60 days of data.
    # The prescription is always the same: get more data (QuantConnect).

    print("\n\n=== RUN 7 — DSR (DEFLATED SHARPE RATIO) ===")
    print("  Multiple testing correction across 15 trials (5 tickers x 3 folds)")
    print("  SR* = expected max Sharpe by chance at k=15 trials")
    print()

    n_trials = 15  # 5 tickers x 3 folds = 15 independent tests

    for ticker in tickers:
        print(f"  {ticker}")
        print(f"  {'':─<56}")
        try:
            df_t = download_data(ticker, period="60d", interval="5m")
            df_t = build_features(df_t)
            fold_res = walk_forward_multi_fold(df_t)

            collectors = {k: [] for k in ["Gross Return", "Total Costs", "Total Return",
                                           "IC", "PSR", "DSR", "SR*", "Sharpe"]}
            for _, m in fold_res:
                for k in collectors:
                    v = m.get(k, np.nan)
                    if not np.isnan(v):
                        collectors[k].append(v)

            def avg(k):
                return np.mean(collectors[k]) if collectors[k] else np.nan

            gr      = avg("Gross Return")
            costs   = avg("Total Costs")
            net     = avg("Total Return")
            ic      = avg("IC")
            psr     = avg("PSR")
            dsr     = avg("DSR")
            sr_star = avg("SR*")
            sharpe  = avg("Sharpe")

            def fmt_pct(v):  return f"{v:+.2%}" if not np.isnan(v) else "   n/a"
            def fmt_stat(v): return f"{v:.1%}"  if not np.isnan(v) else "   n/a"
            def fmt_f(v):    return f"{v:+.2f}" if not np.isnan(v) else "   n/a"

            ic_verdict  = "USEFUL " if (not np.isnan(ic)  and ic  > 0.05) else "LOW    "
            psr_verdict = "REAL   " if (not np.isnan(psr) and psr > 0.95) else \
                          "SOME   " if (not np.isnan(psr) and psr > 0.50) else "NOISE  "
            dsr_verdict = "BEATS SR*" if (not np.isnan(dsr) and dsr > 0.95) else \
                          "BELOW SR*"

            print(f"  1. Gross Return  {fmt_pct(gr)}    {'EDGE   ✓' if (not np.isnan(gr) and gr > 0) else 'NO EDGE ✗'}")
            print(f"  2. Total Costs   {fmt_pct(costs) if not np.isnan(costs) else '   n/a':>8}")
            print(f"  3. Net Return    {fmt_pct(net)}    {'GAIN   ✓' if (not np.isnan(net) and net > 0) else 'LOSS   ✗'}")
            print(f"  4. IC            {fmt_f(ic):>8}    {ic_verdict}{'✓' if (not np.isnan(ic) and ic > 0.05) else '✗'}")
            print(f"  5. PSR           {fmt_stat(psr):>8}    {psr_verdict}{'✓' if (not np.isnan(psr) and psr > 0.95) else '✗'}  (is Sharpe > 0?)")
            print(f"  +  DSR           {fmt_stat(dsr):>8}    {dsr_verdict}  {'✓' if (not np.isnan(dsr) and dsr > 0.95) else '✗'}  (is Sharpe > SR*={avg('SR*'):.2f}?)")
            print(f"     Sharpe        {fmt_f(sharpe):>8}")
            print()

        except Exception as e:
            print(f"    {ticker} failed: {e}")
            print()

    print("  ── DSR LESSON ──────────────────────────────────────────")
    print("  Read the five numbers in order — gross, costs, net, IC, PSR.")
    print("  DSR is the final gate: does the best result survive")
    print("  multiple testing across all 15 trials?")
    print()
    print("  NVDA best fold Sharpe +1.44 < SR* 1.77 — below the bar.")
    print("  DSR does not kill the signal. It sets the correct standard.")
    print("  QuantConnect (4.5 years, 500+ trades) is the prescription.")
    print("  ────────────────────────────────────────────────────────")

    # ── Run 8: Purged walk-forward with embargo ───────────────────────
    # Standard walk-forward has label leakage: the last training bars
    # have forward-return labels that extend into the test period.
    # Purge removes those bars. Embargo adds a clean gap between
    # train end and test start to prevent feature correlation leakage.
    # This is the production-grade validation standard.
    # Compare with the standard walk-forward — if results hold up,
    # the edge is real and not an artifact of leakage.

    print("\n\n=== RUN 8 — PURGED WALK-FORWARD WITH EMBARGO ===")
    print("  embargo_bars = 6  (one forward-return horizon = 30 min)")
    print("  Purge: remove last 6 training bars before test boundary")
    print("  Embargo: skip first 6 bars of test set")
    print()

    for ticker in ["NVDA", "MSFT"]:
        print(f"  {ticker}")
        print(f"  {'':─<56}")
        try:
            df_t = download_data(ticker, period="60d", interval="5m")
            df_t = build_features(df_t)

            std_res    = walk_forward_multi_fold(df_t)
            purged_res = walk_forward_purged(df_t, embargo_bars=6)

            def fold_avg(fold_res, key):
                vals = [m.get(key, np.nan) for _, m in fold_res
                        if not np.isnan(m.get(key, np.nan))]
                return np.mean(vals) if vals else np.nan

            for label, res in [("Standard  ", std_res), ("Purged    ", purged_res)]:
                gr  = fold_avg(res, "Gross Return")
                net = fold_avg(res, "Total Return")
                ic  = fold_avg(res, "IC")
                psr = fold_avg(res, "PSR")
                n_f = len(res)
                print(f"  {label}  Gross {gr:+.2%}  Net {net:+.2%}  "
                      f"IC {ic:+.4f}  PSR {psr:.1%}  ({n_f} folds)")

            print()
            print("  If purged results are similar to standard: leakage was not")
            print("  inflating the edge. If purged results are worse: the standard")
            print("  walk-forward was overstating performance due to label overlap.")
            print()

        except Exception as e:
            print(f"    {ticker} failed: {e}")
            print()

    # ── Run 9: Monte Carlo P&L simulation ────────────────────────────
    # Bootstraps 1,000 equity paths from trade returns.
    # Shows the distribution of outcomes — not just the one historical path.
    # The 5th percentile is the bad-luck scenario used for position sizing.

    print("\n\n=== RUN 9 — MONTE CARLO P&L SIMULATION ===")
    print("  1,000 bootstrap paths from trade-level net returns")
    print("  Each path: same returns, random order (sampling with replacement)")
    print()

    for ticker in ["NVDA", "MSFT"]:
        try:
            df_t       = download_data(ticker, period="60d", interval="5m")
            df_t       = build_features(df_t)
            result, _, _ = walk_forward_predict(df_t)
            bt         = backtest(result)

            mc = monte_carlo_pnl(bt["net_return"], n_paths=1000)
            if mc:
                print_monte_carlo(mc, ticker=ticker)
            else:
                print(f"  {ticker}: insufficient returns for simulation")

        except Exception as e:
            print(f"  {ticker} failed: {e}")

    print()
    print("  ── MONTE CARLO LESSON ────────────────────────────────────")
    print("  One backtest is one path through history.")
    print("  Monte Carlo shows the full distribution of outcomes")
    print("  from the same set of trade returns in different orders.")
    print("  Size positions so the 5th percentile stays within")
    print("  your maximum acceptable drawdown — not the median.")
    print("  ─────────────────────────────────────────────────────────")

    # ── Run 10: Pre-trade risk checklist ─────────────────────────────
    # Every signal must pass this gate before going to production.
    # Critical items block the signal. Advisory items flag for review.
    # This is the final decision point before QuantConnect validation.

    print("\n\n=== RUN 10 — PRE-TRADE RISK CHECKLIST ===")
    print("  Structured approval gate — five numbers + DSR + drawdown + Monte Carlo")
    print()

    for ticker in ["NVDA", "MSFT"]:
        try:
            df_t         = download_data(ticker, period="60d", interval="5m")
            df_t         = build_features(df_t)
            result, _, _ = walk_forward_predict(df_t)
            bt           = backtest(result)
            metrics      = compute_metrics(bt)
            mc           = monte_carlo_pnl(bt["net_return"], n_paths=1000)
            pretrade_checklist(metrics, monte_carlo=mc, label=ticker)
        except Exception as e:
            print(f"  {ticker} failed: {e}")

    print()
    print("  ── CHECKLIST LESSON ──────────────────────────────────────")
    print("  The checklist removes emotional bias from signal approval.")
    print("  Every signal faces the same gate. If it fails a critical")
    print("  item — IC, PSR, gross return, drawdown — it goes back")
    print("  to research. No exceptions. No 'but it looked good'.")
    print()
    print("  On 60 days of data, PSR and DSR will almost always fail.")
    print("  That is expected. The checklist tells you exactly what")
    print("  is missing — not that the signal is dead.")
    print("  ─────────────────────────────────────────────────────────")
