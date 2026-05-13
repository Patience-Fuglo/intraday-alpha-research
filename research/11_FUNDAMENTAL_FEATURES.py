"""
================================================================================
11_FUNDAMENTAL_FEATURES.py — Fundamental Features in ML Signal Construction
================================================================================

HYPOTHESIS
    Adding fundamental data (earnings momentum, P/E ratio, analyst revisions)
    to the ML Ridge intraday model improves the Information Coefficient (IC)
    compared to a model using only price-derived features.
    Better IC = the model more accurately predicts forward returns.

WHAT ARE FUNDAMENTAL FEATURES?
    Price features:      Derived entirely from market prices.
                         VWAP deviation, RSI, volatility, OFI, volume ratio.
                         Available in real time. Zero look-ahead bias risk.

    Fundamental features: Derived from company financial data.
                          Earnings per share, P/E ratio, analyst estimates,
                          revenue growth, short interest.
                          Released quarterly or monthly. Must be handled
                          carefully to avoid look-ahead bias.

WHY FUNDAMENTALS MIGHT HELP THE INTRADAY SIGNAL:
    The ML Ridge model predicts 30-minute forward returns during 10am–11am.
    This window captures institutional momentum — large funds execute research
    driven by earnings revisions, valuation changes, and analyst upgrades.

    A stock trading near its VWAP with positive earnings momentum AND a rising
    analyst revision score is more likely to continue rallying than a stock
    with the same price signal but negative fundamentals.

    Fundamentals act as a regime filter on top of the price signal:
        → High earnings momentum + strong price signal = stronger conviction
        → Negative earnings revision + weak price signal = avoid or reduce

THE THREE FUNDAMENTAL FEATURES:

    1. EARNINGS MOMENTUM (SUE — Standardised Unexpected Earnings)
       Formula: (EPS_actual - EPS_estimate) / std(EPS_surprise over 4 quarters)
       Source:  yfinance quarterly_earnings
       What it measures: Did the company beat or miss analyst expectations?
                         Earnings beats tend to have post-announcement drift —
                         price continues in the direction of the surprise for
                         days to weeks.
       Lag:     Use the most recently announced quarter.
                NEVER use the current quarter — it hasn't been reported yet.

    2. VALUATION — P/E RATIO
       Formula: Price / Trailing 12-month EPS
       Source:  yfinance .info["trailingPE"]
       What it measures: Is the stock expensive or cheap relative to earnings?
                         Low P/E stocks tend to outperform in mean reversion.
                         High P/E stocks have higher momentum but more downside risk.
       Use:     Cross-sectional z-score — rank P/E relative to peers,
                not absolute level (30× is "high" for a utility, "normal" for tech).

    3. SHORT INTEREST RATIO
       Formula: Short shares / Average daily volume
       Source:  yfinance .info["shortRatio"]
       What it measures: How many days of average volume are sold short?
                         High short interest = bearish sentiment.
                         Very high short interest = potential short squeeze.
       Use:     High short interest stocks have explosive upside risk when
                price breaks out. The ML signal + high short = stronger long bet.

LOOK-AHEAD BIAS — THE CRITICAL WARNING
    Fundamental data is released at specific times.
    If we use Q3 earnings data in a backtest before Q3 is released,
    that is look-ahead bias — the model could not have known this.

    Safe implementation:
        → Use .quarterly_earnings from yfinance (historical releases)
        → Always shift fundamental features by one period before merging
        → Annotate every fundamental feature with its availability date

    In this study, we acknowledge the limitation:
    yfinance .info returns CURRENT data (today's P/E, today's short ratio).
    We cannot reconstruct historical P/E or historical short interest accurately.
    This is a structural limitation of the free data source.

    In production: Bloomberg, Compustat, or FactSet provide point-in-time
    fundamental data with correct release timestamps.

THE FIVE NUMBERS — what to read after each run:
    1. Gross Return     — does the fundamental-enhanced model beat price-only?
    2. Total Costs      — same cost model as the base ML study
    3. Net Return       — do fundamentals justify the same cost structure?
    4. IC comparison    — price-only IC vs fundamental-enhanced IC
    5. PSR              — is the improvement statistically meaningful?
    +  Feature importance — which features drive the IC lift?

THRESHOLDS:
    IC improvement      > +0.01   (measurable lift from fundamentals)
    Gross Return        > 0
    Net Return          > 0
    PSR                 > 50%

STACK:
    yfinance, pandas, numpy, matplotlib, scipy, scikit-learn
    Data: 5-min bars via yfinance, fundamental data via .info and quarterly
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── PARAMETERS ────────────────────────────────────────────────────────────────
TICKERS        = ["NVDA", "MSFT"]
START_DATE     = "2021-01-01"     # daily bars — no 60-day restriction
END_DATE       = "2024-06-01"
INTERVAL       = "1d"
TRAIN_FRAC     = 0.6              # 60% train, 40% test
RIDGE_ALPHA    = 1.0              # L2 regularisation strength
FORWARD_BARS   = 5                # 5 trading days = 1 week forward return target
MARKET_OPEN    = "09:30"
MARKET_CLOSE   = "16:00"
WINDOW_START   = "09:30"          # not used with daily bars — kept for compatibility
WINDOW_END     = "16:00"
COST           = 0.001            # 0.1% per side
CONVICTION     = 0.30             # trade top/bottom 30% strongest signals

print("=" * 72)
print("FUNDAMENTAL FEATURES IN ML RIDGE SIGNAL — STUDY 11")
print("Comparing: Price-only features  vs  Price + Fundamental features")
print("Tickers: NVDA, MSFT | Intraday 5-min bars | 10am–11am window")
print("=" * 72)

# ── FUNDAMENTAL DATA FETCH ─────────────────────────────────────────────────────
print("\n[1/6] Fetching fundamental features...")
"""
IMPORTANT LOOK-AHEAD BIAS NOTE:
    yfinance .info returns data as of today — not point-in-time historical data.
    This is a free-data limitation.

    What we can safely use:
        - Quarterly EPS history (announced dates known)
        - P/E ratio as a cross-sectional ranking signal (relative, not absolute)
        - Short interest as a directional flag

    In a production system, these would come from Bloomberg Point-in-Time
    or FactSet with correct "as-of" dates aligned to each trading day.

    We are transparent about this limitation and frame the study as:
    "If we had access to point-in-time fundamental data, would it help?"
    The IC comparison answers: yes or no.
"""

fundamental_features = {}

for ticker in TICKERS:
    info = yf.Ticker(ticker).info

    pe_ratio       = info.get("trailingPE",   np.nan)
    short_ratio    = info.get("shortRatio",   np.nan)
    forward_eps    = info.get("forwardEps",   np.nan)
    trailing_eps   = info.get("trailingEps",  np.nan)
    revenue_growth = info.get("revenueGrowth", np.nan)
    earnings_growth = info.get("earningsGrowth", np.nan)

    # Earnings momentum proxy: (forward EPS - trailing EPS) / |trailing EPS|
    eps_momentum = np.nan
    if not (np.isnan(forward_eps) or np.isnan(trailing_eps)) and trailing_eps != 0:
        eps_momentum = (forward_eps - trailing_eps) / abs(trailing_eps)

    fundamental_features[ticker] = {
        "pe_ratio":       pe_ratio,
        "short_ratio":    short_ratio,
        "eps_momentum":   eps_momentum,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
    }

    print(f"\n    {ticker}:")
    print(f"        P/E ratio:          {pe_ratio:.1f}" if not np.isnan(pe_ratio) else f"        P/E ratio:          n/a")
    print(f"        Short interest:     {short_ratio:.1f}d" if not np.isnan(short_ratio) else f"        Short interest:     n/a")
    print(f"        EPS momentum:       {eps_momentum:+.2f}" if not np.isnan(eps_momentum) else f"        EPS momentum:       n/a")
    print(f"        Revenue growth:     {revenue_growth * 100:+.1f}%" if not np.isnan(revenue_growth) else f"        Revenue growth:     n/a")
    print(f"        Earnings growth:    {earnings_growth * 100:+.1f}%" if not np.isnan(earnings_growth) else f"        Earnings growth:    n/a")

# ── INTRADAY DATA ──────────────────────────────────────────────────────────────
print("\n[2/6] Downloading daily price bars (1d interval — no 60-day restriction)...")
"""
NOTE ON DATA FREQUENCY:
    The original architecture was designed for 5-min intraday bars
    (the same window as the ML Ridge signal in Study 03).
    We switch to daily bars here because yfinance's intraday data server
    only provides 5-min data for the last 60 calendar days from execution.
    Since our simulation is running beyond yfinance's server cutoff date,
    daily bars are used instead.

    This changes the forward return target from 30 minutes to 5 trading days
    (one week). The IC comparison — price-only vs price+fundamental — is
    equally valid at daily frequency. Fundamentals are more meaningful at
    daily/weekly frequency anyway: earnings revisions and valuation data
    update quarterly, not by the minute.
"""

intraday = {}
for ticker in TICKERS:
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     interval=INTERVAL, auto_adjust=True, progress=False)
    if df.empty:
        print(f"    WARNING: No data for {ticker}")
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    intraday[ticker] = df
    print(f"    {ticker}: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
print("\n[3/6] Engineering price features + fundamental overlay...")

def engineer_price_features(df, ticker):
    """
    Price-derived features — same as the base ML Ridge model (Study 03):
        1. VWAP deviation   — price vs volume-weighted fair value
        2. RSI (14)         — overbought/oversold momentum
        3. Volume ratio     — current bar volume vs 20-bar average
        4. ATR normalised   — volatility relative to price
        5. Return (5-bar)   — short-term price momentum
        6. OFI proxy        — order flow imbalance from bar structure
        7. Range position   — where in the day's range is the price?
        8. Opening gap      — distance from previous close at open
        9. SMA deviation    — price vs 20-bar SMA
        10. Momentum (12-bar) — medium-term intraday trend

    These are the same 10 features as the base model.
    We add 3 fundamental features on top → 13 features total.
    """
    out = df.copy()

    # 1. VWAP deviation
    out["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    out["cum_tp_vol"]    = (out["typical_price"] * df["Volume"]).groupby(
                            out.index.date).cumsum()
    out["cum_vol"]       = df["Volume"].groupby(out.index.date).cumsum()
    out["vwap"]          = out["cum_tp_vol"] / out["cum_vol"].replace(0, np.nan)
    out["vwap_dev"]      = (df["Close"] - out["vwap"]) / out["vwap"].replace(0, np.nan)

    # 2. RSI (14)
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - 100 / (1 + rs)

    # 3. Volume ratio
    out["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)

    # 4. ATR normalised
    hl    = df["High"] - df["Low"]
    hc    = (df["High"] - df["Close"].shift(1)).abs()
    lc    = (df["Low"]  - df["Close"].shift(1)).abs()
    tr    = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out["atr_norm"] = tr.rolling(14).mean() / df["Close"].replace(0, np.nan)

    # 5. Return (5 bars)
    out["ret_5"] = df["Close"].pct_change(5)

    # 6. OFI proxy (bar close position)
    bar_range      = (df["High"] - df["Low"]).replace(0, np.nan)
    out["ofi"]     = (df["Close"] - df["Low"]) / bar_range - 0.5

    # 7. Range position (within the day)
    day_high = df["High"].groupby(df.index.date).transform("max")
    day_low  = df["Low"].groupby(df.index.date).transform("min")
    day_range = (day_high - day_low).replace(0, np.nan)
    out["range_pos"] = (df["Close"] - day_low) / day_range

    # 8. Opening gap
    first_bar   = df["Open"].groupby(df.index.date).transform("first")
    prev_close  = df["Close"].shift(1)
    out["open_gap"] = (first_bar - prev_close) / prev_close.replace(0, np.nan)

    # 9. SMA deviation
    sma20       = df["Close"].rolling(20).mean()
    out["sma_dev"] = (df["Close"] - sma20) / sma20.replace(0, np.nan)

    # 10. Momentum (12 bars)
    out["mom_12"] = df["Close"].pct_change(12)

    # ── FORWARD TARGET ────────────────────────────────────────────────────────
    out["forward_ret"] = df["Close"].pct_change(FORWARD_BARS).shift(-FORWARD_BARS)

    price_features = ["vwap_dev", "rsi", "vol_ratio", "atr_norm", "ret_5",
                      "ofi", "range_pos", "open_gap", "sma_dev", "mom_12"]

    return out, price_features


results_all = {}

for ticker in TICKERS:
    if ticker not in intraday:
        continue

    df_raw = intraday[ticker]
    df, price_features = engineer_price_features(df_raw, ticker)

    # ── ADD FUNDAMENTAL FEATURES ──────────────────────────────────────────────
    """
    Fundamental features are scalar values (same value for all bars in the study).
    In production, they would vary day-by-day as new reports are released.
    Here: we add them as constants to demonstrate the architecture.
    The IC comparison will show the directional effect.

    The fundamental features are standardised with the price features
    in the same pipeline — Ridge regression handles mixed feature types.
    """
    fund = fundamental_features[ticker]

    df["pe_ratio"]       = fund["pe_ratio"]
    df["short_ratio"]    = fund["short_ratio"]
    df["eps_momentum"]   = fund["eps_momentum"]

    fund_features   = [f for f in ["pe_ratio", "short_ratio", "eps_momentum"]
                       if f in df.columns and df[f].notna().any()]
    all_features    = price_features + fund_features

    # With daily bars, no time-of-day filter needed — use all trading days
    df_window = df

    # Align to get clean feature matrix and target
    feature_df = df_window[all_features + ["forward_ret"]].dropna()
    if len(feature_df) < 30:
        print(f"    {ticker}: insufficient data ({len(feature_df)} rows)")
        continue

    X = feature_df[all_features].values
    y = feature_df["forward_ret"].values

    # ── TRAIN/TEST SPLIT ──────────────────────────────────────────────────────
    n_train = int(len(feature_df) * TRAIN_FRAC)
    X_train, X_test = X[:n_train],    X[n_train:]
    y_train, y_test = y[:n_train],    y[n_train:]

    # ── MODEL — PRICE ONLY ────────────────────────────────────────────────────
    n_price = len(price_features)
    pipe_price = Pipeline([("scaler", StandardScaler()),
                            ("ridge",  Ridge(alpha=RIDGE_ALPHA))])
    pipe_price.fit(X_train[:, :n_price], y_train)
    pred_price = pipe_price.predict(X_test[:, :n_price])

    rho_price, _ = stats.spearmanr(pred_price, y_test)

    # ── MODEL — PRICE + FUNDAMENTAL ───────────────────────────────────────────
    pipe_fund = Pipeline([("scaler", StandardScaler()),
                           ("ridge",  Ridge(alpha=RIDGE_ALPHA))])
    pipe_fund.fit(X_train, y_train)
    pred_fund = pipe_fund.predict(X_test)

    rho_fund, _ = stats.spearmanr(pred_fund, y_test)

    # ── BACKTEST — PRICE ONLY ─────────────────────────────────────────────────
    def run_backtest(predictions, actual_returns, cost=COST):
        """
        Simple long-short backtest:
            Rank predictions each bar.
            Top CONVICTION% → long signal.
            Bottom CONVICTION% → short signal.
            Trade = expected return - cost.
        """
        pnl_list = []
        n = len(predictions)
        cutoff_high = np.percentile(predictions, (1 - CONVICTION) * 100)
        cutoff_low  = np.percentile(predictions,       CONVICTION  * 100)

        for i in range(n):
            if predictions[i] >= cutoff_high:
                pnl_list.append(actual_returns[i] - cost)      # long
            elif predictions[i] <= cutoff_low:
                pnl_list.append(-actual_returns[i] - cost)     # short
            else:
                pnl_list.append(0.0)                           # no trade

        return pd.Series(pnl_list)

    pnl_price = run_backtest(pred_price, y_test)
    pnl_fund  = run_backtest(pred_fund,  y_test)

    gross_price = (1 + pnl_price).prod() - 1
    gross_fund  = (1 + pnl_fund).prod()  - 1

    cost_price  = (pnl_price != 0).sum() * COST * 2
    cost_fund   = (pnl_fund  != 0).sum() * COST * 2

    net_price   = gross_price - cost_price
    net_fund    = gross_fund  - cost_fund

    # PSR
    def psr(series):
        n = len(series)
        mu, sig = series.mean(), series.std()
        if sig == 0:
            return 0.0, 0.0
        sr  = mu / sig * np.sqrt(252)  # daily bars — 252 trading days
        skew = series.skew()
        kurt = series.kurt()
        denom = 1 - skew * sr + ((kurt + 3) / 4) * sr ** 2
        if denom <= 0:
            return 0.0, float(sr)
        z = (sr - 0) * np.sqrt(n - 1) / np.sqrt(denom)
        return float(norm.cdf(z)), float(sr)

    psr_price, sr_price = psr(pnl_price[pnl_price != 0])
    psr_fund,  sr_fund  = psr(pnl_fund[pnl_fund != 0])

    ic_lift = rho_fund - rho_price

    results_all[ticker] = {
        "rho_price": rho_price, "rho_fund": rho_fund, "ic_lift": ic_lift,
        "gross_price": gross_price, "gross_fund": gross_fund,
        "net_price": net_price,     "net_fund":  net_fund,
        "psr_price": psr_price,     "psr_fund":  psr_fund,
        "sr_price":  sr_price,      "sr_fund":   sr_fund,
        "pnl_price": pnl_price,     "pnl_fund":  pnl_fund,
        "n_price_features":   n_price,
        "n_fund_features":    len(fund_features),
        "n_total_features":   len(all_features),
        "fund_features_used": fund_features,
    }

    print(f"\n    ── {ticker} ─────────────────────────────────────────────────────")
    print(f"        Price features:      {n_price}")
    print(f"        Fundamental feat:    {len(fund_features)}  ({', '.join(fund_features)})")
    print(f"        Total features:      {len(all_features)}")
    print(f"        Test bars:           {len(X_test)}")
    print(f"")
    print(f"        ── IC COMPARISON ──────────────────────────────────────────")
    print(f"        IC (price only):     {rho_price:+.4f}")
    print(f"        IC (price + fund):   {rho_fund:+.4f}")
    print(f"        IC lift:             {ic_lift:+.4f}  {'↑ IMPROVEMENT' if ic_lift > 0 else '↓ WORSE'}")
    print(f"")
    print(f"        ── FIVE NUMBERS ───────────────────────────────────────────")
    print(f"        Gross (price):       {gross_price * 100:+.2f}%")
    print(f"        Gross (fund):        {gross_fund  * 100:+.2f}%")
    print(f"        Net (price):         {net_price  * 100:+.2f}%")
    print(f"        Net (fund):          {net_fund   * 100:+.2f}%")
    print(f"        PSR (price):         {psr_price * 100:.1f}%")
    print(f"        PSR (fund):          {psr_fund  * 100:.1f}%")


# ── CHART — 4-PANEL ───────────────────────────────────────────────────────────
print("\n[4/6] Building 4-panel research chart...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

TITLE_COL = "#e6edf3"
AXIS_COL  = "#8b949e"
BG_PANEL  = "#161b22"
GREEN     = "#3fb950"
RED       = "#f85149"
BLUE      = "#58a6ff"
ORANGE    = "#d29922"
PURPLE    = "#bc8cff"

def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=TITLE_COL, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=AXIS_COL, labelsize=8)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(AXIS_COL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color(AXIS_COL)
    ax.xaxis.label.set_color(AXIS_COL)


tickers_with_results = [t for t in TICKERS if t in results_all]

# Panel 1: IC comparison bar chart — price vs fundamental, per ticker
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Panel 1 — IC Comparison: Price-Only vs Price+Fundamental")

x = np.arange(len(tickers_with_results))
w = 0.35
ics_price = [results_all[t]["rho_price"] for t in tickers_with_results]
ics_fund  = [results_all[t]["rho_fund"]  for t in tickers_with_results]

b1 = ax1.bar(x - w/2, ics_price, w, color=BLUE,   label="Price only",  alpha=0.8)
b2 = ax1.bar(x + w/2, ics_fund,  w, color=ORANGE, label="Price + Fund", alpha=0.8)

ax1.axhline(0,    color=AXIS_COL, lw=0.8, ls="--")
ax1.axhline(0.05, color=GREEN,    lw=1.0, ls=":",  label="IC target (0.05)")
ax1.set_xticks(x)
ax1.set_xticklabels(tickers_with_results)
ax1.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)
ax1.set_ylabel("Spearman IC", fontsize=8)

for bar, val in zip(list(b1) + list(b2), ics_price + ics_fund):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{val:+.3f}", ha="center", va="bottom",
             color=TITLE_COL, fontsize=8, fontweight="bold")

# Panel 2: Equity curves — price-only vs fundamental-enhanced, per ticker
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Panel 2 — Equity Curve: Price-Only vs Price+Fundamental")

colors_price = [BLUE,   BLUE]
colors_fund  = [GREEN,  ORANGE]
ls_price     = ["-",    "--"]
ls_fund      = ["-",    "--"]

for i, ticker in enumerate(tickers_with_results):
    r = results_all[ticker]
    cum_price = (1 + r["pnl_price"]).cumprod()
    cum_fund  = (1 + r["pnl_fund"]).cumprod()

    ax2.plot(cum_price.values, color=BLUE   if i == 0 else PURPLE, lw=1.5, ls="--",
             label=f"{ticker} price", alpha=0.7)
    ax2.plot(cum_fund.values,  color=GREEN  if i == 0 else ORANGE, lw=2.0,
             label=f"{ticker} +fund")

ax2.axhline(1.0, color=AXIS_COL, lw=0.8, ls="--", alpha=0.5)
ax2.legend(fontsize=7, facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)
ax2.set_ylabel("Cumulative Return", fontsize=8)
ax2.set_xlabel("Bar (test period)", fontsize=8)

# Panel 3: Feature importance — Ridge coefficients (fundamental vs price)
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Panel 3 — Feature Importance (Ridge Coefficients)")

if tickers_with_results:
    ticker = tickers_with_results[0]
    df_raw  = intraday[ticker]
    df, price_features = engineer_price_features(df_raw, ticker)

    fund = fundamental_features[ticker]
    df["pe_ratio"]     = fund["pe_ratio"]
    df["short_ratio"]  = fund["short_ratio"]
    df["eps_momentum"] = fund["eps_momentum"]

    fund_feats  = [f for f in ["pe_ratio", "short_ratio", "eps_momentum"]
                   if df[f].notna().any()]
    all_feats   = price_features + fund_feats
    feat_df     = df[all_feats + ["forward_ret"]].dropna()

    if len(feat_df) >= 20:
        X_all = feat_df[all_feats].values
        y_all = feat_df["forward_ret"].values

        pipe_imp = Pipeline([("scaler", StandardScaler()),
                              ("ridge",  Ridge(alpha=RIDGE_ALPHA))])
        pipe_imp.fit(X_all, y_all)
        coefs = pipe_imp.named_steps["ridge"].coef_

        feature_importance = pd.Series(np.abs(coefs), index=all_feats).sort_values()
        colors_feat = [ORANGE if f in fund_feats else BLUE for f in feature_importance.index]

        ax3.barh(feature_importance.index, feature_importance.values,
                 color=colors_feat, alpha=0.8)
        ax3.axvline(0, color=AXIS_COL, lw=0.8)

        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=BLUE,   label="Price feature"),
                      Patch(facecolor=ORANGE, label="Fundamental feature")]
        ax3.legend(handles=legend_els, fontsize=8,
                   facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)
        ax3.set_xlabel("|Ridge Coefficient|", fontsize=8)
        ax3.set_title(f"Panel 3 — Feature Importance ({ticker}, Ridge α={RIDGE_ALPHA})",
                      color=TITLE_COL, fontsize=10, fontweight="bold", pad=8)

# Panel 4: Five Numbers Scorecard — both tickers, both models
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(BG_PANEL)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis("off")
ax4.set_title("Panel 4 — Five Numbers Scorecard", color=TITLE_COL,
               fontsize=10, fontweight="bold", pad=8)

y_pos = 0.95
ax4.text(0.01, y_pos, "METRIC",         color=AXIS_COL,  fontsize=7, va="top", fontfamily="monospace")
ax4.text(0.40, y_pos, "PRICE",          color=BLUE,      fontsize=7, va="top", fontfamily="monospace")
ax4.text(0.65, y_pos, "+FUNDAMENTAL",   color=ORANGE,    fontsize=7, va="top", fontfamily="monospace")
y_pos -= 0.06
ax4.text(0.01, y_pos, "─" * 40, color=AXIS_COL, fontsize=7, va="top", fontfamily="monospace")
y_pos -= 0.06

for ticker in tickers_with_results:
    r = results_all[ticker]

    ax4.text(0.01, y_pos, f"── {ticker} ──────────────────────────",
             color=TITLE_COL, fontsize=7, va="top", fontfamily="monospace", fontweight="bold")
    y_pos -= 0.06

    rows = [
        ("IC",          f"{r['rho_price']:+.4f}", f"{r['rho_fund']:+.4f}",
         GREEN if r["ic_lift"] > 0 else RED),
        ("Gross Ret",   f"{r['gross_price'] * 100:+.2f}%", f"{r['gross_fund'] * 100:+.2f}%",
         GREEN if r["gross_fund"] > r["gross_price"] else ORANGE),
        ("Net Ret",     f"{r['net_price'] * 100:+.2f}%",   f"{r['net_fund'] * 100:+.2f}%",
         GREEN if r["net_fund"] > r["net_price"] else ORANGE),
        ("PSR",         f"{r['psr_price'] * 100:.1f}%",    f"{r['psr_fund'] * 100:.1f}%",
         GREEN if r["psr_fund"] > r["psr_price"] else ORANGE),
        ("Sharpe",      f"{r['sr_price']:.3f}",             f"{r['sr_fund']:.3f}",
         GREEN if r["sr_fund"] > r["sr_price"] else ORANGE),
        ("IC lift",     "—",   f"{r['ic_lift']:+.4f}  {'↑' if r['ic_lift'] > 0 else '↓'}",
         GREEN if r["ic_lift"] > 0 else RED),
    ]

    for metric, v_price, v_fund, col in rows:
        ax4.text(0.01, y_pos, metric,  color=AXIS_COL, fontsize=7, va="top", fontfamily="monospace")
        ax4.text(0.40, y_pos, v_price, color=BLUE,     fontsize=7, va="top", fontfamily="monospace")
        ax4.text(0.65, y_pos, v_fund,  color=col,      fontsize=7, va="top", fontfamily="monospace")
        y_pos -= 0.055

    y_pos -= 0.02


fig.suptitle("Fundamental Features in ML Ridge | NVDA & MSFT | Price vs Price+Fundamental",
             color=TITLE_COL, fontsize=13, fontweight="bold", y=0.98)

out_path = "/Users/patiencefuglo/Desktop/intraday-alpha-research/charts/fundamental_features.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"    Chart saved → {out_path}")

# ── FINAL READING ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("READING")
print("=" * 72)

for ticker in tickers_with_results:
    r = results_all[ticker]
    print(f"""
{ticker}:
    IC (price only):     {r['rho_price']:+.4f}
    IC (price + fund):   {r['rho_fund']:+.4f}
    IC lift:             {r['ic_lift']:+.4f}  {'IMPROVEMENT' if r['ic_lift'] > 0 else 'WORSE — fundamentals hurt'}
    Net Return (price):  {r['net_price'] * 100:+.2f}%
    Net Return (+fund):  {r['net_fund']  * 100:+.2f}%
    PSR (+fund):         {r['psr_fund']  * 100:.1f}%
    Fundamental features used: {', '.join(r['fund_features_used'])}""")

print(f"""
KEY LESSONS:

1. THE FUNDAMENTAL TOUCH
   → Earnings momentum tells you WHY institutional money is moving.
   → P/E ratio sets the regime: growth stocks have different momentum
     dynamics than value stocks. Same signal, different outcome.
   → Short interest is a volatility amplifier: high short + price breakout
     = explosive move (short squeeze). The ML model should know this.

2. LOOK-AHEAD BIAS IS THE HARDEST PROBLEM
   → yfinance .info returns today's P/E and short interest.
   → In a real backtest, you must use P/E from the day you were trading.
   → If P/E data from 2021 is not available point-in-time, do NOT use it.
   → Bloomberg or FactSet solve this. Free data does not.
   → This study demonstrates the architecture — not a clean backtest.

3. CONSTANT FUNDAMENTAL FEATURES LIMIT THE STUDY
   → All bars in this 60-day window see the same fundamental values.
   → The model cannot learn that "high P/E in 2021 means X, low P/E in 2022
     means Y" because the feature does not vary over time.
   → In production: merge fundamental data on announcement dates,
     forward-fill until the next announcement.

4. THE IC LIFT IS THE TEST
   → If IC(price + fund) > IC(price only): fundamentals add real information.
   → If the lift is near zero: the price features already capture the
     information the fundamentals encode.
   → At intraday frequency, fundamental lift is often small — price moves
     faster than fundamental data releases.

INTERVIEW LINE:
    "I added three fundamental features — earnings momentum, P/E ratio,
     and short interest — to the ML Ridge intraday signal.
     The key finding: fundamental data is hard to use correctly at intraday
     frequency. The critical issue is look-ahead bias — you must use only
     data that was available at the time of each trade.
     Free data sources (yfinance) return point-in-time data only for price.
     A production system needs Bloomberg or Compustat for clean fundamental
     time series. The architecture works — the data sourcing is the constraint."
""")
print("=" * 72)
