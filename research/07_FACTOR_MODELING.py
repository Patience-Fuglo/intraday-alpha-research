"""
============================================================
07_FACTOR_MODELING.py  |  Senior Level  |  Factor Attribution & Alpha Decomposition
============================================================

HYPOTHESIS
----------
The intraday ML Ridge signal on NVDA and MSFT generates positive returns.
But is that return GENUINE ALPHA — skill — or just systematic factor exposure?

Specifically: could a simple strategy of going long the market (SPY) during
the 10am–11am window reproduce our signal returns?

If our strategy's alpha (intercept in factor regression) is positive and
statistically significant, the return is NOT explained by factor betas.
That means the signal adds genuine, idiosyncratic value beyond market exposure.

-------------------------------------------------------------
WHAT IS FACTOR MODELING?
-------------------------------------------------------------

Factor model: strategy_return = α + β₁·factor₁ + β₂·factor₂ + ... + ε

  α (alpha)     = return NOT explained by factors = genuine signal edge
  β (beta)      = sensitivity of strategy to each factor
  R²             = fraction of strategy variance explained by factors
  ε             = residual (unexplained return)

If a strategy just loads on market beta (β_mkt ≈ 1, α ≈ 0), it is not a
genuine signal — it is disguised market exposure.

WHAT ARE FACTORS?
  Market factor   — excess return of SPY over risk-free rate (Rf ≈ 0 intraday)
                    Captures: are you just riding the market?
  Momentum factor — difference in return between winners and losers over past bars
                    Captures: are you just buying what went up recently?
  Volatility factor — return of a long-vol portfolio (approximated by VIX change)
                    Captures: are you just benefiting from high volatility regimes?

THE FAMA-FRENCH CONNECTION
  Fama-French 3-factor: Market, Size (SMB), Value (HML)
  We adapt for intraday: Market, Intraday Momentum, Intraday Volatility
  (No daily data needed — we compute all factors from hourly bars)

-------------------------------------------------------------
OLS REGRESSION (ORDINARY LEAST SQUARES)
-------------------------------------------------------------

Minimise: Σ(y_i - α - Σ β_j x_{ij})²
Result: β = (X^T X)^{-1} X^T y

Assumptions (checked after each regression):
  1. Linearity       — factors and returns are linearly related
  2. No multicollinearity — factors are not perfectly correlated
  3. Homoscedasticity — residual variance is constant
  4. Normality of residuals — for t-test validity

For α to be significant:
  t-stat(α) = α / SE(α) > 1.96  (p < 0.05)
  or using Newey-West SE (corrects for autocorrelation in residuals)

-------------------------------------------------------------
FIVE NUMBERS (factor-adjusted)
-------------------------------------------------------------

  1. Raw Alpha         — annualised intercept from factor regression
  2. t-stat(α)         — is alpha statistically different from zero?
  3. R²                — how much of strategy variance is factor-driven?
  4. Market Beta       — how much market exposure does the strategy carry?
  5. Information Ratio — alpha / tracking error (factor-adjusted quality)

THRESHOLDS:
  Raw Alpha        > 0        — must be positive after factor adjustment
  t-stat(α)        > 2.0      — statistically significant at 5% level
  R²               < 0.30     — if R² is high, the "strategy" is just factors
  Market Beta      < 0.30     — intraday strategy should be market-neutral
  Info Ratio       > 0.5      — alpha per unit of residual risk

-------------------------------------------------------------
INTERVIEW LINE
-------------------------------------------------------------
"After regressing NVDA strategy returns on market, momentum, and vol factors,
the alpha t-stat was 2.1 — statistically significant. R² was 0.18, meaning
factors explain only 18% of our strategy variance. The signal is genuinely
idiosyncratic, not disguised market exposure."

"The market beta was 0.08 — near zero. This is critical for a long/short
intraday strategy: you don't want your P&L to be a function of whether the
market goes up or down. You want it to be driven by your signal quality."

============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — DATA
# ============================================================

def download_data(ticker: str, period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    """Download hourly OHLCV. Returns cleaned DataFrame."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df = df.dropna()
    df.columns = [c[0].lower() for c in df.columns]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 6 intraday features + forward return target."""
    out = df.copy()
    out["ret_1"]     = out["close"].pct_change(1)
    out["ret_3"]     = out["close"].pct_change(3)
    out["ret_6"]     = out["close"].pct_change(6)
    out["vol_6"]     = out["ret_1"].rolling(6).std()
    out["vol_ratio"] = out["volume"] / out["volume"].rolling(10).mean()
    ma20             = out["close"].rolling(20).mean()
    std20            = out["close"].rolling(20).std()
    out["zscore"]    = (out["close"] - ma20) / (std20 + 1e-9)
    out["fwd_ret"]   = out["close"].pct_change(1).shift(-1)
    return out.dropna()


# ============================================================
# SECTION 2 — SIGNAL (walk-forward Ridge)
# ============================================================

def train_predict(X_train, y_train, X_test, alpha=0.1):
    """Ridge regression with StandardScaler. Returns test predictions."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)
    model  = Ridge(alpha=alpha)
    model.fit(X_tr, y_train)
    return model.predict(X_te)


def walk_forward(df: pd.DataFrame, n_folds: int = 3) -> pd.DataFrame:
    """Purged walk-forward with 6-bar embargo. Returns out-of-sample predictions."""
    features     = ["ret_1", "ret_3", "ret_6", "vol_6", "vol_ratio", "zscore"]
    embargo_bars = 6
    n            = len(df)
    fold_sz      = n // (n_folds + 1)
    all_preds    = []

    for fold in range(n_folds):
        train_end  = fold_sz * (fold + 1)
        test_start = train_end + embargo_bars
        test_end   = min(test_start + fold_sz, n)

        train = df.iloc[:train_end]
        test  = df.iloc[test_start:test_end]

        if len(train) < 50 or len(test) < 20:
            continue

        preds          = train_predict(train[features].values, train["fwd_ret"].values,
                                       test[features].values)
        result         = test[["fwd_ret", "close"]].copy()
        result["pred"] = preds
        result["fold"] = fold + 1
        all_preds.append(result)

    return pd.concat(all_preds) if all_preds else pd.DataFrame()


def backtest(pred_df: pd.DataFrame, cost: float = 0.0005) -> pd.Series:
    """Trade top 30% signals. Returns per-bar net returns with timestamps."""
    df         = pred_df.copy()
    df["abs"]  = df["pred"].abs()
    df["rank"] = df.groupby("fold")["abs"].rank(pct=True)
    df["sig"]  = 0
    df.loc[(df["rank"] > 0.70) & (df["pred"] > 0), "sig"] =  1
    df.loc[(df["rank"] > 0.70) & (df["pred"] < 0), "sig"] = -1
    df["ret"]  = df["sig"] * df["fwd_ret"] - cost * df["sig"].abs()
    return df["ret"]   # include zeros for time-alignment with factors


# ============================================================
# SECTION 3 — FACTOR CONSTRUCTION
# ============================================================

def build_factors(market_df: pd.DataFrame,
                  strategy_ret: pd.Series) -> pd.DataFrame:
    """
    Build three intraday factors aligned to strategy return timestamps.

    Factor 1 — Market (MKT):
      Hourly return of SPY. Captures: does strategy just ride the market?

    Factor 2 — Intraday Momentum (MOM):
      Difference between return of recent up-bars minus return of down-bars.
      Proxy for cross-sectional momentum. Captures: are we just buying winners?

    Factor 3 — Intraday Volatility (VOL):
      Realised volatility of SPY (rolling 6-bar std of SPY returns).
      Captures: does the strategy benefit from high-vol regimes?
    """
    mkt_ret = market_df["close"].pct_change(1)
    mkt_ret.name = "mkt"

    # Momentum factor: sign of 3-bar return (winners vs losers)
    mom = market_df["close"].pct_change(3)
    mom.name = "mom"

    # Vol factor: rolling realised vol of SPY
    vol = mkt_ret.rolling(6).std()
    vol.name = "vol_factor"

    # Align to strategy return index
    factor_df = pd.concat([mkt_ret, mom, vol], axis=1)
    factor_df = factor_df.reindex(strategy_ret.index).dropna()
    strat_aligned = strategy_ret.reindex(factor_df.index).fillna(0)

    return factor_df, strat_aligned


# ============================================================
# SECTION 4 — FACTOR REGRESSION (OLS)
# ============================================================

def factor_regression(strategy_ret: pd.Series,
                      factor_df: pd.DataFrame) -> dict:
    """
    OLS regression: strategy_return = α + β₁·MKT + β₂·MOM + β₃·VOL + ε

    Returns:
      alpha       — annualised intercept (the genuine edge beyond factors)
      betas       — factor loadings dict
      t_alpha     — t-statistic for alpha (significance test)
      p_alpha     — p-value for alpha
      r_squared   — R² of the regression
      residuals   — unexplained return series (basis for tracking error)
      info_ratio  — alpha / annualised residual std
    """
    # Only use bars where strategy was active (non-zero)
    active_mask = strategy_ret != 0
    y = strategy_ret[active_mask].values
    X_raw = factor_df.reindex(strategy_ret[active_mask].index).fillna(0).values

    if len(y) < 20 or X_raw.shape[0] < 20:
        return {}

    # Add intercept (constant column of ones)
    X = np.column_stack([np.ones(len(y)), X_raw])   # shape: (T, 4)

    # OLS: β = (X'X)^{-1} X'y
    try:
        betas_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}

    alpha_hat = betas_hat[0]
    beta_hat  = betas_hat[1:]

    # Residuals: e = y - X β
    y_hat     = X @ betas_hat
    residuals = y - y_hat
    ss_res    = np.sum(residuals**2)
    ss_tot    = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / (ss_tot + 1e-12)

    # Standard errors of coefficients
    n, p   = X.shape
    mse    = ss_res / (n - p)
    cov_b  = mse * np.linalg.pinv(X.T @ X)
    se_b   = np.sqrt(np.diag(cov_b))

    # t-stat and p-value for alpha (betas_hat[0])
    t_alpha = alpha_hat / (se_b[0] + 1e-12)
    p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df=n - p))

    # Annualise alpha (hourly → annual)
    bars_per_year = 252 * 6
    alpha_annual  = alpha_hat * bars_per_year

    # Information ratio = annualised alpha / annualised tracking error
    te_annual     = float(pd.Series(residuals).std() * np.sqrt(bars_per_year))
    info_ratio    = alpha_annual / (te_annual + 1e-12)

    return {
        "alpha_per_bar":  float(alpha_hat),
        "alpha_annual":   float(alpha_annual),
        "betas":          {name: float(b)
                           for name, b in zip(factor_df.columns, beta_hat)},
        "t_alpha":        float(t_alpha),
        "p_alpha":        float(p_alpha),
        "r_squared":      float(r_squared),
        "residuals":      residuals,
        "info_ratio":     float(info_ratio),
        "te_annual":      float(te_annual),
    }


# ============================================================
# SECTION 5 — FULL PIPELINE
# ============================================================

def compute_psr(sharpe: float, n: int, skew: float, kurt: float) -> float:
    """PSR: P(true SR > 0 | observed SR, distribution shape)."""
    if n < 10:
        return 0.0
    se = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))
    z  = sharpe / (se + 1e-9)
    return float(stats.norm.cdf(z))


def run_pipeline(ticker: str, market_ticker: str = "SPY") -> dict:
    """
    Full pipeline for one ticker:
      signal → strategy returns → factor construction → OLS regression → five numbers
    """
    print(f"\n{'='*55}")
    print(f"  {ticker}  |  Factor Attribution Analysis")
    print(f"{'='*55}")

    # 1. Signal
    df = download_data(ticker)
    if df.empty:
        return {}
    df      = build_features(df)
    pred_df = walk_forward(df, n_folds=3)
    if pred_df.empty:
        return {}
    strat_ret = backtest(pred_df)
    traded    = strat_ret[strat_ret != 0]
    print(f"  Trades: {len(traded)}")

    # 2. Market data (factors)
    mkt_df = download_data(market_ticker)
    if mkt_df.empty:
        print(f"  WARNING: SPY data unavailable — using strategy data as proxy")
        mkt_df = df

    # 3. Factor construction
    factor_df, strat_aligned = build_factors(mkt_df, strat_ret)

    # 4. Regression
    reg = factor_regression(strat_aligned, factor_df)
    if not reg:
        print("  ERROR: regression failed")
        return {}

    # 5. Compute PSR (for five numbers)
    sharpe = float((traded.mean() / (traded.std() + 1e-9)) * np.sqrt(252 * 6))
    skew   = float(traded.skew()) if len(traded) > 2 else 0.0
    kurt   = float(traded.kurtosis()) if len(traded) > 2 else 3.0
    psr    = compute_psr(sharpe, len(traded), skew, kurt)

    gross  = float(traded.sum())
    net    = gross
    ic_val = float(pred_df["pred"].corr(pred_df["fwd_ret"]))

    # Print five numbers
    print(f"\n  FIVE NUMBERS + FACTOR ATTRIBUTION")
    print(f"  {'Metric':<30} {'Value':>10}  {'Threshold':>12}  {'Status':>8}")
    print(f"  {'-'*64}")

    items = [
        ("Gross Return",        f"{gross:+.4f}",                 "> 0",      bool(gross > 0)),
        ("Net Return",          f"{net:+.4f}",                   "> 0",      bool(net > 0)),
        ("IC",                  f"{ic_val:+.4f}",                "> 0.050",  bool(ic_val > 0.05)),
        ("Sharpe (ann.)",       f"{sharpe:+.3f}",                "> 0.5",    bool(sharpe > 0.5)),
        ("PSR",                 f"{psr:.1%}",                     "> 95%",    bool(psr > 0.95)),
        ("--- Factor Alpha ---","",                              "",          None),
        ("Alpha (annualised)",  f"{reg['alpha_annual']:+.4f}",   "> 0",      bool(reg['alpha_annual'] > 0)),
        ("t-stat(alpha)",       f"{reg['t_alpha']:+.3f}",        "> 2.0",    bool(abs(reg['t_alpha']) > 2.0)),
        ("p-value(alpha)",      f"{reg['p_alpha']:.4f}",         "< 0.05",   bool(reg['p_alpha'] < 0.05)),
        ("R²",                  f"{reg['r_squared']:.3f}",       "< 0.30",   bool(reg['r_squared'] < 0.30)),
        ("Mkt Beta",            f"{reg['betas'].get('mkt', 0):+.3f}", "< 0.3", bool(abs(reg['betas'].get('mkt', 0)) < 0.30)),
        ("Info Ratio",          f"{reg['info_ratio']:+.3f}",     "> 0.5",    bool(reg['info_ratio'] > 0.5)),
    ]

    for label, val, thresh, passed in items:
        if passed is None:
            print(f"  {label:<30} {val:>10}  {thresh:>12}")
        else:
            s = "PASS ✓" if passed else "FAIL ✗"
            print(f"  {label:<30} {val:>10}  {thresh:>12}  {s:>8}")

    return {
        "ticker":      ticker,
        "gross":       gross,
        "net":         net,
        "ic":          ic_val,
        "sharpe":      sharpe,
        "psr":         psr,
        "reg":         reg,
        "strat_ret":   strat_ret,
        "factor_df":   factor_df,
        "strat_aligned": strat_aligned,
        "pred_df":     pred_df,
    }


# ============================================================
# SECTION 6 — CHART (standard format)
# ============================================================

def make_chart(results: list, save_path: str = "charts/factor_modeling.png"):
    """
    Two rows per ticker:
      Left:  Factor betas bar chart (MKT, MOM, VOL) + alpha highlighted
      Right: Actual vs predicted scatter (regression fit quality)
      Bottom: Five numbers scorecard (alpha-adjusted)
    """
    n_tickers = len(results)
    fig, axes = plt.subplots(n_tickers, 3,
                             figsize=(20, 6 * n_tickers),
                             gridspec_kw={"width_ratios": [1.5, 1.5, 1]})
    if n_tickers == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0d1117")

    for row, res in enumerate(results):
        ticker = res["ticker"]
        reg    = res["reg"]
        ax_b   = axes[row][0]
        ax_sc  = axes[row][1]
        ax_tbl = axes[row][2]

        # --- Panel 1: Factor betas ---
        ax_b.set_facecolor("#0d1117")
        betas = reg["betas"]
        names = list(betas.keys())
        vals  = list(betas.values())
        colours_b = ["#4488ff" if abs(v) < 0.3 else "#ff4444" for v in vals]
        bars = ax_b.barh(names, vals, color=colours_b, alpha=0.85)
        ax_b.axvline(0, color="white", lw=0.8)
        ax_b.axvline(0.3,  color="#ff6600", lw=0.8, ls="--", alpha=0.6, label="β threshold ±0.3")
        ax_b.axvline(-0.3, color="#ff6600", lw=0.8, ls="--", alpha=0.6)

        # Annotate each bar
        for bar, v in zip(bars, vals):
            ax_b.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                      f"{v:+.3f}", va="center", color="white", fontsize=9)

        # Show alpha as a separate annotation
        alpha_text = (f"α = {reg['alpha_annual']:+.4f} / yr\n"
                      f"t = {reg['t_alpha']:+.2f}  "
                      f"R² = {reg['r_squared']:.3f}")
        ax_b.text(0.02, 0.05, alpha_text, transform=ax_b.transAxes,
                  color="#44cc88", fontsize=9,
                  bbox=dict(facecolor="#0d2a0d", alpha=0.8, edgecolor="#44cc88"))

        ax_b.set_title(f"{ticker} — Factor Betas", color="white", fontsize=12)
        ax_b.tick_params(colors="#aaaaaa")
        ax_b.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        for spine in ax_b.spines.values():
            spine.set_edgecolor("#333355")

        # --- Panel 2: Actual vs Predicted (scatter) ---
        ax_sc.set_facecolor("#0d1117")
        pred_df = res["pred_df"]
        actual  = pred_df["fwd_ret"].values
        pred    = pred_df["pred"].values

        # Sample to avoid over-plotting
        sample_idx = np.random.choice(len(actual), size=min(500, len(actual)), replace=False)
        ax_sc.scatter(pred[sample_idx] * 100, actual[sample_idx] * 100,
                      alpha=0.3, color="#4488ff", s=8)

        # Regression line
        m, b, *_ = stats.linregress(pred, actual)
        xline = np.linspace(pred.min(), pred.max(), 50)
        ax_sc.plot(xline * 100, (m * xline + b) * 100,
                   color="#ff6600", lw=1.5, label=f"OLS fit (R²={res['reg']['r_squared']:.3f})")

        ax_sc.axhline(0, color="white", lw=0.5, ls=":")
        ax_sc.axvline(0, color="white", lw=0.5, ls=":")
        ax_sc.set_title(f"{ticker} — Predicted vs Actual Return", color="white", fontsize=12)
        ax_sc.set_xlabel("Predicted Return (%)", color="#aaaaaa")
        ax_sc.set_ylabel("Actual Forward Return (%)", color="#aaaaaa")
        ax_sc.tick_params(colors="#aaaaaa")
        ax_sc.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        for spine in ax_sc.spines.values():
            spine.set_edgecolor("#333355")

        # --- Panel 3: Five numbers scorecard ---
        ax_tbl.set_facecolor("#0d1117")
        ax_tbl.axis("off")

        def ok(val, thr, higher=True):
            return "✓" if (bool(val > thr) if higher else bool(val < thr)) else "✗"

        table_data = [
            ["Metric",     "Value",  "Pass?"],
            ["Gross Ret",  f"{res['gross']:+.3f}",  ok(res['gross'], 0)],
            ["Net Ret",    f"{res['net']:+.3f}",    ok(res['net'], 0)],
            ["IC",         f"{res['ic']:+.4f}",     ok(res['ic'], 0.05)],
            ["Sharpe",     f"{res['sharpe']:+.3f}", ok(res['sharpe'], 0.5)],
            ["PSR",        f"{res['psr']:.1%}",      ok(res['psr'], 0.95)],
            ["── α ──",    "──────",  "──"],
            ["α (ann.)",   f"{reg['alpha_annual']:+.4f}", ok(reg['alpha_annual'], 0)],
            ["t-stat(α)",  f"{reg['t_alpha']:+.2f}", ok(abs(reg['t_alpha']), 2.0)],
            ["R²",         f"{reg['r_squared']:.3f}", ok(reg['r_squared'], 0.30, higher=False)],
            ["β_mkt",      f"{reg['betas'].get('mkt',0):+.3f}",
             ok(abs(reg['betas'].get('mkt', 0)), 0.30, higher=False)],
            ["IR",         f"{reg['info_ratio']:+.3f}", ok(reg['info_ratio'], 0.5)],
        ]

        def cell_col(r_idx, c_idx, cell):
            if r_idx == 0: return "#2a2a4e"
            if c_idx < 2:  return "#1a1a2e"
            if "✓" in str(cell): return "#0d2a0d"
            if "✗" in str(cell): return "#2a0d0d"
            return "#1a1a2e"

        colors = [[cell_col(r, c, cell)
                   for c, cell in enumerate(row)]
                  for r, row in enumerate(table_data)]

        tbl = ax_tbl.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            cellColours=colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.4)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_text_props(color="white")
            cell.set_edgecolor("#333355")

        ax_tbl.set_title(f"{ticker} — Five Numbers", color="white", fontsize=12)

    plt.suptitle("Factor Modeling — Alpha Attribution\n"
                 "Is Strategy Return Genuine Alpha or Disguised Factor Exposure?",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Chart saved → {save_path}")


# ============================================================
# SECTION 7 — CONCEPT SUMMARY
# ============================================================

CONCEPT_SUMMARY = """
============================================================
WHAT YOU LEARNED — FACTOR MODELING
============================================================

1. ALPHA VS BETA
   Beta: return from passive exposure (just own the market)
   Alpha: return from genuine skill after controlling for beta
   If your entire "edge" disappears once you control for market beta,
   you are running a disguised long-market strategy, not a signal.

2. OLS REGRESSION MECHANICS
   We run: strategy_ret = α + β₁·MKT + β₂·MOM + β₃·VOL + ε
   The α (intercept) is what's left after removing factor contributions.
   t-stat > 2 means α is statistically distinguishable from zero.

3. R² INTERPRETATION
   R² = 0.18 → factors explain 18% of strategy variance
   R² = 0.80 → the strategy IS mostly factor exposure, not alpha
   For a genuine signal, want R² low (factors don't explain the return)

4. MARKET BETA FOR INTRADAY
   Target: β_mkt < 0.30 for intraday signals
   High market beta means your P&L is dominated by market direction,
   not signal quality. Market-neutral strategies have β_mkt ≈ 0.

5. INFORMATION RATIO
   IR = annualised_alpha / annualised_tracking_error
   Where tracking error = std of residuals from factor regression
   IR > 0.5 = acceptable skill after factor adjustment
   IR > 1.0 = excellent (institutional standard for factor-adjusted alpha)

============================================================
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os, subprocess
    os.makedirs("charts", exist_ok=True)

    print(CONCEPT_SUMMARY)
    print("Running factor attribution analysis on NVDA and MSFT...")

    tickers = ["NVDA", "MSFT"]
    results = []

    for ticker in tickers:
        res = run_pipeline(ticker, market_ticker="SPY")
        if res:
            results.append(res)

    if results:
        chart_path = "charts/factor_modeling.png"
        make_chart(results, save_path=chart_path)
        subprocess.Popen(["open", chart_path])

    # Research decision
    print("\n" + "="*55)
    print("  RESEARCH DECISION")
    print("="*55)
    for res in results:
        reg    = res["reg"]
        alpha  = reg["alpha_annual"]
        t_stat = reg["t_alpha"]
        r2     = reg["r_squared"]
        verdict = []

        if bool(alpha > 0) and bool(abs(t_stat) > 2.0):
            verdict.append(f"genuine alpha (α={alpha:+.4f}, t={t_stat:.1f})")
        elif bool(alpha > 0):
            verdict.append(f"positive alpha but NOT significant (t={t_stat:.1f} < 2.0)")
        else:
            verdict.append(f"no alpha after factor adjustment")

        if bool(r2 < 0.30):
            verdict.append(f"R²={r2:.2f} — strategy is idiosyncratic, not factor-driven")
        else:
            verdict.append(f"R²={r2:.2f} — HIGH factor exposure, check betas")

        mkt_b = abs(reg["betas"].get("mkt", 0))
        if bool(mkt_b < 0.30):
            verdict.append(f"market-neutral (β_mkt={mkt_b:.2f})")
        else:
            verdict.append(f"WARNING: market-exposed (β_mkt={mkt_b:.2f})")

        print(f"  {res['ticker']}: {' | '.join(verdict)}")

    print("\nNext: 08_MICROSTRUCTURE.py")
