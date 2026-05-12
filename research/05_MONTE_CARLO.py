"""
============================================================
05_MONTE_CARLO.py  |  Senior Level  |  Monte Carlo P&L Simulation
============================================================

HYPOTHESIS
----------
The NVDA intraday momentum signal shows a gross positive Sharpe.
But is that edge real — or did we just get lucky with the ORDER the trades happened?

Monte Carlo answers this. It asks:
  "If I randomly reshuffled my 200 trade returns, what range of outcomes is possible?"

If the real equity curve is BETTER than the median shuffled path, the sequence
of returns matters, but the edge is in the RETURNS themselves — not the luck.
If the real curve is INSIDE the middle of the simulated range, the edge holds
regardless of trade order. If the real curve falls in the BOTTOM 5%, we got unlucky
and should expect worse performance going forward.

-------------------------------------------------------------
WHAT IS MONTE CARLO SIMULATION?
-------------------------------------------------------------

Monte Carlo = simulate many possible futures by sampling from the past.

In trading we use it like this:
  1. Run the real strategy → collect all trade returns: [+0.003, -0.001, +0.005, ...]
  2. Bootstrap 1,000 new sequences: randomly pick-with-replacement from that list
  3. For each sequence, compound the returns to make an equity curve
  4. Stack all 1,000 curves → you get a DISTRIBUTION of possible outcomes
  5. Read off: median path, p5 (bad luck), p95 (good luck), P(ruin)

Key insight: we are NOT predicting the future. We are asking:
  "If the SAME signal edge repeats, how good or bad could returns get by chance?"

-------------------------------------------------------------
BOOTSTRAP vs PARAMETRIC
-------------------------------------------------------------

Parametric Monte Carlo: assume returns are normally distributed → sample from N(μ, σ)
Bootstrap Monte Carlo: resample actual returns with replacement (no distribution assumption)

We use BOOTSTRAP because:
  - Intraday returns have fat tails (extreme moves happen more than normal predicts)
  - Bootstrap preserves the actual distribution of past returns
  - No assumption needed — the data speaks for itself

Limitation: bootstrap assumes returns are i.i.d. (independent, identically distributed)
In practice, consecutive trades may be correlated (momentum). We accept this simplification.

-------------------------------------------------------------
FIVE NUMBERS (adapted for path analysis)
-------------------------------------------------------------

Standard five numbers still apply to the underlying signal.
Monte Carlo adds three path-level metrics:

  P50 (Median Path)    — the "expected" outcome if edge repeats
  P5  (5th Percentile) — the bad-luck scenario; use this to size positions
  P95 (95th Percentile)— the good-luck scenario
  P(ruin)              — fraction of paths where equity falls below -20%
                         threshold: < 5%

DECISION RULE:
  P50 > 0               → gross edge is real on the median path
  P5  > -15%            → even in bad luck, drawdown stays manageable
  P(ruin) < 5%          → ruin is unlikely given this edge size and trade count

-------------------------------------------------------------
THRESHOLDS (what counts as passing)
-------------------------------------------------------------

  Metric          Threshold    Reason
  -----------     ---------    ------
  Median path     > 0%         Edge must be positive on the average outcome
  P5 path         > -15%       Bad scenario must not blow up the account
  P(ruin <-20%)   < 5%         Ruin must be rare under this signal
  IC              > 0.05       Predictions must track real returns
  PSR             > 95%        Sharpe must be statistically real

-------------------------------------------------------------
INTERVIEW LINE
-------------------------------------------------------------
"Monte Carlo tells me the range of outcomes my signal can produce given my
trade history. The median path confirms the edge is real. The 5th percentile
path tells me the realistic bad-luck scenario — that is the number I use for
position sizing. If P(ruin) is above 5%, the position is too large regardless
of the signal quality."

============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — DATA
# ============================================================

def download_data(ticker: str, period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    """Download hourly OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df = df.dropna()
    df.columns = [c[0].lower() for c in df.columns]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 6 intraday features for the Ridge signal.

    Each feature captures a different dimension of price behaviour:
      momentum   — is price moving up or down?
      volatility — how wide are recent swings?
      volume     — is participation high or low?
      mean-rev   — is price far from its average?
    """
    out = df.copy()

    # Price momentum over 3 and 6 bars
    out["ret_1"]    = out["close"].pct_change(1)               # 1-bar return
    out["ret_3"]    = out["close"].pct_change(3)               # 3-bar momentum
    out["ret_6"]    = out["close"].pct_change(6)               # 6-bar momentum

    # Volatility: rolling standard deviation of returns (normalises for regime)
    out["vol_6"]    = out["ret_1"].rolling(6).std()

    # Volume ratio: current volume vs 10-bar average (detects institutional activity)
    out["vol_ratio"] = out["volume"] / out["volume"].rolling(10).mean()

    # Mean reversion: distance from 20-bar moving average (z-score)
    ma20            = out["close"].rolling(20).mean()
    std20           = out["close"].rolling(20).std()
    out["zscore"]   = (out["close"] - ma20) / (std20 + 1e-9)

    # Forward return: what we are trying to predict (next bar close-to-close)
    out["fwd_ret"]  = out["close"].pct_change(1).shift(-1)

    out = out.dropna()
    return out


# ============================================================
# SECTION 2 — SIGNAL (Ridge Regression)
# ============================================================

def train_predict(X_train, y_train, X_test, alpha=0.1):
    """
    Ridge regression: minimise RSS + alpha * ||w||^2
    alpha controls regularisation strength (prevents overfitting on small samples).
    Returns predictions on the test set.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Standardise: Ridge is sensitive to feature scale
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)

    model  = Ridge(alpha=alpha)
    model.fit(X_tr, y_train)
    return model.predict(X_te)


def walk_forward(df: pd.DataFrame, n_folds: int = 3):
    """
    Purged walk-forward: train on past, test on unseen future.
    embargo_bars = 6 → skip 6 bars after each training window to prevent leakage.
    Returns all out-of-sample predictions concatenated.
    """
    features     = ["ret_1", "ret_3", "ret_6", "vol_6", "vol_ratio", "zscore"]
    embargo_bars = 6                             # one forward-return horizon = 6 bars

    n        = len(df)
    fold_sz  = n // (n_folds + 1)               # rough fold size
    all_preds = []

    for fold in range(n_folds):
        train_end = fold_sz * (fold + 1)
        test_start = train_end + embargo_bars    # embargo: skip leakage zone
        test_end   = test_start + fold_sz

        if test_end > n:
            test_end = n

        train = df.iloc[:train_end]
        test  = df.iloc[test_start:test_end]

        if len(train) < 50 or len(test) < 20:
            continue

        preds = train_predict(
            train[features].values, train["fwd_ret"].values,
            test[features].values
        )

        result = test[["fwd_ret"]].copy()
        result["pred"]    = preds
        result["fold"]    = fold + 1
        all_preds.append(result)

    if not all_preds:
        return pd.DataFrame()

    return pd.concat(all_preds)


# ============================================================
# SECTION 3 — BACKTEST (position → trade returns)
# ============================================================

def backtest(pred_df: pd.DataFrame,
             conviction_threshold: float = 0.30,
             cost_per_trade: float = 0.0005) -> pd.Series:
    """
    Convert predictions to trade returns.

    conviction_threshold: only trade the top 30% most confident predictions
    cost_per_trade: 0.05% round-trip cost (commission + spread)

    Returns a Series of per-trade net returns.
    """
    df = pred_df.copy()

    # Rank predictions within each fold; trade only top 30% by absolute value
    df["abs_pred"] = df["pred"].abs()
    df["rank"]     = df.groupby("fold")["abs_pred"].rank(pct=True)

    df["signal"]   = 0
    df.loc[(df["rank"] > (1 - conviction_threshold)) & (df["pred"] > 0), "signal"] =  1
    df.loc[(df["rank"] > (1 - conviction_threshold)) & (df["pred"] < 0), "signal"] = -1

    # Net return = direction * actual_return - cost
    df["trade_ret"] = df["signal"] * df["fwd_ret"] - cost_per_trade * df["signal"].abs()

    # Keep only bars where we actually traded
    trades = df[df["signal"] != 0]["trade_ret"]
    return trades


# ============================================================
# SECTION 4 — STATISTICS
# ============================================================

def compute_ic(pred_df: pd.DataFrame) -> float:
    """
    IC = Pearson correlation between predictions and actual forward returns.
    Measures whether the model is pointing in the right direction.
    Target: IC > 0.05 (anything above zero is better than random).
    """
    return pred_df["pred"].corr(pred_df["fwd_ret"])


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252 * 6) -> float:
    """
    Annualised Sharpe Ratio = (mean return / std return) * sqrt(periods per year)
    For hourly bars: 252 trading days × ~6 bars per day = 1512 bars/year.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def compute_psr(sharpe_obs: float, n_obs: int,
                skew: float, kurt: float,
                sr_benchmark: float = 0.0) -> float:
    """
    PSR = P(true SR > sr_benchmark | observed SR, skew, kurtosis)
    Lopez de Prado 2014. Corrects for non-normality of returns.

    PSR < 95% → Sharpe may be statistical noise.
    PSR > 95% → Sharpe is statistically significant.
    """
    if n_obs < 10:
        return 0.0
    se = np.sqrt((1 - skew * sharpe_obs + (kurt - 1) / 4 * sharpe_obs**2) / (n_obs - 1))
    z  = (sharpe_obs - sr_benchmark) / (se + 1e-9)
    return float(stats.norm.cdf(z))


def compute_dsr(sharpe_obs: float, n_obs: int,
                skew: float, kurt: float,
                n_trials: int = 15) -> tuple:
    """
    DSR = PSR against SR* (expected max Sharpe from n_trials by chance).
    SR* = expected max of n independent standard normals, scaled by signal vol.

    If DSR < 50%, the observed Sharpe is not above what we'd expect by chance
    from trying 15 different strategies. Multiple testing correction.
    """
    # SR* formula (Lopez de Prado, adjusted for non-normality)
    sr_star_annual = (1 - 0.5772) / np.sqrt(n_trials) + \
                     np.sqrt(np.log(n_trials) / 2)
    # Convert to per-bar units (hourly bars)
    bars_per_year  = 252 * 6
    sr_star        = sr_star_annual / np.sqrt(bars_per_year)
    psr_val        = compute_psr(sharpe_obs, n_obs, skew, kurt, sr_benchmark=sr_star)
    return float(psr_val), float(sr_star_annual)


# ============================================================
# SECTION 5 — MONTE CARLO ENGINE
# ============================================================

def monte_carlo_bootstrap(trade_returns: pd.Series,
                           n_simulations: int = 1000,
                           seed: int = 42) -> dict:
    """
    Bootstrap Monte Carlo simulation.

    Steps:
      1. Collect N actual trade returns
      2. For each simulation: draw N returns with replacement (bootstrap)
      3. Compound returns: equity[t] = equity[t-1] * (1 + r[t])
      4. Record terminal value and min value for each path
      5. Compute percentiles across all 1,000 paths

    Returns dict with all path data and summary statistics.

    WHY resample with replacement?
      Because we do not know the true distribution of returns.
      Resampling preserves the fat tails and skew of the actual data.
    """
    rng     = np.random.default_rng(seed)
    returns = trade_returns.values
    n       = len(returns)

    if n < 10:
        return {}

    # Simulate 1,000 equity paths
    # Each path: compound n draws-with-replacement from actual trade returns
    paths = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        sampled  = rng.choice(returns, size=n, replace=True)   # bootstrap draw
        paths[i] = np.cumprod(1 + sampled) - 1                 # compound returns

    # Terminal returns (end of each path)
    terminal = paths[:, -1]

    # Max drawdown per path: max of (peak - trough) / peak
    peak     = np.maximum.accumulate(paths + 1, axis=1)
    drawdowns = (paths + 1 - peak) / peak                      # negative values = drawdown
    max_dd_per_path = drawdowns.min(axis=1)                    # worst drawdown per path

    return {
        "paths":     paths,                                    # all 1000 paths
        "terminal":  terminal,                                 # terminal returns
        "median":    float(np.percentile(terminal, 50)),      # P50
        "p5":        float(np.percentile(terminal, 5)),       # P5 — bad luck
        "p95":       float(np.percentile(terminal, 95)),      # P95 — good luck
        "p_ruin":    float((max_dd_per_path < -0.20).mean()), # fraction hitting -20% at any point
        "n_trades":  n,
        "n_sims":    n_simulations,
    }


# ============================================================
# SECTION 6 — FULL PIPELINE
# ============================================================

def run_pipeline(ticker: str) -> dict:
    """
    Full pipeline for one ticker:
      download → features → walk-forward → backtest → stats → Monte Carlo
    Returns dict with all metrics.
    """
    print(f"\n{'='*50}")
    print(f"  {ticker}  |  Monte Carlo P&L Simulation")
    print(f"{'='*50}")

    # --- 1. Data ---
    df = download_data(ticker)
    if df.empty:
        print(f"  ERROR: no data for {ticker}")
        return {}
    df = build_features(df)
    print(f"  Bars loaded: {len(df)}")

    # --- 2. Walk-forward ---
    pred_df = walk_forward(df, n_folds=3)
    if pred_df.empty:
        print(f"  ERROR: walk-forward returned no predictions")
        return {}
    print(f"  Predictions: {len(pred_df)}")

    # --- 3. Backtest ---
    trades     = backtest(pred_df)
    n_trades   = len(trades)
    gross_ret  = float(trades.sum())                     # gross total return (no-cost proxy)
    net_ret    = float(trades.sum())                     # already net of cost in backtest()
    total_cost = float((trades[trades < 0].abs().sum())) # approximation
    print(f"  Trades: {n_trades}")

    # --- 4. Signal statistics ---
    ic         = compute_ic(pred_df)
    sharpe     = compute_sharpe(trades)
    skew_val   = float(trades.skew()) if len(trades) > 2 else 0.0
    kurt_val   = float(trades.kurtosis()) if len(trades) > 2 else 3.0
    psr        = compute_psr(sharpe, n_trades, skew_val, kurt_val)
    dsr, sr_star = compute_dsr(sharpe, n_trades, skew_val, kurt_val)
    max_dd     = float((trades.cumsum() - trades.cumsum().cummax()).min())

    # --- 5. Monte Carlo ---
    mc = monte_carlo_bootstrap(trades, n_simulations=1000)

    print(f"\n  FIVE NUMBERS + PATH ANALYSIS")
    print(f"  {'Metric':<30} {'Value':>10}  {'Threshold':>12}  {'Status':>8}")
    print(f"  {'-'*64}")

    def status(val, threshold, higher_is_better=True):
        ok = val > threshold if higher_is_better else val < threshold
        return "PASS ✓" if ok else "FAIL ✗"

    rows = [
        ("Gross Return (sum)",  f"{gross_ret:+.4f}",  "> 0",      gross_ret > 0),
        ("Net Return (sum)",    f"{net_ret:+.4f}",    "> 0",      net_ret > 0),
        ("Trade Count",         f"{n_trades}",         ">= 30",    n_trades >= 30),
        ("IC",                  f"{ic:+.4f}",          "> 0.050",  ic > 0.05),
        ("Sharpe (ann.)",       f"{sharpe:+.3f}",      "> 0.5",    sharpe > 0.5),
        ("PSR",                 f"{psr:.1%}",           "> 95%",    psr > 0.95),
        ("DSR",                 f"{dsr:.1%}",           "> 95%",    dsr > 0.95),
        ("--- Monte Carlo ---", "",                    "",          None),
        ("P50 Median Path",     f"{mc.get('median', 0):+.2%}", "> 0%", mc.get("median", 0) > 0),
        ("P5  Bad-Luck Path",   f"{mc.get('p5', 0):+.2%}",   "> -15%", mc.get("p5", 0) > -0.15),
        ("P95 Good-Luck Path",  f"{mc.get('p95', 0):+.2%}",  "",       None),
        ("P(ruin < -20%)",      f"{mc.get('p_ruin', 1):.1%}","< 5%",   mc.get("p_ruin", 1) < 0.05),
    ]

    for label, value, thresh, passed in rows:
        if passed is None:
            print(f"  {label:<30} {value:>10}  {thresh:>12}  {'':>8}")
        else:
            s = "PASS ✓" if passed else "FAIL ✗"
            print(f"  {label:<30} {value:>10}  {thresh:>12}  {s:>8}")

    return {
        "ticker":    ticker,
        "gross_ret": gross_ret,
        "net_ret":   net_ret,
        "n_trades":  n_trades,
        "ic":        ic,
        "sharpe":    sharpe,
        "psr":       psr,
        "dsr":       dsr,
        "sr_star":   sr_star,
        "max_dd":    max_dd,
        "mc":        mc,
        "trades":    trades,
        "pred_df":   pred_df,
    }


# ============================================================
# SECTION 7 — CHART (standard format + path fan)
# ============================================================

def make_chart(results: list, save_path: str = "charts/monte_carlo_paths.png"):
    """
    Two-panel chart for each ticker:
      Panel 1: Monte Carlo fan — 1,000 paths, P5/P50/P95 highlighted
      Panel 2: Five numbers scorecard table

    Colours:
      grey   = individual paths (1,000 simulations)
      blue   = median path (P50)
      red    = bad-luck path (P5, 5th percentile)
      green  = good-luck path (P95, 95th percentile)
    """
    n_tickers = len(results)
    fig, axes = plt.subplots(n_tickers, 2,
                             figsize=(18, 6 * n_tickers),
                             gridspec_kw={"width_ratios": [2, 1]})

    if n_tickers == 1:
        axes = [axes]   # make it indexable as axes[0]

    fig.patch.set_facecolor("#0d1117")

    for row, res in enumerate(results):
        ticker = res["ticker"]
        mc     = res["mc"]
        ax_fan = axes[row][0]
        ax_tbl = axes[row][1]

        # --- Fan chart ---
        ax_fan.set_facecolor("#0d1117")
        paths   = mc["paths"]
        n_steps = paths.shape[1]
        x       = np.arange(n_steps)

        # Plot all 1,000 paths in light grey (low alpha = transparent overlay)
        for i in range(min(500, paths.shape[0])):   # show 500 to avoid over-cluttering
            ax_fan.plot(x, paths[i] * 100, color="#444455", alpha=0.05, lw=0.5)

        # Compute percentile bands
        p5_path  = np.percentile(paths, 5, axis=0) * 100
        p50_path = np.percentile(paths, 50, axis=0) * 100
        p95_path = np.percentile(paths, 95, axis=0) * 100

        # Shade the P5–P95 band
        ax_fan.fill_between(x, p5_path, p95_path, alpha=0.15, color="#4488ff")

        # Draw key percentile lines
        ax_fan.plot(x, p50_path, color="#4488ff", lw=2.5, label="P50 Median")
        ax_fan.plot(x, p5_path,  color="#ff4444", lw=2.0, ls="--", label="P5  (bad luck)")
        ax_fan.plot(x, p95_path, color="#44cc88", lw=2.0, ls="--", label="P95 (good luck)")

        # Reference lines
        ax_fan.axhline(0, color="white", lw=0.8, ls=":")
        ax_fan.axhline(-20, color="#ff6600", lw=0.8, ls=":", label="Ruin (-20%)")

        ax_fan.set_title(f"{ticker} — 1,000 Bootstrap Paths", color="white", fontsize=13, pad=10)
        ax_fan.set_xlabel("Trade Number", color="#aaaaaa", fontsize=10)
        ax_fan.set_ylabel("Cumulative Return (%)", color="#aaaaaa", fontsize=10)
        ax_fan.tick_params(colors="#aaaaaa")
        for spine in ax_fan.spines.values():
            spine.set_edgecolor("#333355")
        ax_fan.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white", loc="upper left")

        # Annotations: terminal values
        ax_fan.annotate(f"P50: {mc['median']:+.1%}",
                        xy=(n_steps - 1, p50_path[-1]),
                        xytext=(n_steps * 0.85, p50_path[-1] + 2),
                        color="#4488ff", fontsize=9)
        ax_fan.annotate(f"P5:  {mc['p5']:+.1%}",
                        xy=(n_steps - 1, p5_path[-1]),
                        xytext=(n_steps * 0.85, p5_path[-1] - 4),
                        color="#ff4444", fontsize=9)
        ax_fan.annotate(f"P95: {mc['p95']:+.1%}",
                        xy=(n_steps - 1, p95_path[-1]),
                        xytext=(n_steps * 0.85, p95_path[-1] + 2),
                        color="#44cc88", fontsize=9)

        # --- Scorecard table ---
        ax_tbl.set_facecolor("#0d1117")
        ax_tbl.axis("off")

        table_data = [
            ["Metric", "Value", "Pass?"],
            ["Gross Ret", f"{res['gross_ret']:+.3f}",
             "✓" if bool(res['gross_ret'] > 0) else "✗"],
            ["Net Ret", f"{res['net_ret']:+.3f}",
             "✓" if bool(res['net_ret'] > 0) else "✗"],
            ["Trade N", str(res["n_trades"]),
             "✓" if res["n_trades"] >= 30 else "✗"],
            ["IC", f"{res['ic']:+.4f}",
             "✓" if bool(res['ic'] > 0.05) else "✗"],
            ["Sharpe", f"{res['sharpe']:+.3f}",
             "✓" if bool(res['sharpe'] > 0.5) else "✗"],
            ["PSR", f"{res['psr']:.1%}",
             "✓" if bool(res['psr'] > 0.95) else "✗"],
            ["DSR", f"{res['dsr']:.1%}",
             "✓" if bool(res['dsr'] > 0.95) else "✗"],
            ["── MC ──", "──────", "──"],
            ["P50 (med)", f"{mc.get('median', 0):+.2%}",
             "✓" if bool(mc.get('median', 0) > 0) else "✗"],
            ["P5 (bad)", f"{mc.get('p5', 0):+.2%}",
             "✓" if bool(mc.get('p5', 0) > -0.15) else "✗"],
            ["P(ruin)", f"{mc.get('p_ruin', 1):.1%}",
             "✓" if bool(mc.get('p_ruin', 1) < 0.05) else "✗"],
        ]

        col_colors = [["#1a1a2e", "#1a1a2e", "#1a1a2e"]] * len(table_data)
        col_colors[0] = ["#2a2a4e", "#2a2a4e", "#2a2a4e"]   # header row

        def row_color(row_idx, data):
            if row_idx == 0 or data[2] in ["──", "✓", "✗"] is False:
                return col_colors[row_idx]
            if data[2] == "✓":
                return ["#0d1f0d", "#0d1f0d", "#0d2a0d"]
            elif data[2] == "✗":
                return ["#1f0d0d", "#1f0d0d", "#2a0d0d"]
            return col_colors[row_idx]

        cell_colors = [row_color(i, d) for i, d in enumerate(table_data)]

        tbl = ax_tbl.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            cellColours=cell_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.35)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_text_props(color="white")
            cell.set_edgecolor("#333355")

        ax_tbl.set_title(f"{ticker} — Five Numbers", color="white", fontsize=12, pad=10)

    plt.suptitle("Monte Carlo P&L Simulation — Bootstrap 1,000 Paths",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Chart saved → {save_path}")


# ============================================================
# SECTION 8 — CONCEPT SUMMARY (read this after running)
# ============================================================

CONCEPT_SUMMARY = """
============================================================
WHAT YOU LEARNED — MONTE CARLO P&L SIMULATION
============================================================

1. BOOTSTRAP RESAMPLING
   We draw N trade returns with replacement, 1,000 times.
   Each draw is a different "luck" scenario — same signal, different order.
   Result: a fan of 1,000 possible equity curves.

2. PERCENTILE BANDS
   P50 = median path → the "expected" outcome
   P5  = bad luck path → what happens if we get an unlucky draw?
         Use P5 for position sizing: if P5 = -15%, size so -15% = acceptable loss
   P95 = good luck → the upper bound we should not count on

3. P(RUIN)
   Count the fraction of 1,000 paths where equity hits -20% at any point.
   If P(ruin) > 5%, the signal + position size is too aggressive.
   Fix: reduce position size, not the signal.

4. WHY THIS MATTERS FOR INTERVIEWS
   "The signal looked great, but P(ruin) was 12% — too high for our risk budget.
    I halved the position size: P5 went from -22% to -11%, P(ruin) dropped to 2%.
    Same signal, safer sizing."

5. LIMITATION
   Bootstrap assumes returns are i.i.d. — no serial correlation.
   Real trading has momentum clustering. For more realism, use block bootstrap
   (draw blocks of consecutive returns to preserve autocorrelation structure).

============================================================
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os
    os.makedirs("charts", exist_ok=True)

    print(CONCEPT_SUMMARY)
    print("\nRunning Monte Carlo simulation on NVDA and MSFT...")
    print("This will take ~30 seconds (downloading 2y of data, 1,000 simulations each)")

    tickers  = ["NVDA", "MSFT"]
    results  = []

    for ticker in tickers:
        res = run_pipeline(ticker)
        if res:
            results.append(res)

    if results:
        chart_path = "charts/monte_carlo_paths.png"
        make_chart(results, save_path=chart_path)
        import subprocess
        subprocess.Popen(["open", chart_path])

    # Final research decision
    print("\n" + "="*55)
    print("  RESEARCH DECISION")
    print("="*55)
    for res in results:
        mc      = res["mc"]
        verdict = []
        if bool(res["gross_ret"] > 0):
            verdict.append("gross edge real")
        else:
            verdict.append("no gross edge")

        p_ruin = mc.get("p_ruin", 1.0)
        if p_ruin < 0.05:
            verdict.append(f"P(ruin) manageable ({p_ruin:.1%})")
        else:
            verdict.append(f"P(ruin) HIGH ({p_ruin:.1%}) — reduce position size")

        if bool(mc.get("p5", -1) > -0.15):
            verdict.append("bad-luck scenario acceptable")
        else:
            verdict.append("bad-luck scenario painful — revisit sizing")

        print(f"  {res['ticker']}: {' | '.join(verdict)}")

    print("\nNext: 06_PORTFOLIO_OPTIMIZATION.py")
