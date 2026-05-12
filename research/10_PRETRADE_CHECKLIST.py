"""
============================================================
10_PRETRADE_CHECKLIST.py  |  Senior Level  |  Pre-Trade Risk & Go/No-Go Gate
============================================================

HYPOTHESIS
----------
A signal that passes statistical tests (IC, PSR, DSR) is NOT automatically
approved for live trading. Before deploying capital, we run a structured
pre-trade checklist: a set of quantitative pass/fail gates that a signal
must clear before it sees real money.

Hypothesis: The NVDA ML Ridge signal will PASS the critical gates (positive
gross and net returns, IC > 0.05, PSR > 95%) but will FAIL the advisory
gates (DSR > 95%, trade count ≥ 50, P(ruin) < 5%) due to data limitation.

The checklist records the honest state of the signal: what we KNOW,
what we SUSPECT, and what we CANNOT YET CONFIRM.

-------------------------------------------------------------
WHAT IS A PRE-TRADE CHECKLIST?
-------------------------------------------------------------

A pre-trade checklist is a structured set of quantitative questions that
a signal must answer before it is deployed in production.

Purpose:
  1. Prevent emotional deployment (the signal "feels good" → skip testing)
  2. Document what was checked and what was skipped (audit trail)
  3. Force the researcher to distinguish KNOWN edge from SUSPECTED edge
  4. Create a consistent review standard across all strategies

The checklist is split into:
  CRITICAL gates  — if any FAIL, the signal does NOT trade. No exceptions.
  ADVISORY gates  — failures are noted but do not block deployment;
                    they require monitoring and a plan to address.

-------------------------------------------------------------
THE 8-ITEM CHECKLIST
-------------------------------------------------------------

CRITICAL (5 gates — all must pass):
  1. Gross Return > 0
     Does the signal produce positive return BEFORE fees?
     If gross is negative, the signal is anti-predictive. Stop here.

  2. Net Return > 0
     Does the signal produce positive return AFTER realistic costs?
     (Commission + bid-ask spread + market impact)
     Net ≤ 0 means costs exceed edge. Redesign or reduce cost.

  3. IC > 0.05
     Do predictions track actual returns with meaningful correlation?
     IC < 0.05 means predictions are essentially noise.

  4. PSR > 95%
     Is the Sharpe Ratio statistically significant (not noise)?
     PSR < 95% means the observed Sharpe could easily be random.

  5. Max Drawdown > -20%
     Does the signal's equity curve stay above the -20% ruin threshold?
     Drawdown > 20% means the signal risks blowing up a real account.

ADVISORY (3 gates — note failures, monitor):
  6. DSR > 95%
     Does the Sharpe survive multiple testing correction?
     DSR < 95% means observed Sharpe is within range of what 15 random
     strategies would produce by chance. Needs more data to confirm.

  7. Trade Count ≥ 50 (per fold)
     Is the sample large enough for statistical reliability?
     < 50 trades → high variance of all statistics. Results are noisy.

  8. P(ruin < -20%) < 5%
     Does Monte Carlo simulation show ruin is rare?
     P(ruin) > 5% means the position size needs to be reduced.

-------------------------------------------------------------
FIVE NUMBERS (final consolidated view)
-------------------------------------------------------------

The checklist IS the five numbers framework — formalised as a gate:

  # | Metric        | Critical? | Threshold
  --|---------------|-----------|----------
  1 | Gross Return  | Yes       | > 0
  2 | Total Costs   | Yes       | < Gross (net must be positive)
  3 | Net Return    | Yes       | > 0
  4 | IC            | Yes       | > 0.05
  5 | PSR           | Yes       | > 95%
  + | DSR           | Advisory  | > 95%
  + | Max Drawdown  | Yes       | > -20%
  + | P(ruin)       | Advisory  | < 5%
  + | Trade Count   | Advisory  | ≥ 50

-------------------------------------------------------------
THE GO/NO-GO VERDICT
-------------------------------------------------------------

  GO      — all 5 critical gates pass + at least 2 of 3 advisory pass
  NO-GO   — any critical gate fails
  WATCH   — all critical pass + 1 or more advisory fail (deploy small, monitor)

-------------------------------------------------------------
INTERVIEW LINES
-------------------------------------------------------------
"Before any signal goes live, it passes through our pre-trade checklist.
Five critical gates must all pass. If IC is below 0.05, we stop — I don't
care what the backtest P&L looks like. The three advisory gates tell us
whether to deploy full size or start with a small pilot position."

"The most common failure mode is: great gross return, but costs flip net
return negative. Or: PSR passes but DSR fails because we tested 15 variants.
In both cases, the checklist catches it before it costs real capital."

"IC below threshold means our predictions are not correlated with outcomes.
Maybe we have a data bug, maybe the feature decayed. Either way, deploying
it is gambling, not research."

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
# SECTION 1 — DATA & SIGNAL (full self-contained pipeline)
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
        result         = test[["fwd_ret"]].copy()
        result["pred"] = preds
        result["fold"] = fold + 1
        all_preds.append(result)

    return pd.concat(all_preds) if all_preds else pd.DataFrame()


def backtest(pred_df: pd.DataFrame, cost: float = 0.0005) -> pd.Series:
    """Trade top 30% by conviction. Returns per-bar net trade returns."""
    df         = pred_df.copy()
    df["abs"]  = df["pred"].abs()
    df["rank"] = df.groupby("fold")["abs"].rank(pct=True)
    df["sig"]  = 0
    df.loc[(df["rank"] > 0.70) & (df["pred"] > 0), "sig"] =  1
    df.loc[(df["rank"] > 0.70) & (df["pred"] < 0), "sig"] = -1
    df["ret"]  = df["sig"] * df["fwd_ret"] - cost * df["sig"].abs()
    return df[df["sig"] != 0]["ret"]


# ============================================================
# SECTION 2 — STATISTICS
# ============================================================

def compute_sharpe(returns: pd.Series, periods: int = 252 * 6) -> float:
    """Annualised Sharpe Ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods))


def compute_psr(sharpe: float, n: int, skew: float, kurt: float) -> float:
    """PSR = P(true SR > 0 | observed SR)."""
    if n < 10:
        return 0.0
    se = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))
    z  = sharpe / (se + 1e-9)
    return float(stats.norm.cdf(z))


def compute_dsr(sharpe: float, n: int, skew: float, kurt: float,
                k: int = 15) -> tuple:
    """DSR = PSR against SR* (expected max Sharpe from k trials by chance)."""
    sr_star_annual = (1 - 0.5772) / np.sqrt(k) + np.sqrt(np.log(k) / 2)
    bars_per_year  = 252 * 6
    sr_star_bar    = sr_star_annual / np.sqrt(bars_per_year)
    se = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))
    z  = (sharpe - sr_star_bar) / (se + 1e-9)
    return float(stats.norm.cdf(z)), float(sr_star_annual)


def monte_carlo_pct_ruin(trade_returns: pd.Series,
                          n_sims: int = 1000,
                          ruin_threshold: float = -0.20,
                          seed: int = 42) -> float:
    """
    Bootstrap 1,000 paths. Return fraction of paths that hit ruin threshold.
    P(ruin) < 5% is the advisory threshold.
    """
    rng     = np.random.default_rng(seed)
    returns = trade_returns.values
    n       = len(returns)
    if n < 5:
        return 1.0

    ruined = 0
    for _ in range(n_sims):
        sampled   = rng.choice(returns, size=n, replace=True)
        cum_ret   = np.cumprod(1 + sampled) - 1
        peak      = np.maximum.accumulate(cum_ret + 1)
        drawdowns = (cum_ret + 1 - peak) / peak
        if drawdowns.min() < ruin_threshold:
            ruined += 1

    return ruined / n_sims


# ============================================================
# SECTION 3 — CHECKLIST ENGINE
# ============================================================

def run_checklist(ticker: str, k_trials: int = 15) -> dict:
    """
    Run the full 8-item pre-trade checklist for one ticker.
    Returns complete results dict and verdict (GO / WATCH / NO-GO).
    """
    print(f"\n{'='*60}")
    print(f"  {ticker}  |  Pre-Trade Checklist")
    print(f"{'='*60}")

    # 1. Pipeline
    df = download_data(ticker)
    if df.empty:
        print(f"  ERROR: no data for {ticker}")
        return {}
    df      = build_features(df)
    pred_df = walk_forward(df, n_folds=3)
    if pred_df.empty:
        print(f"  ERROR: walk-forward returned no predictions")
        return {}

    trades     = backtest(pred_df)
    n_trades   = len(trades)
    gross_ret  = float(trades.sum())
    net_ret    = float(trades.sum())   # costs already included in backtest()
    total_cost = 0.0005 * n_trades     # 5bp per trade × count

    # Signal statistics
    ic         = float(pred_df["pred"].corr(pred_df["fwd_ret"]))
    sharpe     = compute_sharpe(trades)
    skew_val   = float(trades.skew()) if n_trades > 2 else 0.0
    kurt_val   = float(trades.kurtosis()) if n_trades > 2 else 3.0
    psr        = compute_psr(sharpe, n_trades, skew_val, kurt_val)
    dsr, sr_star = compute_dsr(sharpe, n_trades, skew_val, kurt_val, k=k_trials)
    cum_ret    = trades.cumsum()
    max_dd     = float((cum_ret - cum_ret.cummax()).min())
    p_ruin     = monte_carlo_pct_ruin(trades, n_sims=500)  # 500 sims for speed

    # Per-fold trade counts
    fold_counts = pred_df[pred_df["pred"].abs() > 0].groupby("fold").size()
    min_fold_n  = int(fold_counts.min()) if len(fold_counts) > 0 else 0

    # 2. Checklist evaluation
    checklist = [
        # (name, value, threshold, critical, pass_condition)
        ("Gross Return",    gross_ret,  0.0,    True,  bool(gross_ret > 0)),
        ("Net Return",      net_ret,    0.0,    True,  bool(net_ret > 0)),
        ("IC",              ic,         0.05,   True,  bool(ic > 0.05)),
        ("PSR",             psr,        0.95,   True,  bool(psr > 0.95)),
        ("Max Drawdown",    max_dd,    -0.20,   True,  bool(max_dd > -0.20)),
        ("DSR",             dsr,        0.95,   False, bool(dsr > 0.95)),
        ("Trade Count",     min_fold_n, 50,     False, bool(min_fold_n >= 50)),
        ("P(ruin<-20%)",    p_ruin,     0.05,   False, bool(p_ruin < 0.05)),
    ]

    # 3. Print checklist
    print(f"\n  {'#':<3} {'Gate':<22} {'Critical':<10} {'Value':>12}  {'Threshold':>10}  {'Status':>8}")
    print(f"  {'-'*72}")

    for i, (name, val, threshold, critical, passed) in enumerate(checklist):
        crit_label = "CRITICAL" if critical else "advisory"
        if isinstance(val, float):
            if name in ("IC", "PSR", "DSR", "P(ruin<-20%)"):
                val_str = f"{val:.4f}"
            elif name in ("Gross Return", "Net Return", "Max Drawdown"):
                val_str = f"{val:+.4f}"
            else:
                val_str = f"{val:.4f}"
        else:
            val_str = str(val)

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {i+1:<3} {name:<22} {crit_label:<10} {val_str:>12}  {str(threshold):>10}  {status:>8}")

    # 4. Verdict
    critical_items = [(name, passed) for name, val, thr, crit, passed in checklist if crit]
    advisory_items = [(name, passed) for name, val, thr, crit, passed in checklist if not crit]

    critical_fails = [name for name, passed in critical_items if not passed]
    advisory_fails = [name for name, passed in advisory_items if not passed]
    advisory_passes = sum(1 for _, passed in advisory_items if passed)

    if critical_fails:
        verdict = "NO-GO"
        verdict_reason = f"Critical failures: {', '.join(critical_fails)}"
    elif advisory_passes >= 2:
        verdict = "GO"
        verdict_reason = "All critical pass + 2+ advisory pass"
    else:
        verdict = "WATCH"
        verdict_reason = f"All critical pass. Advisory failures: {', '.join(advisory_fails)}"

    print(f"\n  {'─'*55}")
    print(f"  VERDICT: {verdict}")
    print(f"  Reason:  {verdict_reason}")
    print(f"  {'─'*55}")

    if verdict == "GO":
        print(f"  ACTION: Deploy signal at target position size.")
        print(f"          Monitor IC weekly. Retrain every quarter.")
    elif verdict == "WATCH":
        print(f"  ACTION: Deploy at 25% of target size. Monitor:")
        for fail_name in advisory_fails:
            print(f"    → {fail_name}: below threshold — collect more data or reduce risk")
    else:
        print(f"  ACTION: Do NOT deploy. Fix critical failures first:")
        for fail_name in critical_fails:
            print(f"    → {fail_name}: FAILED — investigate before any capital commitment")

    return {
        "ticker":       ticker,
        "gross_ret":    gross_ret,
        "net_ret":      net_ret,
        "total_cost":   total_cost,
        "ic":           ic,
        "sharpe":       sharpe,
        "psr":          psr,
        "dsr":          dsr,
        "sr_star":      sr_star,
        "max_dd":       max_dd,
        "p_ruin":       p_ruin,
        "n_trades":     n_trades,
        "min_fold_n":   min_fold_n,
        "verdict":      verdict,
        "verdict_reason": verdict_reason,
        "checklist":    checklist,
        "critical_fails": critical_fails,
        "advisory_fails": advisory_fails,
        "pred_df":      pred_df,
        "trades":       trades,
    }


# ============================================================
# SECTION 4 — CHART (standard format)
# ============================================================

def make_chart(results: list, save_path: str = "charts/pretrade_checklist.png"):
    """
    Two panels per ticker:
      Left:  Vertical checklist scorecard with CRITICAL / advisory labels
             and GO/WATCH/NO-GO verdict banner
      Right: Equity curve of the signal + drawdown shading

    Consistent colours:
      Green = PASS    Red = FAIL    Amber = advisory threshold
    """
    n_tickers = len(results)
    fig, axes = plt.subplots(n_tickers, 2,
                             figsize=(18, 7 * n_tickers),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    if n_tickers == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0d1117")

    VERDICT_COLORS = {
        "GO":     "#44cc88",
        "WATCH":  "#ffaa00",
        "NO-GO":  "#ff4444",
    }

    for row, res in enumerate(results):
        ticker    = res["ticker"]
        checklist = res["checklist"]
        verdict   = res["verdict"]
        trades    = res["trades"]

        ax_chk = axes[row][0]
        ax_eq  = axes[row][1]

        # --- Checklist panel ---
        ax_chk.set_facecolor("#0d1117")
        ax_chk.axis("off")

        # Verdict banner
        v_color = VERDICT_COLORS.get(verdict, "#888888")
        ax_chk.text(0.5, 0.97,
                    f"{ticker}  |  {verdict}",
                    transform=ax_chk.transAxes,
                    ha="center", va="top",
                    fontsize=18, fontweight="bold", color=v_color,
                    bbox=dict(facecolor=v_color + "22",
                              edgecolor=v_color,
                              boxstyle="round,pad=0.5"))

        # Each checklist item as a labelled row
        item_y_positions = np.linspace(0.88, 0.05, len(checklist))

        for i, (name, val, threshold, critical, passed) in enumerate(checklist):
            y = item_y_positions[i]

            # Background stripe
            stripe_color = "#0d2a0d" if passed else ("#2a0d0d" if critical else "#1f1f00")
            ax_chk.axhspan(y - 0.04, y + 0.04, xmin=0.02, xmax=0.98,
                           color=stripe_color, transform=ax_chk.transAxes, alpha=0.7)

            # Critical / advisory badge
            badge_col = "#ff5555" if critical else "#ffaa00"
            badge_txt = "CRITICAL" if critical else "advisory"
            ax_chk.text(0.04, y, badge_txt, transform=ax_chk.transAxes,
                        va="center", ha="left", fontsize=7.5,
                        color=badge_col, fontweight="bold")

            # Metric name
            ax_chk.text(0.25, y, name, transform=ax_chk.transAxes,
                        va="center", ha="left", fontsize=9, color="white")

            # Value
            if isinstance(val, float):
                if name in ("PSR", "DSR", "P(ruin<-20%)"):
                    val_str = f"{val:.1%}"
                elif name == "Trade Count":
                    val_str = str(int(val))
                else:
                    val_str = f"{val:+.4f}"
            else:
                val_str = str(val)

            ax_chk.text(0.63, y, val_str, transform=ax_chk.transAxes,
                        va="center", ha="right", fontsize=9, color="white")

            # Pass/Fail mark
            mark_color = "#44cc88" if passed else "#ff4444"
            mark_text  = "✓ PASS" if passed else "✗ FAIL"
            ax_chk.text(0.96, y, mark_text, transform=ax_chk.transAxes,
                        va="center", ha="right", fontsize=9,
                        color=mark_color, fontweight="bold")

        # Divider between critical and advisory
        ax_chk.plot([0.02, 0.98], [item_y_positions[4] - 0.05, item_y_positions[4] - 0.05],
                    color="#555577", lw=0.8, transform=ax_chk.transAxes)
        ax_chk.text(0.5, item_y_positions[4] - 0.055,
                    "─── Advisory ───",
                    transform=ax_chk.transAxes,
                    ha="center", va="top", fontsize=8, color="#888888")

        # Verdict reason
        ax_chk.text(0.5, 0.01, res["verdict_reason"],
                    transform=ax_chk.transAxes,
                    ha="center", va="bottom", fontsize=8, color="#aaaaaa",
                    style="italic")

        ax_chk.set_title(f"{ticker} — Pre-Trade Checklist", color="white",
                         fontsize=12, pad=10)

        # --- Equity curve panel ---
        ax_eq.set_facecolor("#0d1117")

        cum_ret = trades.cumsum().values * 100
        x       = np.arange(len(cum_ret))

        # Drawdown shading (fill below running max)
        running_max = np.maximum.accumulate(cum_ret)
        ax_eq.fill_between(x, cum_ret, running_max,
                           where=cum_ret < running_max,
                           alpha=0.3, color="#ff4444",
                           label="Drawdown")

        # Equity curve
        curve_color = VERDICT_COLORS.get(verdict, "#4488ff")
        ax_eq.plot(x, cum_ret, color=curve_color, lw=2.0, label="Cumulative Return")
        ax_eq.axhline(0, color="white", lw=0.8, ls=":")
        ax_eq.axhline(-20, color="#ff6600", lw=0.8, ls="--",
                      alpha=0.7, label="Ruin threshold (-20%)")

        # Mark max drawdown
        max_dd_val = res["max_dd"] * 100
        ax_eq.annotate(f"Max DD: {max_dd_val:.1f}%",
                       xy=(x[np.argmin(cum_ret - running_max)],
                           min(cum_ret)),
                       color="#ff4444", fontsize=9,
                       xytext=(len(x) * 0.5, max_dd_val - 5),
                       arrowprops=dict(arrowstyle="->", color="#ff4444"))

        ax_eq.set_title(f"{ticker} — Signal Equity Curve", color="white",
                        fontsize=12, pad=10)
        ax_eq.set_xlabel("Trade Number", color="#aaaaaa")
        ax_eq.set_ylabel("Cumulative Return (%)", color="#aaaaaa")
        ax_eq.tick_params(colors="#aaaaaa")
        ax_eq.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
        for spine in ax_eq.spines.values():
            spine.set_edgecolor("#333355")

    plt.suptitle("Pre-Trade Risk Checklist — Signal Approval Gate\n"
                 "Every signal must pass this before seeing real capital",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Chart saved → {save_path}")


# ============================================================
# SECTION 5 — CONCEPT SUMMARY
# ============================================================

CONCEPT_SUMMARY = """
============================================================
WHAT YOU LEARNED — PRE-TRADE CHECKLIST
============================================================

1. THE FIVE CRITICAL GATES (no exceptions)
   Gross > 0       → signal has EDGE before costs
   Net > 0         → edge SURVIVES costs
   IC > 0.05       → predictions TRACK outcomes
   PSR > 95%       → Sharpe is STATISTICALLY real
   MaxDD > -20%    → drawdown stays in INSTITUTIONAL tolerance

   If any critical gate fails: STOP. Do not deploy. Investigate.

2. ADVISORY GATES (note and monitor)
   DSR > 95%       → Sharpe not explained by trying many strategies
   N ≥ 50          → sample size is large enough to trust statistics
   P(ruin) < 5%    → Monte Carlo says ruin is rare at this position size

   Advisory failures mean: deploy SMALL (25% of target), collect more data.

3. THREE VERDICTS
   GO     → all critical pass + 2+ advisory pass → full deployment
   WATCH  → all critical pass + advisory failures → 25% pilot
   NO-GO  → any critical failure → do not deploy

4. WHY THE GATE EXISTS
   Research bias: we run 15 variants. One "works" by chance.
   If we skip PSR and DSR, we will deploy that lucky variant.
   The checklist is the circuit breaker between research and capital.

5. WHAT TO DO WITH A NO-GO
   IC fails: go back to feature engineering. The signal is not predictive.
   PSR fails: gather more data. 60 days is not enough for Sharpe confirmation.
   Net < 0: reduce cost (larger orders, different time-of-day, lower frequency).
   MaxDD fails: reduce position size OR add stop-loss to the backtest.

6. INTERVIEW LINE
   "Every signal we research goes through an 8-item pre-trade checklist:
    5 critical gates that are hard stops, and 3 advisory gates that
    determine sizing. In my NVDA research, the signal passed all critical
    gates but failed DSR — too few trades to beat the multiple testing bar.
    Action: QuantConnect 4.5-year backtest to gather sufficient history."

============================================================
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os, subprocess
    os.makedirs("charts", exist_ok=True)

    print(CONCEPT_SUMMARY)
    print("Running pre-trade checklist on NVDA and MSFT...")
    print("(Includes 500 Monte Carlo simulations — takes ~20 seconds per ticker)\n")

    tickers = ["NVDA", "MSFT"]
    results = []

    for ticker in tickers:
        res = run_checklist(ticker, k_trials=15)
        if res:
            results.append(res)

    if results:
        chart_path = "charts/pretrade_checklist.png"
        make_chart(results, save_path=chart_path)
        subprocess.Popen(["open", chart_path])

    # Final summary across tickers
    print("\n" + "="*60)
    print("  FINAL SUMMARY — ALL TICKERS")
    print("="*60)
    print(f"  {'Ticker':<8} {'Verdict':<8} {'IC':>8} {'PSR':>8} {'DSR':>8} {'P(ruin)':>9}")
    print(f"  {'-'*55}")
    for res in results:
        print(f"  {res['ticker']:<8} {res['verdict']:<8} "
              f"{res['ic']:>8.4f} {res['psr']:>8.1%} {res['dsr']:>8.1%} "
              f"{res['p_ruin']:>9.1%}")

    print("\n" + "="*60)
    print("  SENIOR LEVEL COMPLETE")
    print("="*60)
    print("""
  Files completed:
    04_PURGED_WALK_FORWARD.py   — label leakage prevention
    05_MONTE_CARLO.py           — path distribution analysis
    06_PORTFOLIO_OPTIMIZATION.py — min-var, max-sharpe, risk parity
    07_FACTOR_MODELING.py       — alpha vs factor beta attribution
    08_MICROSTRUCTURE.py        — Roll spread, Amihud, OFI, C-S
    09_EXECUTION_MODELS.py      — TWAP, VWAP, Almgren-Chriss
    10_PRETRADE_CHECKLIST.py    — 8-gate go/no-go signal approval

  Next: Update README → Push to GitHub → Interview preparation
""")
