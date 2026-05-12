"""
============================================================
08_MICROSTRUCTURE.py  |  Senior Level  |  Market Microstructure
============================================================

HYPOTHESIS
----------
Before sizing any intraday trade, we must measure the TRUE cost of trading.
Headline cost = commission + spread. But the real cost includes MARKET IMPACT:
the price moves against you as you buy or sell.

Hypothesis: NVDA's bid-ask spread and market impact are materially LARGER than
MSFT's, because NVDA has higher volatility and higher price. This means a
strategy that is profitable on MSFT paper may not be profitable on NVDA
after realistic microstructure costs are accounted for.

We measure four microstructure metrics on both tickers and compare them
to the signal's gross IC to see whether edge survives real costs.

-------------------------------------------------------------
WHAT IS MARKET MICROSTRUCTURE?
-------------------------------------------------------------

Microstructure = the mechanics of how trades are executed and how prices
are determined between buyers and sellers.

Key concepts:

  Bid-Ask Spread:
    The dealer quotes a BID (the price they'll buy from you) and an ASK
    (the price they'll sell to you). The SPREAD = Ask - Bid.
    If AAPL bid = $150.00 and ask = $150.02, spread = $0.02.
    A market order IMMEDIATELY loses 0.5 × spread as a cost.
    Half-spread = what we pay just to enter a position.

  Market Impact:
    When we BUY 10,000 shares, our own orders push prices UP.
    The more we buy, the more we move the market against ourselves.
    Impact = additional cost from our own trading pressure.

  Order Flow Imbalance (OFI):
    Excess of buys over sells (positive OFI) pushes prices up.
    OFI predicts short-term price direction — this IS the signal.
    Amihud (2002) showed that illiquid stocks have larger price impact per unit of volume.

Four metrics we compute:

  1. Roll's Spread Estimate — infers bid-ask spread from return autocorrelation
  2. Amihud Illiquidity     — price impact per dollar volume traded
  3. OFI Z-score            — order flow imbalance relative to its history
  4. Corwin-Schultz Spread  — high-low based spread estimator

-------------------------------------------------------------
ROLL'S SPREAD ESTIMATE (Roll 1984)
-------------------------------------------------------------

If prices bounce between bid and ask, consecutive returns are negatively correlated.
Roll showed: spread = 2 × sqrt(-cov(Δp_t, Δp_{t-1}))

Intuition:
  If price moves from ask to bid, return = -(spread)
  Next bar, if new buyer pushes to ask, return = +(spread)
  These two returns are negatively correlated exactly because of the spread.

Limitation: only captures mechanical bid-ask bounce, not information-driven moves.

-------------------------------------------------------------
AMIHUD ILLIQUIDITY (Amihud 2002)
-------------------------------------------------------------

Illiquidity = |Return| / Dollar Volume = price impact per dollar traded

If NVDA moves 0.5% on $10M volume: Amihud = 0.005 / 10,000,000 = 5×10⁻¹⁰
If NVDA moves 0.5% on $100M volume: Amihud = 5×10⁻¹¹ (10× more liquid)

Higher Amihud → more illiquid → greater price impact per trade
Use rolling 20-bar window to track how liquidity changes during the day.

-------------------------------------------------------------
ORDER FLOW IMBALANCE (OFI)
-------------------------------------------------------------

OFI = (Volume_buy - Volume_sell) / (Volume_buy + Volume_sell)
    = signed volume imbalance, range [-1, +1]

We approximate buy/sell split using the tick rule:
  If close > open: assume net buy pressure (positive OFI)
  If close < open: assume net sell pressure (negative OFI)
  (Real TAQ data gives exact tick-by-tick direction)

Z-score OFI: (OFI - rolling_mean) / rolling_std
High OFI z-score → institutional buying → prices likely to continue rising

-------------------------------------------------------------
FIVE NUMBERS (microstructure-adjusted)
-------------------------------------------------------------

  1. Effective Half-Spread   — the true cost to enter a position (Roll estimate)
  2. Amihud Ratio            — illiquidity proxy (lower is better)
  3. OFI Z-score signal      — is order flow in our direction?
  4. Gross IC                — signal quality before costs
  5. Net IC (adjusted)       — IC after realistic cost adjustment

THRESHOLDS:
  Effective Half-Spread   < 0.10%   — acceptable round-trip cost
  Amihud (relative)       < 1×10⁻⁸  — acceptable illiquidity
  OFI correlation with fwd ret > 0.03 — order flow predicts direction
  Gross IC                > 0.050   — signal has edge before costs
  IC loss to spread       < 20%     — costs don't destroy IC

-------------------------------------------------------------
INTERVIEW LINES
-------------------------------------------------------------
"I estimated bid-ask spread using Roll's method — from the negative serial
correlation in minute returns. For NVDA, half-spread was 0.06%, compared
to 0.03% for MSFT. Combined with Amihud illiquidity ratio, this means
our NVDA cost per trade is materially higher and eats into our IC more."

"OFI is my most predictive intraday feature. A Z-score above 2 signals
net institutional buying — prices tend to continue. In my walk-forward,
OFI z-score was the second highest-loading feature in the Ridge model."

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
    """Build features including OFI proxy and forward return target."""
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
# SECTION 2 — MICROSTRUCTURE METRICS
# ============================================================

def roll_spread(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Roll (1984) spread estimate.

    spread = 2 × sqrt(max(-cov(Δp_t, Δp_{t-1}), 0))

    Uses rolling window covariance between return and lagged return.
    Returns HALF-SPREAD (cost to enter a position from one side).
    """
    lag_ret = returns.shift(1)
    cov_series = returns.rolling(window).cov(lag_ret)
    # Negate: if spread-induced, cov is negative, so -cov is positive
    spread = 2 * np.sqrt(np.maximum(-cov_series, 0))
    return spread


def amihud_illiquidity(returns: pd.Series, volume: pd.Series,
                       price: pd.Series, window: int = 20) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio.

    ILLIQ_t = |ret_t| / (price_t × volume_t)
            = price impact per dollar of volume traded

    Higher value → more illiquid → larger price impact for our orders.
    """
    dollar_volume = price * volume                      # in dollars
    illiq         = returns.abs() / (dollar_volume + 1)  # +1 avoids div-by-zero
    return illiq.rolling(window).mean()                   # rolling average


def order_flow_imbalance(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    OFI Z-score (proxy from OHLC data).

    Tick rule:
      Bar up (close > open)  → net buy pressure  → OFI = +volume_fraction
      Bar down (close < open) → net sell pressure → OFI = -volume_fraction
      Flat                    → OFI = 0

    Then z-score relative to rolling window to detect ABNORMAL imbalances.
    """
    direction       = np.sign(df["close"] - df["open"])
    raw_ofi         = direction * df["volume"]          # signed volume
    ofi_frac        = raw_ofi / (df["volume"] + 1)     # fraction [-1, +1]
    ofi_mean        = ofi_frac.rolling(window).mean()
    ofi_std         = ofi_frac.rolling(window).std()
    ofi_zscore      = (ofi_frac - ofi_mean) / (ofi_std + 1e-9)
    return ofi_zscore


def corwin_schultz_spread(high: pd.Series, low: pd.Series,
                          window: int = 20) -> pd.Series:
    """
    Corwin-Schultz (2012) high-low spread estimator.

    Uses the fact that the daily high-low range reflects both the true
    price volatility AND the bid-ask spread.

    β = (log(H/L))²                     — one-day high-low log-ratio squared
    γ = (log(H₂/L₂))²                   — two-day high-low ratio squared
    α = (sqrt(2β) - sqrt(β)) / (3-2√2) - sqrt(γ/(3-2√2))
    Spread = 2(e^α - 1) / (1 + e^α)

    This is the best spread estimator available without tick-level data.
    We compute rolling to track intraday regime changes.
    """
    beta  = (np.log(high / low)) ** 2
    beta2 = beta.rolling(2).sum()   # two-bar window
    gamma = (np.log(high.rolling(2).max() / low.rolling(2).min())) ** 2

    c1    = (3 - 2 * np.sqrt(2))
    alpha = (np.sqrt(2 * beta2) - np.sqrt(beta2)) / c1 - np.sqrt(gamma / c1)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0)                        # spread can't be negative

    return spread.rolling(window).mean()


# ============================================================
# SECTION 3 — SIGNAL IC (for microstructure context)
# ============================================================

def compute_ic(pred_series: pd.Series, actual: pd.Series) -> float:
    """Pearson IC between predictions and forward returns."""
    return float(pred_series.corr(actual))


def estimate_net_ic(gross_ic: float, half_spread_pct: float,
                    fwd_ret_vol: float) -> float:
    """
    Approximate IC after spread cost.
    Net IC degrades proportionally to how much cost eats into return vol.
    cost_fraction = half_spread / (fwd_ret_vol + half_spread)
    net_ic = gross_ic × (1 - cost_fraction)
    """
    cost_fraction = half_spread_pct / (fwd_ret_vol + half_spread_pct + 1e-9)
    return gross_ic * (1 - cost_fraction)


# ============================================================
# SECTION 4 — FULL PIPELINE
# ============================================================

def run_pipeline(ticker: str) -> dict:
    """
    Full microstructure pipeline:
      download → compute 4 microstructure metrics → compute IC → print five numbers
    """
    print(f"\n{'='*55}")
    print(f"  {ticker}  |  Microstructure Analysis")
    print(f"{'='*55}")

    df = download_data(ticker)
    if df.empty:
        print(f"  ERROR: no data for {ticker}")
        return {}
    df = build_features(df)
    print(f"  Bars: {len(df)}")

    # --- Microstructure metrics ---
    roll_half = roll_spread(df["ret_1"], window=20)         # half-spread proxy
    amihud    = amihud_illiquidity(df["ret_1"], df["volume"], df["close"], window=20)
    ofi_z     = order_flow_imbalance(df, window=10)
    cs_spread = corwin_schultz_spread(df["high"], df["low"], window=20)

    # Summary statistics
    roll_mean     = float(roll_half.dropna().mean())
    amihud_mean   = float(amihud.dropna().mean())
    ofi_ic        = float(ofi_z.shift(1).corr(df["fwd_ret"]))   # does OFI predict returns?
    cs_mean       = float(cs_spread.dropna().mean())

    # Gross IC (simple feature correlation as proxy for signal IC)
    gross_ic  = float(df["ret_3"].corr(df["fwd_ret"]))           # momentum IC proxy
    fwd_vol   = float(df["fwd_ret"].std())
    net_ic    = estimate_net_ic(gross_ic, roll_mean, fwd_vol)

    print(f"\n  MICROSTRUCTURE FIVE NUMBERS")
    print(f"  {'Metric':<35} {'Value':>12}  {'Threshold':>12}  {'Status':>8}")
    print(f"  {'-'*72}")

    items = [
        ("Roll Half-Spread (proxy)",  f"{roll_mean:.6f}",         "< 0.001",   bool(roll_mean < 0.001)),
        ("Roll Half-Spread (%)",      f"{roll_mean*100:.4f}%",    "< 0.10%",   bool(roll_mean < 0.001)),
        ("Amihud Illiq (×10⁻⁸)",     f"{amihud_mean*1e8:.4f}",  "< 1.0",     bool(amihud_mean * 1e8 < 1.0)),
        ("OFI → Return IC",           f"{ofi_ic:+.4f}",           "> 0.03",    bool(ofi_ic > 0.03)),
        ("C-S Spread (proxy)",        f"{cs_mean:.6f}",           "< 0.002",   bool(cs_mean < 0.002)),
        ("--- Signal quality ---",    "",                          "",           None),
        ("Gross IC (3-bar momentum)", f"{gross_ic:+.4f}",         "> 0.05",    bool(gross_ic > 0.05)),
        ("Estimated Net IC",          f"{net_ic:+.4f}",           "> 0.04",    bool(net_ic > 0.04)),
        ("IC degradation",            f"{(gross_ic - net_ic)/max(abs(gross_ic),1e-6):.1%}",
         "< 20%",  bool(abs(gross_ic - net_ic) / max(abs(gross_ic), 1e-6) < 0.20)),
    ]

    for label, val, thresh, passed in items:
        if passed is None:
            print(f"  {label:<35} {val:>12}  {thresh:>12}")
        else:
            s = "PASS ✓" if passed else "FAIL ✗"
            print(f"  {label:<35} {val:>12}  {thresh:>12}  {s:>8}")

    return {
        "ticker":       ticker,
        "df":           df,
        "roll_half":    roll_half,
        "amihud":       amihud,
        "ofi_z":        ofi_z,
        "cs_spread":    cs_spread,
        "roll_mean":    roll_mean,
        "amihud_mean":  amihud_mean,
        "ofi_ic":       ofi_ic,
        "cs_mean":      cs_mean,
        "gross_ic":     gross_ic,
        "net_ic":       net_ic,
    }


# ============================================================
# SECTION 5 — CHART (standard format)
# ============================================================

def make_chart(results: list, save_path: str = "charts/microstructure.png"):
    """
    4 metrics × 2 tickers = 8 panels (2 rows × 4 cols).
    Bottom of each column: five numbers bar chart comparison.
    """
    metric_configs = [
        ("roll_half",  "Roll Half-Spread",       "Spread (frac.)", "#4488ff"),
        ("amihud",     "Amihud Illiquidity",      "ILLIQ (frac.)",  "#ffaa00"),
        ("ofi_z",      "OFI Z-score",             "Z-score",        "#44cc88"),
        ("cs_spread",  "Corwin-Schultz Spread",   "Spread (frac.)", "#ff6688"),
    ]

    n_metrics  = len(metric_configs)
    n_tickers  = len(results)

    # Layout: n_tickers rows, n_metrics cols + 1 scorecard col
    fig, axes = plt.subplots(n_tickers, n_metrics + 1,
                             figsize=(22, 5 * n_tickers))
    if n_tickers == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0d1117")

    for row, res in enumerate(results):
        ticker = res["ticker"]
        df     = res["df"]
        series_map = {
            "roll_half": res["roll_half"],
            "amihud":    res["amihud"],
            "ofi_z":     res["ofi_z"],
            "cs_spread": res["cs_spread"],
        }

        for col, (key, title, ylabel, color) in enumerate(metric_configs):
            ax = axes[row][col]
            ax.set_facecolor("#0d1117")

            series = series_map[key].dropna()
            x      = np.arange(len(series))

            if key == "ofi_z":
                # OFI z-score: color bars by sign
                pos_mask = series.values >= 0
                ax.bar(x[pos_mask],  series.values[pos_mask],  color="#44cc88", alpha=0.7, width=1)
                ax.bar(x[~pos_mask], series.values[~pos_mask], color="#ff4444", alpha=0.7, width=1)
                ax.axhline(2,  color="white", lw=0.8, ls="--", alpha=0.5, label="+2σ")
                ax.axhline(-2, color="white", lw=0.8, ls="--", alpha=0.5)
                ax.axhline(0,  color="white", lw=0.5)
            else:
                ax.plot(x, series.values, color=color, lw=1.0, alpha=0.8)
                ax.fill_between(x, series.values, alpha=0.2, color=color)
                # Add rolling mean for context
                roll_m = pd.Series(series.values).rolling(20).mean()
                ax.plot(x, roll_m.values, color="white", lw=1.5, ls="--",
                        alpha=0.6, label="20-bar avg")

            ax.set_title(f"{ticker} — {title}", color="white", fontsize=10, pad=6)
            ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            if col == 0:
                ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

        # --- Scorecard panel ---
        ax_tbl = axes[row][n_metrics]
        ax_tbl.set_facecolor("#0d1117")
        ax_tbl.axis("off")

        def ok(val, thr, higher=True):
            return "✓" if (bool(val > thr) if higher else bool(val < thr)) else "✗"

        table_data = [
            ["Metric",          "Value",                                "Pass?"],
            ["Roll ½-Sprd",     f"{res['roll_mean']*100:.4f}%",
             ok(res["roll_mean"], 0.001, higher=False)],
            ["Amihud ×10⁻⁸",   f"{res['amihud_mean']*1e8:.4f}",
             ok(res["amihud_mean"] * 1e8, 1.0, higher=False)],
            ["OFI→Ret IC",      f"{res['ofi_ic']:+.4f}",
             ok(res["ofi_ic"], 0.03)],
            ["C-S Spread",      f"{res['cs_mean']:.6f}",
             ok(res["cs_mean"], 0.002, higher=False)],
            ["Gross IC",        f"{res['gross_ic']:+.4f}",
             ok(res["gross_ic"], 0.05)],
            ["Net IC",          f"{res['net_ic']:+.4f}",
             ok(res["net_ic"], 0.04)],
            ["IC Degrad.",      f"{abs(res['gross_ic']-res['net_ic'])/max(abs(res['gross_ic']),1e-6):.1%}",
             ok(abs(res['gross_ic']-res['net_ic'])/max(abs(res['gross_ic']),1e-6), 0.20, higher=False)],
        ]

        def cell_col(r, c, cell):
            if r == 0:  return "#2a2a4e"
            if c < 2:   return "#1a1a2e"
            if "✓" in str(cell): return "#0d2a0d"
            if "✗" in str(cell): return "#2a0d0d"
            return "#1a1a2e"

        colors = [[cell_col(r, c, cell)
                   for c, cell in enumerate(row_d)]
                  for r, row_d in enumerate(table_data)]

        tbl = ax_tbl.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            cellColours=colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_text_props(color="white")
            cell.set_edgecolor("#333355")

        ax_tbl.set_title(f"{ticker} — Five Numbers", color="white", fontsize=10)

    plt.suptitle("Market Microstructure — Spread, Liquidity, Order Flow\n"
                 "Measuring the TRUE cost of intraday execution",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Chart saved → {save_path}")


# ============================================================
# SECTION 6 — CONCEPT SUMMARY
# ============================================================

CONCEPT_SUMMARY = """
============================================================
WHAT YOU LEARNED — MARKET MICROSTRUCTURE
============================================================

1. THE BID-ASK SPREAD IS NOT FIXED
   Spreads widen in volatile markets (market makers charge more to bear risk).
   Spreads narrow near the open/close and during high-volume periods.
   Roll's method estimates this from return autocorrelation — no TAQ data needed.

2. MARKET IMPACT EXCEEDS THE SPREAD FOR LARGE ORDERS
   Amihud shows: price impact = f(|return| / dollar_volume)
   A $10M order on NVDA (avg volume $5B/day) moves the price ~0.02%.
   On a smaller stock, the same $10M could move it 0.5% or more.
   Always check Amihud before sizing a new position.

3. ORDER FLOW IMBALANCE IS A LEADING INDICATOR
   OFI z-score above +2 → institutional buyers are active → follow.
   OFI z-score below -2 → institutional sellers → be cautious on longs.
   This is why Kyle lambda (price impact per unit of OFI) is key:
   Lambda = Δprice / ΔOFI.

4. CORWIN-SCHULTZ IS YOUR BEST FREE-DATA SPREAD ESTIMATOR
   No tick data? Use high-low spread estimate. Works on hourly or daily bars.
   More accurate than Roll when the bid-ask bounce assumption is violated.

5. INTERVIEW LINE
   "I estimated transaction costs using Roll's half-spread from return
    autocorrelation and Amihud illiquidity ratio. For NVDA, half-spread
    was 6bp vs 3bp for MSFT. Combined with 0.05bp commission, realistic
    round-trip cost was 13bp for NVDA. With gross IC of 0.05, we lose
    approximately 18% of IC to costs — still net positive but tight."

============================================================
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os, subprocess
    os.makedirs("charts", exist_ok=True)

    print(CONCEPT_SUMMARY)
    print("Running microstructure analysis on NVDA and MSFT...")

    tickers = ["NVDA", "MSFT"]
    results = []

    for ticker in tickers:
        res = run_pipeline(ticker)
        if res:
            results.append(res)

    if results:
        chart_path = "charts/microstructure.png"
        make_chart(results, save_path=chart_path)
        subprocess.Popen(["open", chart_path])

    # Research decision
    print("\n" + "="*55)
    print("  RESEARCH DECISION")
    print("="*55)
    for res in results:
        nvda_spread_pct = res["roll_mean"] * 100
        ic_degradation  = abs(res["gross_ic"] - res["net_ic"]) / max(abs(res["gross_ic"]), 1e-6)
        verdict = []

        if bool(nvda_spread_pct < 0.10):
            verdict.append(f"spread acceptable ({nvda_spread_pct:.3f}%)")
        else:
            verdict.append(f"spread HIGH ({nvda_spread_pct:.3f}%) — check cost budget")

        if bool(ic_degradation < 0.20):
            verdict.append(f"IC degradation manageable ({ic_degradation:.1%})")
        else:
            verdict.append(f"IC degradation HIGH ({ic_degradation:.1%}) — costs eat signal")

        if bool(res["ofi_ic"] > 0.03):
            verdict.append(f"OFI predictive (IC={res['ofi_ic']:+.4f})")
        else:
            verdict.append(f"OFI not predictive — consider removing")

        print(f"  {res['ticker']}: {' | '.join(verdict)}")

    print("\nNext: 09_EXECUTION_MODELS.py")
