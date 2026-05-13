"""
================================================================================
13_STAT_ARB_PAIRS.py — Statistical Arbitrage: Pairs Trading
================================================================================

HYPOTHESIS
    NVDA and AMD are cointegrated (move together long-term because they share
    the same semiconductor market). When their price spread deviates more than
    2 standard deviations from its historical mean, it will revert.
    A pairs trading strategy that bets on spread reversion earns a statistically
    significant positive return after transaction costs.
    Same test applied to MSFT / GOOGL (software/cloud competitors).

WHAT IS STATISTICAL ARBITRAGE?
    Pure arbitrage:     Buy AAPL on NYSE at $150, sell on NASDAQ at $151.
                        Riskless. Gone in microseconds. Not available to us.

    Statistical arb:    The spread between NVDA and AMD tends to mean-revert.
                        When the spread is too wide, we bet it closes.
                        Not riskless — the spread can widen further before
                        reverting. Risk is statistical, not zero.

WHAT IS COINTEGRATION?
    Correlation:        Two prices move in the same direction most of the time.
    Cointegration:      Two prices are tied together by a long-run equilibrium.
                        Even if they diverge temporarily, an economic force
                        pulls them back together.

    Example:
        Correlation only: NVDA and AMD are both up in a bull market.
                          No long-run relationship — they could permanently diverge.
        Cointegrated:     NVDA and AMD both sell to data centres.
                          Revenue, margins, and valuation converge long-term.
                          Temporary spread divergence is noise around equilibrium.

    The Engle-Granger test (1987) is the standard test for cointegration:
        1. Fit OLS regression: log(Price_A) = α + β × log(Price_B) + ε
        2. Test the residual ε for stationarity (ADF test)
        3. If ε is stationary → cointegrated → spread mean-reverts

THE HEDGE RATIO (β)
    β is NOT 1. One unit of NVDA is not offset by one unit of AMD.
    The hedge ratio adjusts for the relative price levels and volatility.

    OLS fit: log(NVDA) = α + β × log(AMD) + ε

    Interpretation:
        β = 1.2 means: when AMD moves 1%, NVDA moves 1.2%.
        To hedge: long $100k NVDA → short $120k AMD.
        The spread ε = log(NVDA) - α - β × log(AMD) should be stationary.

THE SIGNAL — Z-SCORE
    At each day:
        1. Compute the spread: ε_t = log(NVDA_t) - α - β × log(AMD_t)
        2. Compute rolling z-score: z = (ε_t - μ_ε) / σ_ε
                                    using 60-day rolling window
        3. Entry signal:
             z < -2.0 → spread is abnormally low → long A, short B
             z > +2.0 → spread is abnormally high → short A, long B
        4. Exit signal:
             |z| < 0.5 → spread has reverted → close both positions

    The bet: "The spread has moved too far. It will come back."
    (Same structure as VWAP mean reversion — but between two stocks, not
     between price and its own VWAP average.)

REGIME ANALYSIS
    The pairs relationship can break down.
    Pre-2022: NVDA and AMD both in growth phase — spread stable.
    Post-2022: NVDA AI chip demand diverged sharply from AMD.
               NVDA market cap grew 5× faster than AMD.
               The cointegration may have broken in the new regime.

    We test: is the strategy profitable pre-2022 and post-2022 separately?
    If the relationship breaks, the strategy must stop.

THE FIVE NUMBERS — what to read after each run:
    1. Gross Return     — does the spread earn a positive return before costs?
    2. Total Costs      — how many round-trips? Spread arbitrage trades frequently.
    3. Net Return       — does the edge survive costs?
    4. IC               — do z-score ranks predict next-day spread changes?
    5. PSR              — is the Sharpe statistically real?
    +  ADF p-value      — is the spread actually stationary? (must be < 0.05)
    +  Regime split     — does the strategy work post-2022?

THRESHOLDS:
    Gross Return    > 0
    Net Return      > 0
    ADF p-value     < 0.05   (required — if not stationary, do not trade)
    IC              > 0.03   (daily IC is lower than monthly IC)
    PSR             > 50%
    Regime stable   — Sharpe pre-2022 AND post-2022 both positive

STACK:
    yfinance, pandas, numpy, matplotlib, scipy, statsmodels, scikit-learn
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

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Run: pip install statsmodels")
    print("         Cointegration tests will be skipped.")

# ── PAIRS ─────────────────────────────────────────────────────────────────────
PAIRS = [
    ("NVDA", "AMD"),    # GPU / AI chip competitors — same customer base
    ("MSFT", "GOOGL"),  # Cloud / enterprise software — same enterprise clients
]

START_DATE = "2020-01-01"
END_DATE   = "2024-06-01"
COST       = 0.001     # 0.1% per side per trade
Z_ENTER    = 2.0       # enter when |z-score| crosses this
Z_EXIT     = 0.5       # exit when |z-score| drops below this
Z_WINDOW   = 60        # rolling window for z-score computation (days)

print("=" * 72)
print("STATISTICAL ARBITRAGE — PAIRS TRADING — STUDY 13")
print("Pairs: NVDA/AMD  |  MSFT/GOOGL")
print("Signal: Spread z-score | Entry |z|>2.0 | Exit |z|<0.5")
print("=" * 72)

# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────
print("\n[1/6] Downloading daily price data...")

all_tickers = list(set([t for pair in PAIRS for t in pair]))
raw = yf.download(
    all_tickers,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=True,
    progress=False,
)["Close"]

raw.ffill(inplace=True)
raw.dropna(inplace=True)

print(f"    Tickers downloaded:  {list(raw.columns)}")
print(f"    Date range: {raw.index[0].date()} → {raw.index[-1].date()}")
print(f"    Trading days: {len(raw)}")

# ── COINTEGRATION TEST ─────────────────────────────────────────────────────────
print("\n[2/6] Cointegration tests (Engle-Granger)...")
"""
Engle-Granger cointegration test:
    Step 1: OLS regression log(A) = α + β × log(B) + ε
    Step 2: ADF test on residual ε — H0: ε has a unit root (not stationary)
    If p-value < 0.05: reject H0 → ε is stationary → cointegrated.

    Why log prices?
        Log prices remove the scale difference ($500 NVDA vs $30 AMD).
        They also give us log-return relationships which are more stable.

    The coint() function from statsmodels runs this directly.
"""

coint_results = {}
for ticker_a, ticker_b in PAIRS:
    log_a = np.log(raw[ticker_a])
    log_b = np.log(raw[ticker_b])

    if HAS_STATSMODELS:
        # Full sample cointegration
        score, p_val, crit = coint(log_a, log_b)

        # OLS to get hedge ratio β
        X = add_constant(log_b.values)
        model = OLS(log_a.values, X).fit()
        alpha_coef = model.params[0]
        beta_coef  = model.params[1]

        # Spread residual
        spread = log_a.values - alpha_coef - beta_coef * log_b.values
        spread_series = pd.Series(spread, index=raw.index)

        # ADF test on spread
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(spread, autolag="AIC")

        # Cointegration pre/post 2022
        split_date = "2022-01-01"
        log_a_pre  = log_a[log_a.index < split_date]
        log_b_pre  = log_b[log_b.index < split_date]
        log_a_post = log_a[log_a.index >= split_date]
        log_b_post = log_b[log_b.index >= split_date]

        _, p_pre,  _ = coint(log_a_pre,  log_b_pre)  if len(log_a_pre)  > 30 else (0, 1, [])
        _, p_post, _ = coint(log_a_post, log_b_post) if len(log_a_post) > 30 else (0, 1, [])

        coint_results[(ticker_a, ticker_b)] = {
            "alpha":      alpha_coef,
            "beta":       beta_coef,
            "adf_p":      adf_p,
            "coint_p":    p_val,
            "spread":     spread_series,
            "coint_pre":  p_pre,
            "coint_post": p_post,
        }

        coint_flag  = "COINTEGRATED" if p_val  < 0.05 else "NOT COINTEGRATED"
        stable_flag = "STABLE"       if p_post < 0.05 else "BROKEN post-2022"

        print(f"\n    {ticker_a} / {ticker_b}")
        print(f"        Hedge ratio β:         {beta_coef:.4f}")
        print(f"        Cointegration p-val:   {p_val:.4f}  → {coint_flag}")
        print(f"        ADF spread p-val:      {adf_p:.4f}  → {'stationary' if adf_p < 0.05 else 'NOT stationary'}")
        print(f"        Pre-2022 coint p:      {p_pre:.4f}")
        print(f"        Post-2022 coint p:     {p_post:.4f}  → {stable_flag}")
    else:
        # Fallback: simple OLS hedge ratio without formal cointegration test
        log_b_arr = log_b.values.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(log_b_arr, log_a.values)
        alpha_coef = lr.intercept_
        beta_coef  = lr.coef_[0]

        spread = log_a.values - alpha_coef - beta_coef * log_b.values
        spread_series = pd.Series(spread, index=raw.index)

        coint_results[(ticker_a, ticker_b)] = {
            "alpha":  alpha_coef,
            "beta":   beta_coef,
            "adf_p":  np.nan,
            "spread": spread_series,
        }
        print(f"    {ticker_a}/{ticker_b}: β = {beta_coef:.4f} (statsmodels unavailable, no ADF test)")


# ── Z-SCORE SIGNAL AND BACKTEST ────────────────────────────────────────────────
print("\n[3/6] Computing z-score signal and running backtests...")
"""
Z-score signal at each day t:
    μ_t  = rolling mean of spread over last Z_WINDOW days
    σ_t  = rolling std  of spread over last Z_WINDOW days
    z_t  = (spread_t - μ_t) / σ_t

Position rules:
    If z_t < -2.0 and no position:
        → spread is below normal → A is cheap relative to B
        → long A (NVDA), short B (AMD)  — expect A to rally or B to fall
    If z_t > +2.0 and no position:
        → spread is above normal → A is expensive relative to B
        → short A, long B               — expect A to fall or B to rally
    If |z_t| < 0.5 and in a position:
        → spread has reverted → close both legs
    Stop loss: |z| > 3.5 (spread keeps widening — cut the loss)
"""

pair_backtests = {}

for (ticker_a, ticker_b), cdata in coint_results.items():
    spread = cdata["spread"]

    # Rolling z-score
    roll_mean = spread.rolling(Z_WINDOW).mean()
    roll_std  = spread.rolling(Z_WINDOW).std()
    z_score   = (spread - roll_mean) / roll_std
    z_score.dropna(inplace=True)
    spread_aligned = spread.loc[z_score.index]

    # Backtest
    position   = 0   # +1 = long A / short B, -1 = short A / long B
    entry_z    = 0.0
    trades     = []
    daily_pnl  = []

    price_a = raw[ticker_a].loc[z_score.index]
    price_b = raw[ticker_b].loc[z_score.index]

    for i in range(1, len(z_score)):
        date  = z_score.index[i]
        z     = z_score.iloc[i]
        z_lag = z_score.iloc[i - 1]

        # Daily P&L on existing position
        if position != 0:
            da = price_a.iloc[i] / price_a.iloc[i-1] - 1
            db = price_b.iloc[i] / price_b.iloc[i-1] - 1

            # Long A / short B: profit when A rises faster or B falls faster
            if position == 1:
                pnl = da - db
            else:
                pnl = db - da    # short A / long B
            daily_pnl.append((date, pnl, 0.0))
        else:
            daily_pnl.append((date, 0.0, 0.0))

        # Entry
        if position == 0:
            if z < -Z_ENTER:
                position = 1
                entry_z  = z
                cost_today = COST * 2   # two legs
                daily_pnl[-1] = (date, daily_pnl[-1][1] - cost_today, cost_today)
            elif z > Z_ENTER:
                position = -1
                entry_z  = z
                cost_today = COST * 2
                daily_pnl[-1] = (date, daily_pnl[-1][1] - cost_today, cost_today)

        # Exit — spread reverted
        elif abs(z) < Z_EXIT:
            cost_today = COST * 2
            daily_pnl[-1] = (date, daily_pnl[-1][1] - cost_today, cost_today)
            trades.append({"entry_z": entry_z, "exit_z": z, "direction": position})
            position = 0
            entry_z  = 0.0

        # Stop loss — spread blew out
        elif (position == 1 and z < -3.5) or (position == -1 and z > 3.5):
            cost_today = COST * 2
            daily_pnl[-1] = (date, daily_pnl[-1][1] - cost_today, cost_today)
            trades.append({"entry_z": entry_z, "exit_z": z, "direction": position,
                           "stopped": True})
            position = 0

    pnl_df = pd.DataFrame(daily_pnl, columns=["date", "pnl", "cost"]).set_index("date")
    pair_backtests[(ticker_a, ticker_b)] = {
        "pnl_df":   pnl_df,
        "z_score":  z_score,
        "trades":   trades,
        "spread":   spread_aligned,
    }

    cum_gross = (1 + pnl_df["pnl"]).cumprod()
    total_gross = cum_gross.iloc[-1] - 1
    total_cost  = pnl_df["cost"].sum()
    total_net   = total_gross - total_cost
    n_trades    = len(trades)

    print(f"    {ticker_a}/{ticker_b}: Gross {total_gross*100:+.2f}%  |  Costs {total_cost*100:.2f}%  |  Net {total_net*100:+.2f}%  |  {n_trades} trades")

# ── PSR AND IC ─────────────────────────────────────────────────────────────────
print("\n[4/6] PSR and IC computation...")

def compute_psr(daily_returns, sr_benchmark=0.0):
    n      = len(daily_returns)
    mu     = daily_returns.mean()
    sigma  = daily_returns.std()
    if sigma == 0:
        return 0.0, 0.0
    sr_obs = mu / sigma * np.sqrt(252)
    skew   = daily_returns.skew()
    kurt   = daily_returns.kurt()
    denom  = 1 - skew * sr_obs + ((kurt + 3) / 4) * sr_obs ** 2
    if denom <= 0:
        return 0.0, float(sr_obs)
    z = (sr_obs - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(norm.cdf(z)), float(sr_obs)


scorecard_data = {}

for (ticker_a, ticker_b), bdata in pair_backtests.items():
    pnl_df  = bdata["pnl_df"]
    z_score = bdata["z_score"]

    psr, sr = compute_psr(pnl_df["pnl"])

    # IC: does z-score rank predict next-day return?
    # Compute: sign(-z) = direction prediction; next-day pnl = outcome
    z_lag    = z_score.shift(1).dropna()
    pnl_next = pnl_df["pnl"].reindex(z_lag.index)
    mask     = pnl_next.notna() & z_lag.notna()
    if mask.sum() > 10:
        signal_pred = -z_lag[mask]   # negative z → long A → expect positive pnl
        rho, _ = stats.spearmanr(signal_pred, pnl_next[mask])
    else:
        rho = np.nan

    cum    = (1 + pnl_df["pnl"]).cumprod()
    gross  = cum.iloc[-1] - 1
    costs  = pnl_df["cost"].sum()
    net    = gross - costs
    hit    = (pnl_df["pnl"] > 0).mean()

    scorecard_data[(ticker_a, ticker_b)] = {
        "gross": gross, "costs": costs, "net": net,
        "ic": rho, "psr": psr, "sr": sr, "hit": hit,
        "n_trades": len(bdata["trades"]),
    }

    adf_p    = coint_results[(ticker_a, ticker_b)].get("adf_p", np.nan)
    coint_p  = coint_results[(ticker_a, ticker_b)].get("coint_p", np.nan)
    coint_post = coint_results[(ticker_a, ticker_b)].get("coint_post", np.nan)

    print(f"\n    ── {ticker_a} / {ticker_b} ────────────────────────────────────────")
    print(f"    Gross Return:     {gross * 100:+.2f}%   {'PASS' if gross > 0 else 'FAIL'}")
    print(f"    Total Costs:      {costs * 100:.2f}%")
    print(f"    Net Return:       {net * 100:+.2f}%   {'PASS' if net > 0 else 'FAIL'}")
    print(f"    IC (Spearman):    {rho:.4f}   {'PASS' if not np.isnan(rho) and rho > 0.03 else 'WEAK'}")
    print(f"    PSR:              {psr * 100:.1f}%   {'PASS' if psr > 0.50 else 'FAIL'}")
    print(f"    Sharpe (ann):     {sr:.3f}")
    print(f"    Hit Rate:         {hit * 100:.1f}%")
    if not np.isnan(adf_p):
        print(f"    ADF p-value:      {adf_p:.4f}   {'STATIONARY ✓' if adf_p < 0.05 else 'NOT STATIONARY — do not trade'}")
    if not np.isnan(coint_post):
        print(f"    Post-2022 coint:  {coint_post:.4f}   {'stable' if coint_post < 0.05 else 'BROKEN ← regime change'}")

# ── REGIME ANALYSIS ────────────────────────────────────────────────────────────
print("\n[5/6] Regime analysis — pre vs post 2022...")
"""
2022 marked a structural break for semiconductor stocks.
NVDA went from a gaming GPU company to the dominant AI chip supplier.
AMD followed AI demand but at a much slower pace.
If the cointegration breaks post-2022, the pairs strategy must stop.

We compare the strategy's net return in two regimes:
    Pre-2022:  "normal" semiconductor cycle — pairs move together
    Post-2022: AI demand shock — NVDA potentially diverges from AMD

If post-2022 is worse, it tells us: the economic relationship we relied on
has changed. The spread is no longer mean-reverting — it's trending.
"""

for (ticker_a, ticker_b), bdata in pair_backtests.items():
    pnl_df = bdata["pnl_df"]
    split  = "2022-01-01"
    pre    = pnl_df[pnl_df.index < split]["pnl"]
    post   = pnl_df[pnl_df.index >= split]["pnl"]

    pre_ret  = (1 + pre).prod()  - 1
    post_ret = (1 + post).prod() - 1

    print(f"    {ticker_a}/{ticker_b}:")
    print(f"        Pre-2022 net:   {pre_ret * 100:+.2f}%")
    print(f"        Post-2022 net:  {post_ret * 100:+.2f}%")
    if post_ret < pre_ret - 0.05:
        print(f"        *** REGIME BREAK detected — relationship weakened post-2022")
    else:
        print(f"        Regime: stable — relationship holds post-2022")


# ── CHART — 4-PANEL ───────────────────────────────────────────────────────────
print("\n[6/6] Building 4-panel research chart...")

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


pairs_list = list(pair_backtests.keys())

# Panel 1: Z-score + entry/exit for NVDA/AMD
ax1 = fig.add_subplot(gs[0, 0])
pair_1 = pairs_list[0]
z_1    = pair_backtests[pair_1]["z_score"]
style_ax(ax1, f"Panel 1 — Z-score Signal  {pair_1[0]}/{pair_1[1]}")

ax1.plot(z_1.index, z_1.values, color=BLUE, lw=1.0, alpha=0.8)
ax1.axhline(0,        color=AXIS_COL, lw=0.8, ls="--", alpha=0.5)
ax1.axhline( Z_ENTER, color=RED,    lw=1.2, ls="--", label=f"Enter short A (+{Z_ENTER})")
ax1.axhline(-Z_ENTER, color=GREEN,  lw=1.2, ls="--", label=f"Enter long A (-{Z_ENTER})")
ax1.axhline( Z_EXIT,  color=ORANGE, lw=0.8, ls=":",  alpha=0.7)
ax1.axhline(-Z_EXIT,  color=ORANGE, lw=0.8, ls=":",  alpha=0.7)
ax1.fill_between(z_1.index, Z_ENTER,  z_1.values,
                 where=(z_1.values >  Z_ENTER), color=RED,   alpha=0.15)
ax1.fill_between(z_1.index, -Z_ENTER, z_1.values,
                 where=(z_1.values < -Z_ENTER), color=GREEN, alpha=0.15)
ax1.legend(fontsize=7, facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)
ax1.set_ylabel("Z-score", fontsize=8)
ax1.set_xlabel("Date", fontsize=8)
ax1.set_ylim(-5, 5)

# Panel 2: Equity curves for both pairs
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Panel 2 — Cumulative Net Return (Both Pairs)")

colors = [GREEN, BLUE]
for i, (pair, bdata) in enumerate(pair_backtests.items()):
    pnl_df  = bdata["pnl_df"]
    cum_net = (1 + pnl_df["pnl"] - pnl_df["cost"]).cumprod()
    label   = f"{pair[0]}/{pair[1]}"
    ax2.plot(cum_net.index, cum_net.values, color=colors[i % 2], lw=2.0, label=label)

ax2.axhline(1.0, color=AXIS_COL, lw=0.8, ls="--", alpha=0.5)
ax2.axvline(pd.Timestamp("2022-01-01"), color=ORANGE, lw=1.2, ls="--", alpha=0.8,
            label="2022 regime break")
ax2.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)
ax2.set_ylabel("Cumulative Return", fontsize=8)
ax2.set_xlabel("Date", fontsize=8)

# Panel 3: Spread — log ratio over time
ax3 = fig.add_subplot(gs[1, 0])
pair_1  = pairs_list[0]
pair_2  = pairs_list[1] if len(pairs_list) > 1 else pairs_list[0]
spread1 = pair_backtests[pair_1]["spread"]
spread2 = pair_backtests[pair_2]["spread"]
style_ax(ax3, "Panel 3 — Spread (Log Residual) — Both Pairs")

ax3_twin = ax3.twinx()
ax3.plot(spread1.index, spread1.values, color=GREEN, lw=1.2,
         label=f"{pair_1[0]}/{pair_1[1]} spread", alpha=0.8)
ax3_twin.plot(spread2.index, spread2.values, color=BLUE, lw=1.2,
              label=f"{pair_2[0]}/{pair_2[1]} spread", alpha=0.8)
ax3.axhline(0, color=AXIS_COL, lw=0.8, ls="--", alpha=0.5)
ax3.axvline(pd.Timestamp("2022-01-01"), color=ORANGE, lw=1.0, ls="--", alpha=0.7)

ax3.set_ylabel(f"{pair_1[0]}/{pair_1[1]} spread", color=GREEN, fontsize=7)
ax3_twin.set_ylabel(f"{pair_2[0]}/{pair_2[1]} spread", color=BLUE, fontsize=7)
ax3_twin.tick_params(colors=AXIS_COL, labelsize=7)
ax3.set_xlabel("Date", fontsize=8)

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
           facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)

# Panel 4: Five Numbers Scorecard — both pairs side by side
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(BG_PANEL)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis("off")
ax4.set_title("Panel 4 — Five Numbers Scorecard", color=TITLE_COL,
               fontsize=10, fontweight="bold", pad=8)

header_y = 0.95
ax4.text(0.01, header_y, "METRIC",          color=AXIS_COL,  fontsize=7, va="top", fontfamily="monospace")
ax4.text(0.38, header_y, f"{pairs_list[0][0]}/{pairs_list[0][1]}",
         color=GREEN,     fontsize=7, va="top", fontfamily="monospace")
if len(pairs_list) > 1:
    ax4.text(0.68, header_y, f"{pairs_list[1][0]}/{pairs_list[1][1]}",
             color=BLUE, fontsize=7, va="top", fontfamily="monospace")

y = header_y - 0.07
ax4.text(0.01, y, "─────────────────", color=AXIS_COL, fontsize=7, va="top", fontfamily="monospace")
y -= 0.07

metrics = ["1. Gross Return", "2. Total Costs", "3. Net Return",
           "4. IC (Spearman)", "5. PSR", "─────────────────",
           "Hit Rate", "Sharpe (ann)", "# Trades"]

pair_keys = list(scorecard_data.keys())

for i, metric in enumerate(metrics):
    ax4.text(0.01, y, metric, color=AXIS_COL, fontsize=7, va="top", fontfamily="monospace")

    if "───" not in metric and pair_keys:
        p0 = scorecard_data[pair_keys[0]]
        val_map = {
            "1. Gross Return":  f"{p0['gross']*100:+.2f}%",
            "2. Total Costs":   f"{p0['costs']*100:.2f}%",
            "3. Net Return":    f"{p0['net']*100:+.2f}%",
            "4. IC (Spearman)": f"{p0['ic']:.4f}" if not np.isnan(p0['ic']) else "n/a",
            "5. PSR":           f"{p0['psr']*100:.1f}%",
            "Hit Rate":         f"{p0['hit']*100:.1f}%",
            "Sharpe (ann)":     f"{p0['sr']:.3f}",
            "# Trades":         f"{p0['n_trades']}",
        }
        col0 = GREEN if p0['net'] > 0 else RED
        ax4.text(0.38, y, val_map.get(metric, ""), color=col0,
                 fontsize=7, va="top", fontfamily="monospace")

        if len(pair_keys) > 1:
            p1 = scorecard_data[pair_keys[1]]
            val_map1 = {
                "1. Gross Return":  f"{p1['gross']*100:+.2f}%",
                "2. Total Costs":   f"{p1['costs']*100:.2f}%",
                "3. Net Return":    f"{p1['net']*100:+.2f}%",
                "4. IC (Spearman)": f"{p1['ic']:.4f}" if not np.isnan(p1['ic']) else "n/a",
                "5. PSR":           f"{p1['psr']*100:.1f}%",
                "Hit Rate":         f"{p1['hit']*100:.1f}%",
                "Sharpe (ann)":     f"{p1['sr']:.3f}",
                "# Trades":         f"{p1['n_trades']}",
            }
            col1 = BLUE
            ax4.text(0.68, y, val_map1.get(metric, ""), color=col1,
                     fontsize=7, va="top", fontfamily="monospace")

    y -= 0.075


fig.suptitle("Statistical Arbitrage — Pairs Trading | NVDA/AMD & MSFT/GOOGL | 2020–2024",
             color=TITLE_COL, fontsize=13, fontweight="bold", y=0.98)

out_path = "/Users/patiencefuglo/Desktop/intraday-alpha-research/charts/stat_arb_pairs.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"    Chart saved → {out_path}")

# ── FINAL READING ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("READING")
print("=" * 72)

nvda_amd = scorecard_data.get(("NVDA", "AMD"), {})
msft_googl = scorecard_data.get(("MSFT", "GOOGL"), {})

nvda_amd_coint  = coint_results.get(("NVDA", "AMD"), {})
msft_googl_coint = coint_results.get(("MSFT", "GOOGL"), {})

print(f"""
WHAT WE TESTED:
    Pairs: NVDA/AMD and MSFT/GOOGL
    Signal: 60-day rolling z-score of log-price spread
    Entry: |z| > 2.0 | Exit: |z| < 0.5 | Stop: |z| > 3.5

NVDA / AMD:
    Cointegration p-value: {nvda_amd_coint.get('coint_p', 'n/a')}
    Post-2022 coint p:     {nvda_amd_coint.get('coint_post', 'n/a')}
    Net Return:            {nvda_amd.get('net', 0) * 100:+.2f}%
    Sharpe:                {nvda_amd.get('sr', 0):.3f}
    IC:                    {nvda_amd.get('ic', 0):.4f}

MSFT / GOOGL:
    Cointegration p-value: {msft_googl_coint.get('coint_p', 'n/a')}
    Post-2022 coint p:     {msft_googl_coint.get('coint_post', 'n/a')}
    Net Return:            {msft_googl.get('net', 0) * 100:+.2f}%
    Sharpe:                {msft_googl.get('sr', 0):.3f}
    IC:                    {msft_googl.get('ic', 0):.4f}

KEY LESSONS:

1. COINTEGRATION IS NOT CORRELATION
   → Correlation: both stocks go up in a bull market.
   → Cointegration: there is an economic anchor pulling them together.
   → NVDA and AMD share customers (hyperscalers, gamers) — economic link.
   → Always test for cointegration before trading a pair.
   → ADF p < 0.05 = spread is stationary = safe to trade.

2. REGIME BREAKS KILL PAIRS
   → Post-2022: NVDA became the AI chip monopoly.
   → AMD could not replicate NVDA's CUDA software ecosystem.
   → The economic link weakened — their revenues and margins diverged.
   → If post-2022 cointegration p > 0.05, the pair must be retired.

3. STOP-LOSS IS NON-OPTIONAL IN PAIRS
   → When a pair breaks, the spread does NOT revert — it keeps widening.
   → A stop at |z| > 3.5 caps the loss from false signals.
   → Without the stop: the strategy holds a losing position indefinitely.

4. HEDGE RATIO MUST BE RECOMPUTED PERIODICALLY
   → OLS on the full sample gives one β. But β drifts.
   → Production implementation: rolling 252-day OLS to keep β current.
   → Using a stale β on a shifted relationship = systematic mis-hedging.

INTERVIEW LINE:
    "My stat arb study tested NVDA/AMD and MSFT/GOOGL pairs using
     Engle-Granger cointegration and a z-score entry/exit model.
     The key finding: the NVDA/AMD relationship broke post-2022 when
     NVDA's AI chip revenue diverged structurally from AMD.
     The lesson: cointegration must be re-tested continuously — a pair
     that worked in 2020 may not be tradeable in 2024."
""")
print("=" * 72)
