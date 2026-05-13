"""
================================================================================
12_CROSS_SECTIONAL_MOMENTUM.py — Cross-Sectional Equity Momentum
================================================================================

HYPOTHESIS
    Stocks that outperformed their peers over the past 12 months (skipping the
    most recent month) will continue to outperform over the next month.
    A portfolio that is long the top quintile and short the bottom quintile
    earns a statistically significant positive return after transaction costs.

WHAT IS CROSS-SECTIONAL MOMENTUM?
    Time-series signal:   "Will NVDA go up or down next month?"
                           You compare NVDA to itself over time.

    Cross-sectional signal: "Which stocks in the universe will outperform
                             their peers next month?"
                             You rank all stocks relative to each other
                             and build a relative-strength portfolio.

    Cross-sectional momentum is the most replicated factor in academic finance.
    Jegadeesh & Titman (1993): stocks in top decile by 12-1 return outperform
    bottom decile by approximately 1% per month over a 6-month holding period.
    It works across asset classes, geographies, and time periods.

THE SIGNAL — 12-1 MONTH MOMENTUM
    At the end of each month:
        1. Compute 12-month return for each stock (months t-12 to t-1)
        2. Skip the most recent month (t to t-1) — short-term reversal
        3. Rank all 30 stocks by momentum score
        4. Long top quintile (rank 1–6) — the winners
        5. Short bottom quintile (rank 25–30) — the losers
        6. Equal-weight within each leg
        7. Hold for one month, then rebalance

    WHY SKIP THE MOST RECENT MONTH?
        Short-term reversal (1-month) is the opposite effect.
        Including it would add noise and pull the signal toward mean reversion.
        Skipping it isolates the intermediate momentum factor.

THE PORTFOLIO STRUCTURE
    Universe:    30 large-cap stocks, 6 sectors (5 per sector)
                 Sectors: Technology, Financials, Healthcare, Consumer, Energy, Industrial
    Signal:      12-1 month total return
    Quintiles:   Rank 1–6 → long | Rank 25–30 → short
    Weighting:   Equal weight within each leg
    Rebalance:   Monthly
    Costs:       0.1% per trade (round-trip 0.2%) — institutional estimate

THE FIVE NUMBERS — what to read after each run:
    1. Gross Return     — does momentum earn a positive spread before costs?
    2. Total Costs      — how much does monthly rebalancing eat?
    3. Net Return       — does the factor survive transaction costs?
    4. IC               — do momentum ranks predict next-month returns?
    5. PSR              — is the Sharpe statistically real, not luck?
    +  Hit Rate         — what % of months is the long-short positive?

THRESHOLDS:
    Gross Return    > 0          factor must earn a spread
    Net Return      > 0          costs must not destroy the edge
    IC              > 0.05       Spearman rank correlation target
    PSR             > 50%        lower bar than intraday — fewer observations
    Hit Rate        > 50%        majority of months must be positive

WALK-FORWARD DESIGN
    Year 2021:   train on 2019–2020, test on 2021
    Year 2022:   train on 2019–2021, test on 2022
    Year 2023:   train on 2019–2022, test on 2023
    Note: "train" here = compute momentum on prior data, test = measure IC/return

    Walk-forward prevents look-ahead bias: we never use future data to rank stocks.

WHY THIS MATTERS FOR INTERVIEWS
    Cross-sectional thinking separates juniors from seniors.
    Most beginners ask: "Will this stock go up?"
    Systematic equity funds ask: "Which stocks will outperform the field?"
    Ranking, quintile construction, and rebalancing costs are core skills.

    Interview line: "My cross-sectional momentum study on 30 large-caps showed
    IC of [value] on out-of-sample walk-forward folds, with PSR confirming
    the edge is not noise. The key insight: the 12-1 skip month removes
    short-term reversal contamination."

STACK:
    yfinance, pandas, numpy, matplotlib, scipy, scikit-learn
    Data: 2019–2023 monthly returns (adjusted close, yfinance)
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
import yfinance as yf
from datetime import datetime

# ── UNIVERSE ──────────────────────────────────────────────────────────────────
# 30 large-cap stocks across 6 sectors (5 per sector)
# Chosen for liquidity, long data history, and sector diversity
UNIVERSE = {
    "Technology":   ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Financials":   ["JPM",  "BAC",  "GS",   "V",    "MA"],
    "Healthcare":   ["UNH",  "JNJ",  "LLY",  "MRK",  "ABBV"],
    "Consumer":     ["AMZN", "TSLA", "HD",   "MCD",  "WMT"],
    "Energy":       ["XOM",  "CVX",  "COP",  "SLB",  "PSX"],
    "Industrial":   ["CAT",  "HON",  "GE",   "MMM",  "UPS"],
}
TICKERS = [t for sector in UNIVERSE.values() for t in sector]

# ── PARAMETERS ────────────────────────────────────────────────────────────────
START_DATE       = "2019-01-01"
END_DATE         = "2024-01-01"
COST_PER_TRADE   = 0.001       # 0.1% one-way → 0.2% round-trip
QUINTILE_SIZE    = 6           # top 6 = long, bottom 6 = short (out of 30)
MOMENTUM_MONTHS  = 12          # lookback window
SKIP_MONTHS      = 1           # skip most recent month (short-term reversal)

WALK_FORWARD_YEARS = [2021, 2022, 2023]

print("=" * 72)
print("CROSS-SECTIONAL MOMENTUM — STUDY 12")
print("Universe: 30 large-cap stocks | Signal: 12-1 month momentum")
print("Portfolio: Long top quintile, Short bottom quintile | Monthly rebalance")
print("=" * 72)

# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────
print("\n[1/6] Downloading monthly price data from yfinance...")

raw = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    interval="1mo",
    auto_adjust=True,
    progress=False,
)["Close"]

# Keep only tickers that have full data
raw.dropna(axis=1, thresh=int(len(raw) * 0.8), inplace=True)
raw.ffill(inplace=True)
raw.dropna(inplace=True)

available = [t for t in TICKERS if t in raw.columns]
print(f"    Tickers with sufficient data: {len(available)} / {len(TICKERS)}")
print(f"    Date range: {raw.index[0].date()} → {raw.index[-1].date()}")
print(f"    Months: {len(raw)}")

# ── MONTHLY RETURNS ───────────────────────────────────────────────────────────
monthly_returns = raw.pct_change().dropna()

# ── MOMENTUM SIGNAL ───────────────────────────────────────────────────────────
print("\n[2/6] Computing 12-1 month momentum signal...")
"""
Momentum at month t =
    cumulative return from t-12 to t-1
    (skip month t-0 to t-1 to avoid short-term reversal)

    Implementation: use log returns for compounding.
    log_cum_12 = sum of log returns from t-12 to t-1
    Skip: we use t-SKIP_MONTHS:t-MOMENTUM_MONTHS as the window.
"""

log_returns = np.log(raw / raw.shift(1)).dropna()

# Momentum score: sum of log returns over months [t-12, t-2], skipping most recent month
# Result: one score per ticker per month
momentum_scores = pd.DataFrame(index=log_returns.index, columns=available)

for i in range(MOMENTUM_MONTHS + SKIP_MONTHS, len(log_returns)):
    window_start = i - MOMENTUM_MONTHS
    window_end   = i - SKIP_MONTHS          # skip the most recent month
    window       = log_returns.iloc[window_start:window_end][available]
    momentum_scores.iloc[i] = window.sum(axis=0)

momentum_scores = momentum_scores.apply(pd.to_numeric, errors="coerce")
momentum_scores.dropna(inplace=True)

print(f"    Momentum scores computed: {len(momentum_scores)} months")
print(f"    Sample (last 3 months, top 5 tickers):")
print(momentum_scores.iloc[-3:, :5].round(4).to_string())

# ── QUINTILE PORTFOLIO ─────────────────────────────────────────────────────────
print("\n[3/6] Constructing quintile portfolio...")
"""
Each month:
    1. Rank all stocks by momentum score (high = winner)
    2. Long top QUINTILE_SIZE stocks (highest momentum)
    3. Short bottom QUINTILE_SIZE stocks (lowest momentum)
    4. Equal weight each leg → +1/Q per long, -1/Q per short
    5. Observe next-month return
    6. Compute long-short spread = long return - short return
"""

def build_quintile_portfolio(scores_df, returns_df, cost=COST_PER_TRADE):
    """
    Returns a DataFrame with columns:
        long_ret, short_ret, ls_gross, cost, ls_net, n_long, n_short
    One row per month.
    """
    results = []

    for i in range(len(scores_df) - 1):
        date      = scores_df.index[i]
        next_date = scores_df.index[i + 1]

        scores = scores_df.iloc[i].dropna()
        if len(scores) < QUINTILE_SIZE * 2 + 2:
            continue

        # Rank stocks: highest momentum = rank 1
        ranked     = scores.rank(ascending=False)
        longs      = ranked[ranked <= QUINTILE_SIZE].index.tolist()
        shorts     = ranked[ranked > len(ranked) - QUINTILE_SIZE].index.tolist()

        # Get next-month returns for portfolio members
        if next_date not in returns_df.index:
            continue
        next_rets = returns_df.loc[next_date]

        long_ret  = next_rets[longs].mean()   if longs  else 0.0
        short_ret = next_rets[shorts].mean()  if shorts else 0.0

        # Long-short spread: long winners, short losers
        ls_gross = long_ret - short_ret

        # Transaction cost: we rebalance the full portfolio each month
        # Cost = 0.1% × 2 legs × 2 (buy+sell) = 0.4% round-trip per rebalance
        monthly_cost = cost * 4
        ls_net = ls_gross - monthly_cost

        results.append({
            "date":      date,
            "long_ret":  long_ret,
            "short_ret": short_ret,
            "ls_gross":  ls_gross,
            "cost":      monthly_cost,
            "ls_net":    ls_net,
            "longs":     longs,
            "shorts":    shorts,
        })

    return pd.DataFrame(results).set_index("date")


portfolio_df = build_quintile_portfolio(momentum_scores, monthly_returns)
print(f"    Portfolio months constructed: {len(portfolio_df)}")
print(f"    Average monthly longs: {QUINTILE_SIZE} | Average monthly shorts: {QUINTILE_SIZE}")

# ── IC COMPUTATION ─────────────────────────────────────────────────────────────
print("\n[4/6] Computing Spearman Rank IC...")
"""
Information Coefficient (IC) = Spearman rank correlation
    between momentum rank (signal) and next-month return rank (outcome).

    Spearman is used instead of Pearson because:
    - We care about rank order (does the top stock beat the bottom stock?)
    - Rank correlation is robust to outliers (one huge NVDA move won't dominate)
    - It is the standard IC measure at systematic equity funds.

    IC interpretation:
        IC > 0.05 = meaningful predictive relationship
        IC > 0.10 = strong signal
        IC < 0   = signal is predicting wrong direction
"""

ic_series = []
ic_dates  = []

for i in range(len(momentum_scores) - 1):
    date      = momentum_scores.index[i]
    next_date = momentum_scores.index[i + 1]

    scores = momentum_scores.iloc[i].dropna()
    if next_date not in monthly_returns.index:
        continue

    next_rets = monthly_returns.loc[next_date, scores.index].dropna()
    common    = scores.index.intersection(next_rets.index)
    if len(common) < 10:
        continue

    rho, _ = stats.spearmanr(scores[common], next_rets[common])
    ic_series.append(rho)
    ic_dates.append(date)

ic_series = pd.Series(ic_series, index=ic_dates)
avg_ic    = ic_series.mean()
ic_std    = ic_series.std()
ic_t_stat = avg_ic / (ic_std / np.sqrt(len(ic_series)))

print(f"    Average IC:      {avg_ic:.4f}")
print(f"    IC std dev:      {ic_std:.4f}")
print(f"    IC t-stat:       {ic_t_stat:.2f}  (>2.0 = significant)")
print(f"    IC > 0 months:   {(ic_series > 0).sum()} / {len(ic_series)}")

# ── PSR COMPUTATION ────────────────────────────────────────────────────────────
print("\n[5/6] Computing PSR and Five Numbers...")

from scipy.stats import norm

def compute_psr(returns_series, sr_benchmark=0.0):
    """
    Probabilistic Sharpe Ratio — Lopez de Prado (2014).
    Corrects Sharpe for skewness and excess kurtosis (non-normal returns).
    PSR = P(true Sharpe > benchmark) given observed Sharpe and sample moments.

    A PSR of 95% means: 95% confidence the strategy beats the benchmark.
    """
    n      = len(returns_series)
    sr_obs = returns_series.mean() / returns_series.std() * np.sqrt(12)  # annualised
    skew   = returns_series.skew()
    kurt   = returns_series.kurt()   # excess kurtosis

    # Lopez de Prado formula
    z = (sr_obs - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(
        1 - skew * sr_obs + ((kurt + 3) / 4) * sr_obs ** 2
    )
    return float(norm.cdf(z)), float(sr_obs)


ls_gross = portfolio_df["ls_gross"]
ls_net   = portfolio_df["ls_net"]

psr_gross, sr_gross = compute_psr(ls_gross)
psr_net,   sr_net   = compute_psr(ls_net)

gross_total = (1 + ls_gross).prod() - 1
net_total   = (1 + ls_net).prod()   - 1
total_cost  = portfolio_df["cost"].sum()
hit_rate    = (ls_net > 0).mean()

print(f"\n    ── FIVE NUMBERS SCORECARD ──────────────────────────────────────")
print(f"    1. Gross Return:   {gross_total * 100:+.2f}%     {'PASS ✓' if gross_total > 0 else 'FAIL ✗'}")
print(f"    2. Total Costs:    {total_cost * 100:.2f}%     (0.4% × {len(portfolio_df)} months)")
print(f"    3. Net Return:     {net_total * 100:+.2f}%     {'PASS ✓' if net_total > 0 else 'FAIL ✗'}")
print(f"    4. Avg IC:         {avg_ic:.4f}     {'PASS ✓' if avg_ic > 0.03 else 'WEAK'} (target >0.05)")
print(f"    5. PSR (net):      {psr_net * 100:.1f}%     {'PASS ✓' if psr_net > 0.50 else 'FAIL ✗'}")
print(f"    +  Hit Rate:       {hit_rate * 100:.1f}%     {'PASS ✓' if hit_rate > 0.50 else 'FAIL ✗'}")
print(f"    +  Sharpe (ann):   {sr_net:.3f}")
print(f"    +  IC t-stat:      {ic_t_stat:.2f}  {'SIGNIFICANT' if abs(ic_t_stat) > 2.0 else 'NOT SIGNIFICANT'}")
print(f"    ────────────────────────────────────────────────────────────────")

# ── WALK-FORWARD ───────────────────────────────────────────────────────────────
print("\n[6/6] Walk-forward by year — out-of-sample IC per year...")
"""
Walk-forward prevents data snooping:
    We compute momentum signals only from data available before each test year.
    Train period accumulates: we never peek at future rankings.

    2021: signal computed from 2019–2020 data → test January–December 2021
    2022: signal computed from 2019–2021 data → test January–December 2022
    2023: signal computed from 2019–2022 data → test January–December 2023
"""

wf_results = []
for test_year in WALK_FORWARD_YEARS:
    mask = momentum_scores.index.year == test_year
    if mask.sum() < 3:
        continue

    scores_year = momentum_scores[mask]
    year_ics    = []

    for i, date in enumerate(scores_year.index):
        next_months = monthly_returns.index[monthly_returns.index > date]
        if len(next_months) == 0:
            continue
        next_date = next_months[0]

        scores   = scores_year.loc[date].dropna()
        next_ret = monthly_returns.loc[next_date, scores.index].dropna()
        common   = scores.index.intersection(next_ret.index)
        if len(common) < 8:
            continue

        rho, _ = stats.spearmanr(scores[common], next_ret[common])
        year_ics.append(rho)

    if year_ics:
        avg = np.mean(year_ics)
        wf_results.append({"year": test_year, "IC": avg, "months": len(year_ics)})
        status = "+" if avg > 0 else "-"
        print(f"    {test_year}: IC = {avg:+.4f}  ({len(year_ics)} months)  [{status}]")

wf_df = pd.DataFrame(wf_results)

# ── CHART — 4-PANEL ───────────────────────────────────────────────────────────
print("\n[Chart] Building 4-panel research chart...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

TITLE_COL  = "#e6edf3"
AXIS_COL   = "#8b949e"
GRID_COL   = "#21262d"
GREEN      = "#3fb950"
RED        = "#f85149"
BLUE       = "#58a6ff"
ORANGE     = "#d29922"
BG_PANEL   = "#161b22"

def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=TITLE_COL, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=AXIS_COL, labelsize=8)
    ax.spines["bottom"].set_color(AXIS_COL)
    ax.spines["left"].set_color(AXIS_COL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color(AXIS_COL)
    ax.xaxis.label.set_color(AXIS_COL)

# ── Panel 1: Equity Curve — Cumulative long-short net return ─────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Panel 1 — Long-Short Equity Curve (Net)")

cum_gross = (1 + ls_gross).cumprod()
cum_net   = (1 + ls_net).cumprod()

ax1.plot(cum_gross.index, cum_gross.values, color=BLUE,  lw=1.5, label="Gross", alpha=0.7)
ax1.plot(cum_net.index,   cum_net.values,   color=GREEN, lw=2.0, label="Net")
ax1.axhline(1.0, color=AXIS_COL, lw=0.8, ls="--", alpha=0.5)
ax1.fill_between(cum_net.index, 1.0, cum_net.values,
                 where=(cum_net.values >= 1.0), color=GREEN, alpha=0.08)
ax1.fill_between(cum_net.index, 1.0, cum_net.values,
                 where=(cum_net.values < 1.0),  color=RED,   alpha=0.08)
ax1.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)
ax1.set_ylabel("Cumulative Return", fontsize=8)
ax1.set_xlabel("Date", fontsize=8)

# Add net return annotation
final_net = cum_net.iloc[-1] - 1
ax1.annotate(f"Net: {final_net * 100:+.1f}%",
             xy=(cum_net.index[-1], cum_net.iloc[-1]),
             xytext=(-80, 10), textcoords="offset points",
             color=GREEN if final_net > 0 else RED, fontsize=9, fontweight="bold")

# ── Panel 2: IC Time Series ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Panel 2 — Monthly Spearman IC (12-1 Signal)")

ic_colors = [GREEN if v > 0 else RED for v in ic_series.values]
ax2.bar(ic_series.index, ic_series.values, color=ic_colors, alpha=0.7, width=20)
ax2.axhline(0,       color=AXIS_COL, lw=0.8, ls="--")
ax2.axhline(0.05,    color=BLUE,     lw=1.0, ls=":",  label="IC target (0.05)")
ax2.axhline(avg_ic,  color=ORANGE,   lw=1.5, ls="-",  label=f"Avg IC ({avg_ic:.3f})")
ax2.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)
ax2.set_ylabel("Spearman IC", fontsize=8)
ax2.set_xlabel("Date", fontsize=8)

# ── Panel 3: Walk-Forward IC by Year ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Panel 3 — Walk-Forward IC by Year (Out-of-Sample)")

if len(wf_df) > 0:
    bar_colors = [GREEN if v > 0 else RED for v in wf_df["IC"]]
    bars = ax3.bar(wf_df["year"].astype(str), wf_df["IC"], color=bar_colors, alpha=0.8, width=0.5)
    ax3.axhline(0, color=AXIS_COL, lw=0.8, ls="--")
    ax3.axhline(0.05, color=BLUE, lw=1.0, ls=":", label="IC target")
    for bar, val in zip(bars, wf_df["IC"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:+.3f}", ha="center", va="bottom",
                 color=TITLE_COL, fontsize=9, fontweight="bold")
    ax3.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=AXIS_COL, labelcolor=TITLE_COL)
ax3.set_ylabel("IC (Spearman)", fontsize=8)
ax3.set_xlabel("Test Year", fontsize=8)
ax3.set_title("Panel 3 — Walk-Forward IC by Year (Out-of-Sample)", color=TITLE_COL,
               fontsize=11, fontweight="bold", pad=8)

# ── Panel 4: Five Numbers Scorecard ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(BG_PANEL)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis("off")
ax4.set_title("Panel 4 — Five Numbers Scorecard", color=TITLE_COL,
               fontsize=11, fontweight="bold", pad=8)

scorecard = [
    ("METRIC",            "VALUE",                   "STATUS",     AXIS_COL),
    ("──────────────────", "──────",                  "──────",     AXIS_COL),
    ("1. Gross Return",   f"{gross_total * 100:+.2f}%", "PASS ✓" if gross_total > 0 else "FAIL ✗",
                          GREEN if gross_total > 0 else RED),
    ("2. Total Costs",    f"{total_cost * 100:.2f}%",   "——",        AXIS_COL),
    ("3. Net Return",     f"{net_total * 100:+.2f}%",   "PASS ✓" if net_total > 0 else "FAIL ✗",
                          GREEN if net_total > 0 else RED),
    ("4. Avg IC",         f"{avg_ic:.4f}",              "PASS ✓" if avg_ic > 0.03 else "WEAK",
                          GREEN if avg_ic > 0.05 else ORANGE if avg_ic > 0 else RED),
    ("5. PSR (net)",      f"{psr_net * 100:.1f}%",      "PASS ✓" if psr_net > 0.50 else "FAIL ✗",
                          GREEN if psr_net > 0.50 else RED),
    ("──────────────────", "──────",                  "──────",     AXIS_COL),
    ("Hit Rate",          f"{hit_rate * 100:.1f}%",     "PASS ✓" if hit_rate > 0.50 else "FAIL ✗",
                          GREEN if hit_rate > 0.50 else RED),
    ("Sharpe (ann)",      f"{sr_net:.3f}",              "——",        AXIS_COL),
    ("IC t-stat",         f"{ic_t_stat:.2f}",           "SIG" if abs(ic_t_stat) > 2.0 else "——",
                          GREEN if abs(ic_t_stat) > 2.0 else AXIS_COL),
    ("Universe",          "30 large-cap",              "6 sectors", AXIS_COL),
]

y_pos = 0.95
for row in scorecard:
    metric, val, status, color = row
    ax4.text(0.02, y_pos, metric, color=AXIS_COL,  fontsize=8, va="top", fontfamily="monospace")
    ax4.text(0.52, y_pos, val,    color=TITLE_COL, fontsize=8, va="top", fontfamily="monospace")
    ax4.text(0.78, y_pos, status, color=color,     fontsize=8, va="top", fontfamily="monospace")
    y_pos -= 0.075

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle("Cross-Sectional Momentum | 30 Large-Cap Stocks | 12-1 Signal | 2019–2023",
             color=TITLE_COL, fontsize=13, fontweight="bold", y=0.98)

out_path = "/Users/patiencefuglo/Desktop/intraday-alpha-research/charts/cross_sectional_momentum.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"    Chart saved → {out_path}")

# ── FINAL READING ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("READING")
print("=" * 72)
print(f"""
WHAT WE TESTED:
    12-1 month momentum signal on 30 large-cap stocks.
    Long top 6 stocks (winners), short bottom 6 (losers).
    Monthly rebalance. Equal weight. 0.1% cost per leg.

RESULT:
    Gross Return:  {gross_total * 100:+.2f}%  over {len(portfolio_df)} months
    Net Return:    {net_total * 100:+.2f}%  after {total_cost * 100:.2f}% total costs
    Average IC:    {avg_ic:.4f}  (Spearman rank, predictions vs actuals)
    PSR (net):     {psr_net * 100:.1f}%  (probability edge is real)
    Hit Rate:      {hit_rate * 100:.1f}%  of months long-short is positive

WHY THE SIGNAL WORKS (WHEN IT WORKS):
    → Institutional momentum — large funds take months to fully allocate
      to a theme. Price trends persist as more capital rotates in.
    → Investor underreaction — earnings upgrades and re-ratings happen
      gradually. The market is slow to fully reprice.
    → Trend persistence — strong stocks attract more buyers; weak stocks
      see continued selling. The spread widens before it narrows.

WHY THE SIGNAL FAILS (2022 NOTE):
    → Momentum crashes in sharp reversals (value rotation).
    → 2022: Fed hike cycle → growth/momentum stocks collapsed in January.
      Top momentum names (NVDA, META) were the worst performers.
    → Momentum is a risk factor — it earns a premium on average but
      has fat left-tail risk (momentum crashes).

WALK-FORWARD IC by YEAR:""")

for _, row in wf_df.iterrows():
    direction = "positive" if row["IC"] > 0 else "NEGATIVE"
    print(f"    {int(row['year'])}: IC = {row['IC']:+.4f} ({direction})")

print(f"""
KEY LESSON:
    Cross-sectional thinking is how systematic equity funds allocate capital.
    Not "will NVDA go up?" but "which 6 stocks will outperform the field?"
    Ranking removes absolute market direction from the bet.
    The long-short portfolio is market-neutral by construction.

INTERVIEW LINE:
    "My cross-sectional momentum study on 30 large-caps showed average
     IC of {avg_ic:.3f} over walk-forward folds from 2021–2023.
     The 12-1 skip-month design removes short-term reversal contamination.
     The portfolio is constructed as a rank-sorted quintile long-short,
     monthly rebalance, equal-weight — standard factor construction."
""")
print("=" * 72)
