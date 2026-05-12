"""
============================================================
09_EXECUTION_MODELS.py  |  Senior Level  |  TWAP, VWAP & Almgren-Chriss
============================================================

HYPOTHESIS
----------
A signal tells you WHAT to trade. Execution tells you HOW to trade it.

Naive execution (one large market order) incurs large market impact.
Smart execution slices the order into smaller pieces over time.

Hypothesis: For a $500,000 NVDA position, VWAP execution reduces total
market impact cost by at least 20% compared to naive single-shot execution,
because concentrating volume in high-liquidity windows reduces price impact
per unit of order flow (Kyle lambda effect).

We model three execution strategies:
  1. Naive (single market order at open)
  2. TWAP (equal slices across N time intervals)
  3. VWAP (slices proportional to historical volume profile)

And we use Almgren-Chriss to compute the OPTIMAL slice schedule that
balances: market impact (cost of urgency) vs timing risk (cost of waiting).

-------------------------------------------------------------
WHAT IS TWAP?
-------------------------------------------------------------

TWAP = Time-Weighted Average Price
Goal: achieve the average price over the execution window.
Method: divide total shares Q into N equal slices: q_t = Q / N
        execute one slice every time interval.

Example: buy 10,000 shares of NVDA over 60 minutes.
  TWAP: 1,000 shares every 6 minutes.
  Cost = sum of prices at each slice time.

Advantage: simple, predictable, low information leakage.
Disadvantage: ignores volume profile — executes same size in thin periods
              (e.g., 10:00am) as in liquid periods (e.g., 10:30am).
              In thin windows, impact is higher.

-------------------------------------------------------------
WHAT IS VWAP?
-------------------------------------------------------------

VWAP = Volume-Weighted Average Price
Goal: match the market's natural volume pattern.
Method: estimate volume profile v_t (fraction of daily volume in each bar).
        Set slice size q_t = Q × v_t.

Example: 60% of NVDA daily volume trades in first 2 hours.
  TWAP: 50% of execution in first 2 hours.
  VWAP: 60% of execution in first 2 hours.

By matching volume, VWAP executes in proportion to market liquidity.
Result: lower impact because you are not a disproportionate fraction of flow.

Advantage: lower impact cost vs TWAP in liquid windows.
Disadvantage: harder to pre-compute (needs volume forecast), risk of
              price drift if direction holds during execution window.

-------------------------------------------------------------
WHAT IS ALMGREN-CHRISS?
-------------------------------------------------------------

Almgren & Chriss (2000) — "Optimal execution of portfolio transactions."
The canonical academic model for institutional execution.

Two types of cost:
  1. PERMANENT IMPACT — your buy permanently shifts the price up.
     Other participants observe your order flow and update fair value.
     Cost = η × (Q / T) × T = η × Q
     where η = permanent impact coefficient

  2. TEMPORARY IMPACT — your buy transiently moves the price up.
     The price bounces back after you stop buying (market maker adjustment).
     Cost = γ × (q_t / V_t) = γ × (execution rate / market volume)
     where γ = temporary impact coefficient, V_t = market volume per bar

OPTIMAL SCHEDULE: Almgren-Chriss minimises:
  E[cost] + λ × Var[cost]
  where λ = risk aversion (how much you penalise variance vs expected cost)

Result: optimal trajectory X*(t).
  λ → 0: VWAP (ignores risk, spreads execution evenly)
  λ → ∞: immediate execution (risk-averse, ignores cost)
  λ ∈ between: hyperbolic decay schedule

KEY INSIGHT: the optimal schedule front-loads execution (execute more
at the beginning) to reduce timing risk, while still spreading out
enough to avoid excessive impact. The faster you trade, the MORE
you impact the price — but waiting carries TIMING RISK (price may move).

-------------------------------------------------------------
FIVE NUMBERS (execution quality)
-------------------------------------------------------------

  1. Implementation Shortfall (IS)   — gap between decision price and avg exec price
  2. TWAP Slippage                   — avg execution price vs TWAP benchmark
  3. VWAP Slippage                   — avg execution price vs VWAP benchmark
  4. AC Permanent Impact             — permanent price shift from our trading
  5. AC Temporary Impact             — transient price shift, peak during execution

THRESHOLDS:
  IS < 10bp         — acceptable total slippage (0.10%)
  TWAP slippage < 5bp
  VWAP slippage < 3bp  (VWAP harder to beat than TWAP)
  Permanent Impact < 5bp
  Temporary Impact < 8bp

-------------------------------------------------------------
INTERVIEW LINES
-------------------------------------------------------------
"In Almgren-Chriss, there are two impact types. Permanent impact reflects
information: when I buy, other participants infer I have a signal and
update their fair value upward. This cost cannot be recovered. Temporary
impact is a liquidity cost: I pay a premium to transact quickly, but the
price bounces back. The optimal schedule front-loads execution to reduce
timing risk while staying below the market participation rate to limit impact."

"VWAP beats TWAP because it executes proportionally to volume. In the first
hour of US trading, volume is typically 40% of the day's total. VWAP
concentrates execution there, where impact per share is lower. TWAP
spreads uniformly, so it gets the same impact in thin mid-day sessions."

============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — DATA
# ============================================================

def download_data(ticker: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    """Download hourly OHLCV. Returns cleaned DataFrame."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df = df.dropna()
    df.columns = [c[0].lower() for c in df.columns]
    return df


# ============================================================
# SECTION 2 — VOLUME PROFILE (foundation for VWAP execution)
# ============================================================

def estimate_volume_profile(df: pd.DataFrame) -> pd.Series:
    """
    Estimate the typical intraday volume profile.

    For each bar position within the trading day, compute the average
    fraction of daily volume traded in that bar.

    Returns: Series of length = bars_per_day, summing to 1.0.
    This is the blueprint for VWAP execution sizing.
    """
    # Group by hour-of-day to capture intraday volume pattern
    if hasattr(df.index, 'hour'):
        df = df.copy()
        df["hour"] = df.index.hour
        # Average volume by hour across all days
        hourly_vol = df.groupby("hour")["volume"].mean()
        profile    = hourly_vol / hourly_vol.sum()    # normalise to fractions summing to 1
        return profile
    else:
        # Fallback: uniform profile (TWAP equivalent)
        n = min(6, len(df))
        return pd.Series(np.ones(n) / n)


# ============================================================
# SECTION 3 — EXECUTION MODELS
# ============================================================

def simulate_twap(prices: np.ndarray, total_shares: int, n_slices: int) -> dict:
    """
    TWAP: divide Q shares equally across N time slices.

    At each slice: execute Q/N shares at the prevailing market price.
    Total cost = sum(price_i * slice_size) for i in 1..N.

    Implementation Shortfall = (avg_exec_price - decision_price) / decision_price.
    The decision price is the price at which we first decided to trade.
    """
    slice_sz     = total_shares / n_slices
    # Pick evenly-spaced execution times within the price array
    times        = np.linspace(0, len(prices) - 1, n_slices, dtype=int)
    exec_prices  = prices[times]

    avg_exec     = float(np.average(exec_prices, weights=np.ones(n_slices)))
    decision_px  = float(prices[0])   # price when signal fires
    slippage_bps = (avg_exec - decision_px) / decision_px * 10000   # in basis points

    return {
        "method":       "TWAP",
        "avg_exec":     avg_exec,
        "decision_px":  decision_px,
        "slippage_bps": float(slippage_bps),
        "exec_times":   times,
        "exec_prices":  exec_prices,
        "shares_per_slice": slice_sz,
    }


def simulate_vwap(prices: np.ndarray, volume: np.ndarray,
                  total_shares: int) -> dict:
    """
    VWAP: divide Q shares proportionally to the volume in each bar.

    q_t = Q × (volume_t / total_volume_in_window)
    Execute q_t shares at price_t.

    By matching the market's natural volume distribution, we become a
    constant fraction of market flow → lower impact per share.
    """
    # Normalise volume to get proportion per bar
    vol_profile  = volume / (volume.sum() + 1e-9)   # fractions summing to 1
    shares_sched = total_shares * vol_profile        # shares per bar

    # Weighted average execution price
    avg_exec     = float(np.average(prices, weights=vol_profile))
    decision_px  = float(prices[0])
    slippage_bps = (avg_exec - decision_px) / decision_px * 10000

    return {
        "method":       "VWAP",
        "avg_exec":     avg_exec,
        "decision_px":  decision_px,
        "slippage_bps": float(slippage_bps),
        "vol_profile":  vol_profile,
        "shares_sched": shares_sched,
    }


def simulate_naive(prices: np.ndarray, total_shares: int) -> dict:
    """
    Naive: execute ALL shares at once at the open price.
    Worst case for market impact — used as benchmark for improvement.
    """
    exec_price   = float(prices[0])
    slippage_bps = 0.0    # no slippage vs decision price (we ARE the decision price)

    return {
        "method":       "Naive",
        "avg_exec":     exec_price,
        "decision_px":  exec_price,
        "slippage_bps": float(slippage_bps),
    }


# ============================================================
# SECTION 4 — ALMGREN-CHRISS MODEL
# ============================================================

def almgren_chriss(total_shares: int,
                   exec_bars: int,
                   avg_daily_volume: float,
                   price: float,
                   volatility_daily: float,
                   eta: float = 0.1,         # permanent impact coefficient
                   gamma: float = 0.1,       # temporary impact coefficient
                   lambda_risk: float = 1e-6 # risk aversion
                   ) -> dict:
    """
    Almgren-Chriss (2000) optimal execution trajectory.

    Parameters:
      total_shares      Q — total shares to execute
      exec_bars         T — number of bars for execution
      avg_daily_volume  V — average market volume per bar (shares)
      price             P — current market price
      volatility_daily  σ — daily volatility of the stock (fraction)
      eta               η — permanent impact coefficient
                            ΔP_permanent = η × (trading_rate / V)
      gamma             γ — temporary impact coefficient
                            ΔP_temp = γ × (q_t / V)
      lambda_risk       λ — risk aversion: higher → execute faster

    Returns:
      trajectory    X(t): cumulative shares remaining at each bar
      trade_sched   q(t): shares traded each bar
      perm_impact   total permanent impact in dollars
      temp_impact   total temporary impact in dollars
      total_impact  sum of both
    """
    # Almgren-Chriss characteristic time scale
    # κ = sqrt(λ σ² / γ)  — decay rate of optimal trajectory
    sigma_bar = volatility_daily / np.sqrt(252)   # per-bar vol (rough)
    kappa     = np.sqrt(lambda_risk * sigma_bar**2 / (gamma + 1e-9))

    # Optimal trajectory: X(t) = Q × sinh(κ(T-t)) / sinh(κT)
    # Hyperbolic decay — more execution front-loaded as λ increases
    t_vals     = np.arange(exec_bars + 1)
    kT         = kappa * exec_bars

    if kT > 700:  # prevent overflow in sinh
        kT = 700.0

    denom      = np.sinh(kT) + 1e-9
    X          = total_shares * np.sinh(kappa * (exec_bars - t_vals)) / denom
    X          = np.maximum(X, 0)                  # no negative shares remaining

    # Trade schedule: q(t) = X(t) - X(t+1)
    q          = np.diff(-X)                       # shares traded each bar
    q          = np.maximum(q, 0)

    # Bar-by-bar prices (use flat price for model illustration)
    bar_prices = np.ones(exec_bars) * price

    # Permanent impact: η × (q_t / V) × P for each bar, cumulated
    participation_rate = q / (avg_daily_volume + 1e-9)
    perm_per_bar       = eta * participation_rate * bar_prices * q
    perm_total         = float(perm_per_bar.sum())

    # Temporary impact: γ × (q_t / V) × P for each bar
    temp_per_bar       = gamma * participation_rate * bar_prices * q
    temp_total         = float(temp_per_bar.sum())

    total_impact       = perm_total + temp_total
    total_bps          = total_impact / (total_shares * price + 1e-9) * 10000

    return {
        "trajectory":        X,
        "trade_schedule":    q,
        "t_vals":            t_vals,
        "perm_impact_usd":   perm_total,
        "temp_impact_usd":   temp_total,
        "total_impact_usd":  total_impact,
        "total_impact_bps":  float(total_bps),
        "participation_rate": participation_rate,
        "kappa":             float(kappa),
    }


# ============================================================
# SECTION 5 — FULL PIPELINE
# ============================================================

def run_pipeline(ticker: str, order_size_usd: float = 500_000) -> dict:
    """
    Full execution pipeline for one ticker:
      download → estimate volume profile → run TWAP/VWAP/Naive → Almgren-Chriss
    """
    print(f"\n{'='*55}")
    print(f"  {ticker}  |  Execution Model Analysis")
    print(f"  Order size: ${order_size_usd:,.0f}")
    print(f"{'='*55}")

    df = download_data(ticker)
    if df.empty:
        print(f"  ERROR: no data for {ticker}")
        return {}

    # Take one representative trading day's bars for simulation
    # Use the last 6 bars (one trading session, roughly 10am-4pm)
    prices = df["close"].values[-6:]
    volume = df["volume"].values[-6:]
    price  = float(prices[0])

    # Shares to buy
    total_shares     = int(order_size_usd / price)
    n_slices         = len(prices)   # 6 hourly bars in execution window

    print(f"  Price: ${price:.2f}  |  Shares to buy: {total_shares:,}")
    print(f"  Execution window: {n_slices} bars (6 hours)")

    # --- Run execution models ---
    naive = simulate_naive(prices, total_shares)
    twap  = simulate_twap(prices, total_shares, n_slices)
    vwap  = simulate_vwap(prices, volume, total_shares)

    # --- Almgren-Chriss ---
    avg_vol_per_bar = float(df["volume"].mean())
    daily_vol       = float(df["close"].pct_change(1).std() * np.sqrt(6))  # rough daily vol from hourly
    ac = almgren_chriss(
        total_shares    = total_shares,
        exec_bars       = n_slices,
        avg_daily_volume = avg_vol_per_bar,
        price           = price,
        volatility_daily = daily_vol,
        eta             = 0.1,
        gamma           = 0.1,
        lambda_risk     = 1e-6,
    )

    # --- Print five numbers ---
    print(f"\n  FIVE NUMBERS — EXECUTION QUALITY")
    print(f"  {'Metric':<35} {'Value':>12}  {'Threshold':>10}  {'Status':>8}")
    print(f"  {'-'*68}")

    items = [
        ("Naive Slippage",     f"{naive['slippage_bps']:+.1f} bps",  "< 10 bps",
         bool(abs(naive['slippage_bps']) < 10)),
        ("TWAP Slippage",      f"{twap['slippage_bps']:+.1f} bps",   "< 5 bps",
         bool(abs(twap['slippage_bps']) < 5)),
        ("VWAP Slippage",      f"{vwap['slippage_bps']:+.1f} bps",   "< 3 bps",
         bool(abs(vwap['slippage_bps']) < 3)),
        ("AC Permanent Impact", f"{ac['total_impact_bps']:+.2f} bps", "< 5 bps",
         bool(ac['total_impact_bps'] < 5)),
        ("AC Temp/Perm Split",
         f"P={ac['perm_impact_usd']:+.0f} T={ac['temp_impact_usd']:+.0f}",
         "",
         None),
        ("Participation Rate",
         f"{float(ac['participation_rate'].mean()*100):.2f}% avg",
         "< 10%",
         bool(float(ac['participation_rate'].mean()) < 0.10)),
        ("VWAP vs TWAP Improvement",
         f"{abs(twap['slippage_bps']) - abs(vwap['slippage_bps']):+.1f} bps",
         "> 0",
         bool(abs(twap['slippage_bps']) > abs(vwap['slippage_bps']))),
    ]

    for label, val, thresh, passed in items:
        if passed is None:
            print(f"  {label:<35} {val:>12}  {thresh:>10}")
        else:
            s = "PASS ✓" if passed else "FAIL ✗"
            print(f"  {label:<35} {val:>12}  {thresh:>10}  {s:>8}")

    return {
        "ticker":        ticker,
        "prices":        prices,
        "volume":        volume,
        "total_shares":  total_shares,
        "price":         price,
        "naive":         naive,
        "twap":          twap,
        "vwap":          vwap,
        "ac":            ac,
        "n_slices":      n_slices,
    }


# ============================================================
# SECTION 6 — CHART (standard format)
# ============================================================

def make_chart(results: list, save_path: str = "charts/execution_models.png"):
    """
    Three panels per ticker:
      Panel 1: Execution schedule comparison (TWAP vs VWAP vs AC)
      Panel 2: Almgren-Chriss trajectory (shares remaining over time)
      Panel 3: Five numbers scorecard
    """
    n_tickers = len(results)
    fig, axes = plt.subplots(n_tickers, 3,
                             figsize=(20, 6 * n_tickers),
                             gridspec_kw={"width_ratios": [2, 2, 1]})
    if n_tickers == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0d1117")

    for row, res in enumerate(results):
        ticker      = res["ticker"]
        prices      = res["prices"]
        volume      = res["volume"]
        twap        = res["twap"]
        vwap        = res["vwap"]
        ac          = res["ac"]
        n_slices    = res["n_slices"]
        total_sh    = res["total_shares"]
        ax_sched    = axes[row][0]
        ax_traj     = axes[row][1]
        ax_tbl      = axes[row][2]

        bars = np.arange(n_slices)

        # --- Panel 1: Execution schedules ---
        ax_sched.set_facecolor("#0d1117")

        # TWAP: equal slices
        twap_sched = np.ones(n_slices) * (total_sh / n_slices)

        # VWAP: volume-proportional
        vol_profile = volume / (volume.sum() + 1e-9)
        vwap_sched  = total_sh * vol_profile

        # AC: Almgren-Chriss schedule (pad or trim to n_slices)
        ac_sched_raw = ac["trade_schedule"]
        ac_sched = ac_sched_raw[:n_slices] if len(ac_sched_raw) >= n_slices else \
                   np.pad(ac_sched_raw, (0, n_slices - len(ac_sched_raw)))

        width = 0.25
        ax_sched.bar(bars - width, twap_sched / total_sh * 100,
                     width, label="TWAP",  color="#4488ff", alpha=0.85)
        ax_sched.bar(bars,         vwap_sched / total_sh * 100,
                     width, label="VWAP",  color="#44cc88", alpha=0.85)
        ax_sched.bar(bars + width, ac_sched / (total_sh + 1e-9) * 100,
                     width, label="AC",    color="#ffaa00", alpha=0.85)

        ax_sched.set_title(f"{ticker} — Execution Schedule (% of Order)", color="white", fontsize=11)
        ax_sched.set_xlabel("Bar (Hour)", color="#aaaaaa")
        ax_sched.set_ylabel("% of Total Order", color="#aaaaaa")
        ax_sched.tick_params(colors="#aaaaaa")
        ax_sched.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
        for spine in ax_sched.spines.values():
            spine.set_edgecolor("#333355")

        # Annotate slippage for each method
        txt  = (f"Naive: {res['naive']['slippage_bps']:+.1f} bps\n"
                f"TWAP:  {twap['slippage_bps']:+.1f} bps\n"
                f"VWAP:  {vwap['slippage_bps']:+.1f} bps")
        ax_sched.text(0.98, 0.97, txt, transform=ax_sched.transAxes,
                      va="top", ha="right", color="white", fontsize=9,
                      bbox=dict(facecolor="#1a1a2e", alpha=0.85, edgecolor="#555577"))

        # --- Panel 2: Almgren-Chriss trajectory ---
        ax_traj.set_facecolor("#0d1117")
        t     = ac["t_vals"]
        X     = ac["trajectory"]
        ax_traj.plot(t, X / total_sh * 100, color="#ffaa00", lw=2.5,
                     label="AC Optimal Trajectory")
        ax_traj.fill_between(t, X / total_sh * 100, alpha=0.15, color="#ffaa00")

        # Reference: linear (TWAP) trajectory
        ax_traj.plot(t, np.linspace(100, 0, len(t)), color="#4488ff", lw=1.5,
                     ls="--", alpha=0.7, label="TWAP (linear)")

        # Annotate impact split
        perm_pct = ac["perm_impact_usd"] / (ac["total_impact_usd"] + 1e-9) * 100
        temp_pct = 100 - perm_pct
        impact_txt = (f"Permanent: {ac['perm_impact_usd']:,.0f} USD ({perm_pct:.0f}%)\n"
                      f"Temporary: {ac['temp_impact_usd']:,.0f} USD ({temp_pct:.0f}%)\n"
                      f"Total:     {ac['total_impact_bps']:.1f} bps")
        ax_traj.text(0.98, 0.55, impact_txt, transform=ax_traj.transAxes,
                     va="center", ha="right", color="white", fontsize=9,
                     bbox=dict(facecolor="#1a1a2e", alpha=0.85, edgecolor="#555577"))

        ax_traj.set_title(f"{ticker} — Almgren-Chriss Trajectory", color="white", fontsize=11)
        ax_traj.set_xlabel("Bar (Time)", color="#aaaaaa")
        ax_traj.set_ylabel("Shares Remaining (%)", color="#aaaaaa")
        ax_traj.tick_params(colors="#aaaaaa")
        ax_traj.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
        ax_traj.set_ylim(-5, 110)
        for spine in ax_traj.spines.values():
            spine.set_edgecolor("#333355")

        # --- Panel 3: Five numbers scorecard ---
        ax_tbl.set_facecolor("#0d1117")
        ax_tbl.axis("off")

        def ok(val, thr, higher=False):
            return "✓" if (bool(val > thr) if higher else bool(abs(val) < abs(thr))) else "✗"

        table_data = [
            ["Metric",         "Value",                                   "Pass?"],
            ["Naive slip",     f"{res['naive']['slippage_bps']:+.1f}bps",
             ok(abs(res["naive"]["slippage_bps"]), 10)],
            ["TWAP slip",      f"{twap['slippage_bps']:+.1f}bps",
             ok(abs(twap["slippage_bps"]), 5)],
            ["VWAP slip",      f"{vwap['slippage_bps']:+.1f}bps",
             ok(abs(vwap["slippage_bps"]), 3)],
            ["AC impact",      f"{ac['total_impact_bps']:.1f}bps",
             ok(abs(ac["total_impact_bps"]), 5)],
            ["Part. Rate",
             f"{float(ac['participation_rate'].mean()*100):.1f}%",
             "✓" if bool(float(ac["participation_rate"].mean()) < 0.10) else "✗"],
            ["VWAP>TWAP",
             f"{abs(twap['slippage_bps'])-abs(vwap['slippage_bps']):+.1f}bps",
             "✓" if bool(abs(twap["slippage_bps"]) > abs(vwap["slippage_bps"])) else "✗"],
        ]

        def cell_col(r, c, cell):
            if r == 0: return "#2a2a4e"
            if c < 2:  return "#1a1a2e"
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
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_text_props(color="white")
            cell.set_edgecolor("#333355")

        ax_tbl.set_title(f"{ticker} — Execution Scorecard", color="white", fontsize=11)

    plt.suptitle("Execution Models — TWAP vs VWAP vs Almgren-Chriss\n"
                 "Measuring and minimising the cost of order execution",
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
WHAT YOU LEARNED — EXECUTION MODELS
============================================================

1. WHY EXECUTION MATTERS
   A strategy with gross IC = 0.05 earns ~5bp per trade.
   If execution costs 8bp, the strategy is UNPROFITABLE despite having edge.
   The execution model IS part of the signal research, not an afterthought.

2. TWAP vs VWAP
   TWAP: equal shares each bar. Simple but volume-blind.
   VWAP: shares proportional to volume. Naturally concentrates in liquid hours.
   VWAP beats TWAP when there is a strong intraday volume pattern (open/close spikes).
   They converge when volume is uniform throughout the day.

3. ALMGREN-CHRISS: TWO COSTS
   Permanent: you provide information to the market. Other participants
              re-price. You pay this even if you stop trading.
   Temporary: you demand liquidity. Market makers charge a premium.
              This cost goes away after you stop — prices bounce back.
   Optimal: trade fast enough to avoid timing risk,
            slow enough to avoid excessive impact. λ balances the two.

4. PARTICIPATION RATE
   Your order as a % of market volume per bar.
   > 10-20% participation → market notices you → impact spirals.
   Institutional rule: stay below 10% of market volume.
   For NVDA ($5B/day volume): 10% limit ≈ $500M execution capacity per day.

5. IMPLEMENTATION SHORTFALL
   IS = (avg execution price - decision price) / decision price
   IS captures: spread cost + impact + delay cost (price moved before we executed)
   IS is the TRUE measure of execution quality, better than beating the TWAP/VWAP benchmark.

============================================================
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os, subprocess
    os.makedirs("charts", exist_ok=True)

    print(CONCEPT_SUMMARY)
    print("Running execution model analysis on NVDA and MSFT...")
    print("Order size: $500,000 per position\n")

    tickers = ["NVDA", "MSFT"]
    results = []

    for ticker in tickers:
        res = run_pipeline(ticker, order_size_usd=500_000)
        if res:
            results.append(res)

    if results:
        chart_path = "charts/execution_models.png"
        make_chart(results, save_path=chart_path)
        subprocess.Popen(["open", chart_path])

    # Research decision
    print("\n" + "="*55)
    print("  RESEARCH DECISION")
    print("="*55)
    for res in results:
        twap_slip = res["twap"]["slippage_bps"]
        vwap_slip = res["vwap"]["slippage_bps"]
        ac_bps    = res["ac"]["total_impact_bps"]
        verdict   = []

        improvement = abs(twap_slip) - abs(vwap_slip)
        if improvement > 0:
            verdict.append(f"VWAP saves {improvement:.1f}bps vs TWAP")
        else:
            verdict.append(f"TWAP competitive with VWAP (uniform volume)")

        if bool(ac_bps < 5):
            verdict.append(f"AC impact acceptable ({ac_bps:.1f}bps)")
        else:
            verdict.append(f"AC impact HIGH ({ac_bps:.1f}bps) — reduce order size or extend window")

        part_rate = float(res["ac"]["participation_rate"].mean())
        if bool(part_rate < 0.10):
            verdict.append(f"participation rate safe ({part_rate:.1%})")
        else:
            verdict.append(f"participation rate HIGH ({part_rate:.1%}) — scale down")

        print(f"  {res['ticker']}: {' | '.join(verdict)}")

    print("\nNext: 10_PRETRADE_CHECKLIST.py")
