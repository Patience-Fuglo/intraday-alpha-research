# Research Memo
## Senior Level Studies — Production Validation Framework
### Bullseye Alpha | Patience Fuglo | May 2026
#### Seven standalone research studies: Purged Walk-Forward · Monte Carlo · Portfolio Optimization · Factor Modeling · Microstructure · Execution Models · Pre-Trade Checklist

---

# WHAT LEVEL 4 IS AND WHY IT EXISTS

Levels 1 through 3 answered one question in increasingly sophisticated ways:

```
Level 1:  Does the price pattern exist?
Level 2:  Does it survive real conditions — regime, cost, time?
Level 3:  Can a model learn it systematically?
```

Level 4 asks a completely different question:

```
Level 4:  Can I trust this with real capital?
```

This is not a statistical question. It is a production question. Before a signal
touches a live account, seven independent gates must be cleared — each one designed
to catch a specific class of failure that backtesting alone will not find.

The difference between Level 3 and Level 4 is the difference between a research
finding and a trading decision. Level 3 tells you the signal looks good on paper.
Level 4 tells you whether "looks good on paper" will survive contact with reality.

Each Level 4 study is standalone. Each has its own hypothesis, its own data pipeline,
its own five numbers scorecard, and its own conclusion. None of them requires the others.

---

# THE SEVEN STUDIES — OVERVIEW

```
Study 1 — Purged Walk-Forward      research/04_PURGED_WALK_FORWARD.py
           Was the measured IC real, or was it inflated by label leakage?

Study 2 — Monte Carlo               research/05_MONTE_CARLO.py
           Is the equity curve genuine edge, or lucky trade sequencing?

Study 3 — Portfolio Optimization    research/06_PORTFOLIO_OPTIMIZATION.py
           Does smarter allocation improve Sharpe on high-vol tech stocks?

Study 4 — Factor Modeling           research/07_FACTOR_MODELING.py
           Is the return genuine alpha, or disguised factor exposure?

Study 5 — Microstructure            research/08_MICROSTRUCTURE.py
           What is the TRUE cost of execution after spread and impact?

Study 6 — Execution Models          research/09_EXECUTION_MODELS.py
           Which execution method minimises market impact on a $500k order?

Study 7 — Pre-Trade Checklist       research/10_PRETRADE_CHECKLIST.py
           Does the signal pass the 8-item production gate?
```

---

# STUDY 1 — PURGED WALK-FORWARD WITH EMBARGO

## The Problem It Solves

Standard walk-forward has a hidden flaw: label leakage.

```
Standard Walk-Forward:
  Train on bars 1–100.
  Test on bars 101–130.

  Problem: bar 100's training label = return over bars 100–106.
           Bars 101–106 are inside the TEST set.
           The model was trained on information from the test period.
           This is data leakage disguised as a clean split.
```

The result is that your measured IC and Sharpe are slightly inflated.
Not enough to flip a bad signal to good — but enough to matter.

## The Fix

```
Purge:   Remove the last h training bars whose labels overlap the test window.
         h = forward return horizon = 6 bars (one 30-minute return).
         These bars "know about" the future — remove them from training.

Embargo: Skip the first h test bars after the training cutoff.
         These bars' features were computed partly from training data.
         They are contaminated by proximity — skip them.

Result:  Clean separation. No bar in the training set has any information
         from the test period. No bar in the test set has contamination
         from the training period.
```

## Why This Matters for Production

A model deployed in production retrain at quarter end. The last week of
the quarter is the embargo zone — you do not evaluate performance there
because it is contaminated by training data proximity. Purged walk-forward
replicates this in research so the measured performance matches what you
will actually see live.

## Five Numbers Scorecard

```
Metric            Standard WF    Purged WF     Change
──────────────    ──────────     ─────────     ──────
Gross Return      varies         varies        typically smaller (leakage removed)
IC                slightly high  honest        IC drops ~5–15% after purge
Sharpe            slightly high  honest        Sharpe drops in line with IC
PSR               slightly high  honest        PSR follows Sharpe
Leakage Test      n/a            run           explicit comparison built in
```

## Interview Line

```
"I use purged walk-forward with a 6-bar embargo to prevent label leakage.
Standard walk-forward training labels overlap the test period by the
length of the forward return horizon. Purging removes those bars from
training. The embargo removes the first h test bars that are too close
to the training window. The result is a clean separation — measured IC
matches what you would see in live trading."
```

---

# STUDY 2 — MONTE CARLO P&L SIMULATION

## The Problem It Solves

A strategy can produce a good equity curve purely by luck — if the winning
trades happened to cluster at the right time.

```
Signal has 200 trades.
100 winners, 100 losers.
If the 100 winners cluster in months 1–3 and losses cluster in months 4–6:
  → equity curve looks great early and terrible late.
  → we might deploy after month 3 and get all the losses.

If the 100 winners are randomly distributed across months 1–6:
  → equity curve is stable and representative.

The backtest does not tell you which scenario you are in.
Monte Carlo does.
```

## The Method

```
Step 1:  Run the real strategy. Collect all 200 trade returns.
         [+0.003, -0.001, +0.005, -0.002, ...]

Step 2:  Bootstrap 1,000 simulated sequences.
         Each: randomly draw 200 returns with replacement from the real list.
         Different order each time = different equity curve each time.

Step 3:  Compound each sequence into an equity curve.
         Stack all 1,000 curves. This is the distribution of possible futures
         given this signal edge and this trade count.

Step 4:  Read the distribution:
         P50 = median outcome (expected)
         P5  = bad-luck scenario (5th percentile)
         P95 = good-luck scenario (95th percentile)
         P(ruin) = fraction of paths that hit -20% drawdown at any point
```

## What Each Number Tells You

```
P50 > 0:        The signal has positive expected value — edge is real.
P5  > -15%:     Even bad luck stays within tolerable drawdown.
P95:            Do not plan for this. It is the ceiling, not the floor.
P(ruin) < 5%:   Ruin is rare enough to accept at this position size.
                If P(ruin) > 5%: REDUCE POSITION SIZE, not the signal.
```

## The Key Insight

```
Monte Carlo does not predict the future.
It maps the RANGE of outcomes given a fixed signal edge.

If P50 is positive but P5 is -40%, the signal has edge but the
position size is too aggressive. The fix is not to change the signal.
The fix is to reduce size until P5 is acceptable.

P5 drives position sizing. Not P50. Not the backtest return.
"What is the worst plausible outcome?" — that is the question.
```

## Interview Line

```
"I run 1,000 bootstrap simulations by resampling my trade returns with
replacement. The 5th percentile path — the bad-luck scenario — is what
I use to size positions. If P(ruin below -20%) is above 5%, the position
is too large regardless of signal quality. I halve the position size
until P(ruin) falls below 5%. Same signal, safer sizing."
```

---

# STUDY 3 — PORTFOLIO OPTIMIZATION

## The Problem It Solves

Equal-weighting two assets looks balanced. It is not.

```
NVDA annual vol ≈ 60%
MSFT annual vol ≈ 25%

Equal weight: 50% NVDA, 50% MSFT.

Risk contribution of NVDA:
  0.5² × 0.60²
  ─────────────────────────────── ≈ 85%
  0.5² × 0.60² + 0.5² × 0.25²

You think you are 50/50. You are 85/15 in risk terms.
NVDA dominates the portfolio variance entirely.
```

## Four Methods Compared

```
Equal Weight (EW):
  w_i = 1/N for all assets.
  Simple. Baseline. Rarely optimal.

Minimum Variance (MV):
  Minimise w^T Σ w subject to sum(w) = 1, w ≥ 0.
  Ignores expected returns — only needs the covariance matrix.
  More stable out-of-sample than Max-Sharpe because covariance
  is estimated with less noise than expected returns.

Maximum Sharpe (Tangency Portfolio):
  Maximise μ^T w / sqrt(w^T Σ w).
  Optimal IF expected return estimates are accurate.
  In practice: IC ≈ 0.05 means our return forecasts are very noisy.
  Garbage in → garbage out. Rarely beats Min-Variance out-of-sample.

Risk Parity (RP):
  Each asset contributes equally to portfolio variance.
  w_NVDA ∝ 1/σ_NVDA, w_MSFT ∝ 1/σ_MSFT.
  Corrects the 85/15 problem above.
  Popular at systematic funds (Bridgewater All Weather uses this).
```

## The Diversification Ratio

```
Diversification Ratio = (Σ w_i × σ_i) / σ_portfolio

If DR > 1: combining assets reduced vol below the weighted average.
           This is the benefit of diversification.
If DR ≈ 1: assets are highly correlated. Little diversification benefit.

Risk Parity typically achieves the highest DR because it is
explicitly designed to balance risk contributions across assets.
```

## Interview Line

```
"Min-Variance often beats Max-Sharpe out-of-sample. The reason:
to run Max-Sharpe you need reliable expected return estimates.
With IC of 0.05, my forecasts carry high estimation error.
The covariance matrix is estimated with less noise.
So Min-Variance, which only needs the covariance matrix, tends
to be more robust. In my tests, Risk Parity improved portfolio
Sharpe vs equal weight by correcting the hidden 85/15 risk split."
```

---

# STUDY 4 — FACTOR MODELING

## The Problem It Solves

A strategy can appear to have alpha when it is actually just riding
a known factor. If the market is up 10% and your strategy is up 10%,
you have not demonstrated skill — you have demonstrated market exposure.

```
Factor model:
  strategy_return = α + β₁·MKT + β₂·MOM + β₃·VOL + ε

  α (alpha)    = return NOT explained by factors = genuine skill
  β (beta)     = sensitivity to each factor
  R²            = fraction of strategy variance explained by factors
  ε             = residual (unexplained) return
```

## The Three Factors We Test

```
Market (MKT):    hourly return of SPY.
                 Tests: are you just riding the market?

Momentum (MOM):  recent trend direction in SPY.
                 Tests: are you just buying what went up?

Volatility (VOL): realised vol of SPY.
                 Tests: does your strategy only work in high-vol regimes?
```

## What Good Results Look Like

```
α > 0, t-stat(α) > 2.0:   genuine alpha, statistically significant.
R² < 0.30:                 factors explain less than 30% of variance.
                           strategy is idiosyncratic, not factor-driven.
β_mkt < 0.30:              near-market-neutral.
                           P&L driven by signal, not market direction.
IR > 0.5:                  alpha per unit of residual risk is acceptable.
```

## The Market-Neutral Requirement

```
For an intraday long/short strategy, β_mkt should be near zero.
High market beta means your P&L goes up when SPY goes up and down
when SPY goes down. That is not a signal — that is a long-only bet.

Target: |β_mkt| < 0.30.
If β_mkt is high: add a market hedge (short SPY ETF proportionally).
```

## Interview Line

```
"After regressing strategy returns on market, momentum, and vol factors,
I check that alpha is positive with t-stat > 2, R² is low (strategy
is idiosyncratic), and market beta is below 0.30. High R² is a red flag —
it means the strategy is disguised factor exposure. A genuine signal
should have most of its return in the unexplained residual."
```

---

# STUDY 5 — MARKET MICROSTRUCTURE

## The Problem It Solves

Backtests use mid-price. Real trading happens at ask (when you buy)
and bid (when you sell). The gap between mid-price and execution price
is the microstructure cost — and it is LARGER than it looks.

```
Gross IC:  0.054 on NVDA
Cost:      half-spread ≈ 0.06%

Every trade leaks 6bp just crossing the spread.
On a 30-minute hold with 5bp average gross profit per trade:
  Gross per trade: +5bp
  Spread cost:     -6bp
  Net:             -1bp

The signal has edge. The execution kills it.
```

## Four Metrics

```
Roll's Spread Estimate (Roll 1984):
  Infers bid-ask spread from return autocorrelation.
  If prices bounce between bid and ask, consecutive returns are
  negatively correlated. spread = 2 × sqrt(-cov(Δp_t, Δp_{t-1}))
  Works with OHLCV data — no tick data needed.

Amihud Illiquidity (Amihud 2002):
  ILLIQ = |return| / (price × volume)
  = price impact per dollar of volume traded.
  High Amihud → larger impact from your own orders.
  Check before sizing: higher Amihud = smaller safe position.

Order Flow Imbalance (OFI):
  OFI = (buy volume - sell volume) / total volume
  Positive OFI = institutional buyers are net active.
  High OFI Z-score predicts short-term price continuation.
  This is the intraday version of the momentum signal.

Corwin-Schultz (2012):
  High-low spread estimator. Better than Roll when the
  bid-ask bounce assumption is violated.
  Uses: spread ≈ f(H/L ratio over one and two bars).
  Best estimator available without tick-level data.
```

## The Cost-Adjusted IC

```
Net IC ≈ Gross IC × (1 - cost_fraction)
where cost_fraction = half_spread / (fwd_return_vol + half_spread)

If gross IC = 0.05 and half-spread eats 20% of fwd return vol:
  Net IC ≈ 0.05 × 0.80 = 0.040 — below threshold.

The signal has gross edge but NOT net edge at this cost level.
Fix: reduce frequency (hold longer to amortise the spread cost)
     or reduce spread by using limit orders instead of market orders.
```

## Interview Line

```
"I estimated bid-ask spread using Roll's method from return autocorrelation
and Corwin-Schultz from the high-low ratio. For NVDA, half-spread was ~6bp
vs ~3bp for MSFT. Combined with Amihud illiquidity, NVDA's true round-trip
cost was ~14bp. With gross IC of 0.05 generating roughly 5bp per trade,
the signal is net-negative on NVDA without execution improvement. This
tells me to either increase the holding period or switch to limit orders."
```

---

# STUDY 6 — EXECUTION MODELS

## The Problem It Solves

A signal tells you WHAT to trade. Execution tells you HOW to trade it.
Sending one large market order announces your intention to the market.
Other participants see the order flow and trade against you.

```
$500,000 NVDA order sent all at once:
  Your buy pressure temporarily pushes NVDA up while you're filling.
  Other algo traders see the large order and front-run it.
  You paid more than you needed to — permanent impact.
```

## Three Execution Methods

```
Naive (single shot):
  Execute all shares at once at the open.
  Worst impact. Best for tiny orders on very liquid stocks.
  Baseline for measuring improvement.

TWAP (Time-Weighted Average Price):
  Divide Q shares into N equal slices.
  Execute one slice every k minutes.
  Simple. Predictable. Ignores volume profile.
  Flaw: executes the same size in thin windows (10:45am) as
        in liquid windows (10:00am). Impact higher in thin windows.

VWAP (Volume-Weighted Average Price):
  Divide Q shares proportionally to volume per bar.
  q_t = Q × (volume_t / total_volume_in_window)
  Concentrates execution in high-liquidity windows.
  Lower impact because you are a constant % of market flow.
  Better than TWAP when volume profile is uneven (it usually is).
```

## Almgren-Chriss — The Optimal Method

```
Almgren & Chriss (2000) proved there exists an OPTIMAL trajectory
that minimises: E[cost] + λ × Var[cost]
where λ = risk aversion parameter.

Two cost types:
  Permanent impact: you provide information to the market.
                    Other participants re-price permanently.
                    You cannot recover this cost even if you stop trading.
                    Cost = η × (q_t / V_t) × P × q_t

  Temporary impact: you demand liquidity.
                    Market makers charge a premium to take the other side.
                    Price bounces back after you stop — this cost is recoverable.
                    Cost = γ × (q_t / V_t) × P × q_t

Optimal trajectory: X*(t) = Q × sinh(κ(T−t)) / sinh(κT)
  κ = sqrt(λσ² / γ)  — decay rate controlled by risk aversion λ.

  λ → 0:    VWAP (no risk penalty → spread execution evenly)
  λ → ∞:    immediate execution (risk-averse → execute now, ignore cost)
  λ between: front-loaded hyperbolic decay — execute faster early
             to reduce timing risk while avoiding peak impact.
```

## The Participation Rate Rule

```
Your order ÷ market volume per bar = participation rate.
Keep below 10%. Above 10%, market participants notice and trade against you.

NVDA daily volume ≈ $5 billion.
10% limit ≈ $500 million execution capacity per day.
Our $500k order = 0.01% participation — safe.

For a hedge fund running $10 billion: NVDA capacity = $500M/day.
Above that, market impact grows rapidly. This is why large funds
cannot just run small-cap signals at scale.
```

## Interview Line

```
"VWAP beats TWAP because it matches the market's natural volume pattern.
In the first hour of US trading, volume is typically 30–40% of daily total.
VWAP concentrates execution there where impact is lowest. TWAP spreads
uniformly, so it executes the same size in thin mid-day periods where
impact per share is higher. Almgren-Chriss gives the theoretically
optimal trajectory — front-loaded to reduce timing risk while staying
below 10% participation rate to limit market impact."
```

---

# STUDY 7 — PRE-TRADE CHECKLIST

## Why a Checklist Exists

```
Without a checklist, deployment decisions are emotional.
"This backtest looks great — let's trade it."

The checklist removes emotion from the decision.
If any critical gate fails, the signal does not trade.
No exceptions. No overrides.
```

## The 8-Item Gate

```
CRITICAL — any failure = NO-GO, full stop:

  1. Gross Return > 0
     If gross is negative, the signal is anti-predictive.
     All subsequent analysis is irrelevant. Stop here.

  2. Net Return > 0
     Gross edge must survive costs.
     If net ≤ 0: redesign the signal or reduce execution costs.

  3. IC > 0.05
     Predictions must correlate meaningfully with actual returns.
     IC < 0.05: signal is noise. Rebuild features.

  4. PSR > 95%
     Sharpe must be statistically real, not sampling noise.
     PSR < 95% with 60 days of data → not confirmed. Need more data.

  5. Max Drawdown > −20%
     Equity curve must not blow up.
     DD below −20% → position size is too large for this signal.

ADVISORY — failures noted, monitoring plan required:

  6. DSR > 95%
     Sharpe must survive multiple testing correction.
     DSR < 95% → observed Sharpe within chance range of 15 trials.
     Action: collect more data, do not expand to new tickers yet.

  7. Trade Count ≥ 50 per fold
     Statistics are unreliable below 50 observations.
     < 50 trades → every metric has wide confidence intervals.
     Action: wait for more history before trusting the numbers.

  8. P(ruin < −20%) < 5%
     Monte Carlo ruin probability must be manageable.
     > 5% → reduce position size, not the signal.
```

## The Three Verdicts

```
GO     All 5 critical pass + ≥2 advisory pass.
       Action: deploy at target position size. Retrain quarterly.

WATCH  All 5 critical pass + 1 or more advisory fail.
       Action: deploy at 25% of target. Monitor weekly.
               Document which advisory items failed and why.
               Set a date to re-evaluate with more data.

NO-GO  Any critical gate fails.
       Action: do not deploy. Fix the specific failure first.
               Investigate root cause — data bug? Feature decay?
               Cost model wrong? Only resubmit after root cause fixed.
```

## Our NVDA Result

```
Critical gates:    5 of 5 PASS on yfinance walk-forward
Advisory gates:    DSR FAIL (too few trades), P(ruin) borderline

Verdict: WATCH

Interpretation:
  The signal has a measurable gross edge and the Sharpe is statistically
  real (PSR > 95%). But 60 days of yfinance data produces only ~20 trades
  per fold — too few for DSR confirmation and Monte Carlo reliability.

  Required action: QuantConnect 4.5-year backtest.
  With 4.5 years of minute data, trade count rises from 20 to ~1,500+.
  This is why QUANTCONNECT_ML_RIDGE.py exists.

QuantConnect result: IC turned negative on NVDA (regime change post-2022).
                     Net return positive (+21%) but Sharpe too low (0.127).
                     Conclusion: feature redesign required before deployment.
```

---

# CROSS-STUDY SYNTHESIS

What the seven studies say together about the NVDA ML Ridge signal:

```
Study 1 — Purged WF:      IC is honest, not inflated by leakage.
                           The measured edge is real, even if small.

Study 2 — Monte Carlo:     P(ruin) is high at yfinance data scale.
                           Not because the signal is bad — because
                           60 days and 20 trades is not enough to
                           distinguish signal from noise statistically.

Study 3 — Portfolio Opt:   Adding MSFT with risk parity weighting
                           improves portfolio Sharpe vs equal weight.
                           The two signals are not perfectly correlated.

Study 4 — Factor Model:    Strategy is market-neutral (β_mkt ≈ 0.03).
                           Low R² (< 0.02) — return is idiosyncratic.
                           Alpha t-stat is not significant at 60-day scale.

Study 5 — Microstructure:  NVDA half-spread ≈ 6bp, MSFT ≈ 3bp.
                           At 5bp gross IC per trade, NVDA is net-negative.
                           Longer holding period or limit orders required.

Study 6 — Execution:       VWAP reduces impact vs TWAP on NVDA.
                           Participation rate is safe at $500k order size.

Study 7 — Checklist:       WATCH verdict. All critical gates pass.
                           DSR and trade count advisory gates fail.
                           QuantConnect validation required before deployment.
```

**Overall conclusion:** The signal has a measurable gross edge on NVDA. The statistical confirmation problem is a data volume problem, not a signal quality problem. The fix is longer history (QuantConnect), not feature redesign. The execution cost problem on NVDA is real — limit orders or longer holding period required to make net IC positive.

---

# THE INTERVIEW FRAMEWORK FOR LEVEL 4

When a systematic fund interviewer asks about your validation process,
this is the answer structure:

```
Step 1 — Leakage test
  "I use purged walk-forward with a 6-bar embargo to ensure no label
   overlap between training and test periods."

Step 2 — Path distribution
  "I run 1,000 bootstrap Monte Carlo paths. The 5th percentile determines
   position sizing. P(ruin) must be below 5%."

Step 3 — Allocation
  "I combine signals using risk parity weights, not equal weight.
   High-vol assets like NVDA are underweighted proportionally to their
   inverse volatility to balance risk contributions."

Step 4 — Factor check
  "I regress strategy returns on market, momentum, and vol factors.
   I require alpha t-stat > 2, R² < 0.30, and market beta < 0.30."

Step 5 — Cost reality
  "I estimate true execution cost using Roll's spread and Amihud
   illiquidity. I verify that net IC remains positive after cost adjustment."

Step 6 — Execution plan
  "I use VWAP execution to match market volume profile and keep
   participation rate below 10%. I model impact using Almgren-Chriss."

Step 7 — Production gate
  "Every signal passes through an 8-item checklist: 5 critical stops
   and 3 advisory monitors. A single critical failure blocks deployment."
```

This is not a list of concepts you memorised. This is a process you ran.
The code is in `research/`. The charts are in `charts/`. The conclusion is
documented above. You can walk an interviewer through every step.

---

# PAPER TRADING — PRODUCTION DEPLOYMENT

## What Paper Trading Is

```
Paper trading runs the live algorithm on real-time market data
with simulated capital — no real money at risk.

Purpose: verify that the backtest edge persists in live conditions
         before committing real capital.

Why it matters:
  Backtests assume  →  perfect fills at bar close price
  Live trading has  →  latency, partial fills, real spread, order book depth
  Paper trading     →  exposes the gap before it costs money
```

## How to Deploy `QUANTCONNECT_ML_RIDGE.py` in Paper Mode

```
1. Log into quantconnect.com
2. Open the project containing QUANTCONNECT_ML_RIDGE.py
3. Click Deploy Live (top right of the IDE)
4. Under Brokerage, select QuantConnect Paper Trading
5. Set Cash to $100,000 (matches backtest starting capital)
6. Click Deploy
```

## What to Monitor After Deployment

| Metric | Backtest Value | Alert If |
|--------|---------------|----------|
| IC (NVDA) | −0.047 | Stays below 0 for 4+ consecutive weeks |
| IC (MSFT) | +0.034 | Drops below 0 consistently |
| Win rate | 51% | Falls below 45% for 30+ days |
| Fee drag | ~$80/month | Exceeds $150/month |
| Max drawdown | −27.3% | Approaches −15% (reduce size immediately) |
| Model retrain | Every 63 days | Confirm log entry fires on schedule |

## Decision Rules After 30 Days

```
MSFT IC holds positive  →  reduce position from 45% to 25%/ticker → redeploy
Both ICs negative       →  feature redesign required, pause paper trading
Drawdown > −15%         →  halt, review signal before continuing
```

## What to Expect

```
Live IC will likely be LOWER than backtest IC.
Reason: backtest fills at bar close, live fills with latency and spread.

If live MSFT IC holds above 0.02 (not 0.034) → acceptable degradation.
If live MSFT IC falls to 0 or below → the regime mismatch from the
QuantConnect backtest is confirmed. Feature redesign required before
any live capital deployment.

The minimum viable path to live trading:
  Paper trading 30 days → MSFT IC positive → reduce position concentration
  → redeploy paper → 30 more days → if stable → live with $10,000 pilot
```

---

*Bullseye Alpha | Systematic Equity Research | bullseyealpha.com*
