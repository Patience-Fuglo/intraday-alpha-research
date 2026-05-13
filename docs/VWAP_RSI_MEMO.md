# Alpha Research Memo
## Intraday Mean Reversion Signal — VWAP + RSI
### Bullseye Alpha | Patience Fuglo | May 2026
#### Status: Closed — No statistically significant edge found (PSR < 1% across all versions)

---

# CHAPTER 1 — THE FIRST HYPOTHESIS

Every research programme starts with an observation.

The observation here was simple: intraday prices on AAPL seemed to drift away from
their daily average and then return. When price dropped sharply below where most
participants had traded that day, it tended to recover. When it rose sharply above,
it tended to fall back.

That observation became Hypothesis 1.

**What is Mean Reversion?**
```
Mean reversion is the belief that price cannot stay far from fair value forever.
When a rubber band is stretched too far  →  it snaps back.
When a stock price strays too far from average  →  it snaps back.

The bet: "This price is too extreme. It will return to normal."

Mean reversion works BEST in:  sideways, choppy, range-bound markets
Mean reversion fails BADLY in:  trending markets — price just keeps going
```

**The Hypothesis:**
```
HYPOTHESIS
──────────
Intraday prices dislocate from VWAP (intraday fair value) and mean-revert.
RSI extremes identify the moment of maximum dislocation.

Long:  price BELOW VWAP  AND  RSI < 25  →  buy, expect recovery
Short: price ABOVE VWAP  AND  RSI > 75  →  sell, expect decline
Exit:  price crosses back through VWAP  →  reversion complete

Instrument: AAPL (first), IWM (out-of-sample)
Frequency:  5-minute bars
Platform:   QuantConnect LEAN, Interactive Brokers cost model
Window:     Jan 2020 – Jun 2024 (4.5 years, $100,000 capital)

The bet: "Price has moved too far. It will come back."
```

---

# CHAPTER 2 — WHAT EACH TOOL DOES

**What is VWAP?**
```
VWAP = Volume-Weighted Average Price
     = the average price every market participant paid today,
       weighted by the number of shares they traded

Formula:
  VWAP = Σ(typical_price × volume) / Σ(volume)
  typical_price = (high + low + close) / 3

It is intraday fair value.
Think of it as the market's answer to: "What is a fair price right now?"

Resets at 9:30am every trading day. Reflects only that day's activity.
By 4pm it represents the full day's consensus.

If price is BELOW VWAP  →  stock is cheap relative to today's average
If price is ABOVE VWAP  →  stock is expensive relative to today's average

This is why institutions reference VWAP as a benchmark:
beating VWAP on a buy means you paid less than average.
```

**What is RSI?**
```
RSI = Relative Strength Index (0 to 100)
    = measures how extreme recent price moves have been

Formula:
  avg_gain  = rolling mean of positive returns over N bars
  avg_loss  = rolling mean of negative returns over N bars (absolute)
  RS        = avg_gain / avg_loss
  RSI       = 100 - (100 / (1 + RS))

RSI < 25  →  extreme oversold   →  recent selling dominated, move may be exhausted
RSI > 75  →  extreme overbought →  recent buying dominated, move may be exhausted
Between   →  neutral            →  no clear extreme

RSI is the CONFIRMATION.
VWAP tells you WHERE price is relative to fair value.
RSI tells you HOW EXTREME the current move has been.
Both must agree before entering.
```

**Why NOT one condition alone:**
```
VWAP alone:  price is below fair value. So what?
             It could keep going lower — a downtrend is still below VWAP.

RSI alone:   readings are extreme. On which asset? In which direction?
             RSI below 30 in a free-falling stock means nothing alone.

Together:
  Price is below VWAP  (dislocated from fair value)
  AND RSI is below 25  (the selling pressure is exhausted)
  = This is the best setup for a snap-back.

One condition is noise. Two conditions pointing the same direction are signal.
```

**What is the expected P&L structure:**
```
Win:  price snaps back to VWAP  →  +0.3% to +0.5% gain
Loss: price keeps falling        →  -0.5% to -1.0% loss (stopped out)

For the strategy to make money:
Win rate × Avg win  >  Loss rate × Avg loss

That is EV — Expected Value.
EV positive = money made per trade on average.
EV negative = money lost per trade on average, no matter what win rate says.
```

---

# CHAPTER 3 — THE LEVERS

Before running a single version, identify what can be tuned and WHY.

The correct question is: "What causes a better or worse mean-reversion setup?"

```
LOGICAL LEVERS FOR MEAN REVERSION:
┌──────────────────────┬──────────────────────────────────────────────────────┐
│ Lever                │ What it controls                                      │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ RSI threshold        │ How extreme does RSI need to be before entry?         │
│ (25, 30, 20)         │ Looser = more trades, weaker signals                  │
│                      │ Tighter = fewer trades, stronger signals               │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ VWAP distance filter │ How far from VWAP must price be?                      │
│ (0.1%, 0.3%, 0.5%)   │ Prevents entries on tiny dislocations                 │
│                      │ Larger distance = higher quality, fewer trades         │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Stop loss            │ Max acceptable loss per trade                         │
│ (0.5%, 0.75%, 1.0%)  │ Tighter = exits losers faster, reduces drawdown       │
│                      │ Does not create edge — only manages risk               │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Regime filter        │ Is the market environment right for mean reversion?   │
│ (ATR + SMA200)       │ ATR: is volatility low enough to fade?                │
│                      │ SMA200: is price near trend (no runaway move)?        │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

**What is a regime filter:**
```
A regime filter asks: "Is the current market environment suitable for this signal?"

Mean reversion needs  →  calm, range-bound market (rubber band can snap back)
Mean reversion fails  →  trending market (rubber band just stretches further)

How to detect a calm market:
  ATR/Price < 2.5%     →  volatility is low — moves are manageable
  Price within SMA200  →  no major trend active — price is ranging

When both conditions are true → market is safe for mean reversion entries.
When either fails            → do not trade, regime is hostile.

The regime filter does NOT create edge.
It removes hostile environments where the signal fires in the wrong direction.
```

**NOT logical levers for mean reversion:**
```
Volume filter  →  volume tells you about momentum (who is committing to the move)
               →  mean reversion does NOT need momentum — it bets against it
               →  belongs in ORB, not VWAP+RSI

Time window    →  mean reversion can happen at any time of day
               →  ORB uses a specific 10am–11am window because
                  that is when institutional momentum is strongest
               →  mean reversion has no such window

These are momentum levers. Borrowed from ORB. Do not mix frameworks.
```

---

# CHAPTER 4 — RUN HISTORY (FIVE NUMBERS EACH VERSION)

Every run changes ONE variable. Read the five numbers in sequence.

**The Five Numbers Framework:**
```
1. Gross Return   →  does the signal have edge BEFORE fees?
                     If gross is negative → close hypothesis. Stop here.
                     Costs, stops, and filters cannot fix a negative-gross signal.

2. Total Costs    →  what is the fee drag?
                     Commission + slippage on every entry and exit.

3. Net Return     →  gross minus costs — the real result.

4. Win Rate       →  what fraction of trades were winners?
                     Must be read WITH P/L ratio, not alone.

5. PSR            →  is the Sharpe ratio statistically real?
                     Below 95% = the result is within noise, cannot trust it.
```

---

## v1 — Baseline (no filter)

**What changed:** Pure VWAP+RSI signal. No regime filter. No stop loss. All signals traded.

**THE FIVE NUMBERS:**
```
                               WHAT IT MEASURES                     VERDICT
Gross Return    −14.40%       DOES THE SIGNAL HAVE EDGE?           Negative = NO EDGE   ✗
Total Costs     $4,667        WHAT DID FEES COST?                  High — many trades   ✗
Net Return      −14.40%       REAL RESULT AFTER ALL COSTS?         Below 0% = LOSS      ✗
Win Rate        —             WHAT FRACTION WERE WINNERS?          Not recorded         —
PSR             0.107%        IS SHARPE STATISTICALLY REAL?        Below 95% = NOISE    ✗
```

**Reading:**
```
The signal lost money BEFORE costs.
Gross is negative — this is the critical signal.
Regime did not matter: VWAP mean reversion fired in trending markets
and faded in the wrong direction.

What happened in practice:
Price drops below VWAP  →  signal says BUY (expects snap back)
Market says             →  NO, this is a downtrend, keep going
Price keeps falling     →  losing trade

Adding filters to a gross-negative signal does not help.
There is no edge to uncover.
But v1 does not close the hypothesis — this could be the wrong environment,
not the wrong signal. The regime filter must be tested first.
```

**Decision:** Add regime filter. Prevent trading in trending conditions.

---

## v2 — Regime filter added (ATR volatility + SMA200 trend)

**One change made:** ATR/Price < 2.5% AND price within 3% of SMA200. Only trade calm, ranging conditions.

**THE FIVE NUMBERS:**
```
                               WHAT IT MEASURES                     VERDICT
Gross Return    −2.33%        DOES THE SIGNAL HAVE EDGE?           Still negative       ✗
Total Costs     $1,626        WHAT DID FEES COST?                  65% reduction        ✓
Net Return      −2.33%        REAL RESULT AFTER ALL COSTS?         Below 0% = LOSS      ✗
Win Rate        53%           WHAT FRACTION WERE WINNERS?          More than half       ✓
P/L Ratio       0.85          AVG WIN vs AVG LOSS                  Below 1.0 = LOSING   ✗
PSR             0.295%        IS SHARPE STATISTICALLY REAL?        Below 95% = NOISE    ✗
```

**Expected Value per trade:**
```
EV  =  (Win Rate × Avg Win)  +  (Loss Rate × Avg Loss)
EV  =  (53% × +0.28%)  +  (47% × −0.33%)
EV  =  +0.148%  −  0.155%
EV  =  −0.007% per trade   ← strategy loses on every trade on average
```

**Reading:**
```
The regime filter was the most important structural improvement in the research.
It removed 65% of fees by preventing trades in hostile environments.
It lifted net return from −14.4% to −2.33%.

But the core problem remained:
  53% win rate  →  sounds good, more than half are winners
  0.85 P/L ratio → average loss (0.33%) exceeds average win (0.28%)
  EV = −0.007%  → loses a fraction of a percent on every single trade

A 53% win rate with a 0.85 P/L ratio is a losing strategy.
No amount of filtering or parameter tuning fixes negative EV.
This is a signal design problem, not a parameter problem.

Win rate is meaningless without P/L ratio.
Both must be read together.
```

**Decision:** Tighten stop loss to reduce average loss size.

---

## v3 — Stop loss tightened (1.0% → 0.5%)

**One change made:** Exit if trade loses 0.5% instead of 1.0%.

**THE FIVE NUMBERS:**
```
                               WHAT IT MEASURES                     VERDICT
Gross Return    −4.32%        DOES THE SIGNAL HAVE EDGE?           Worse than v2        ✗
Total Costs     $1,788        WHAT DID FEES COST?                  Similar to v2        →
Net Return      −4.32%        REAL RESULT AFTER ALL COSTS?         Below 0% = LOSS      ✗
Win Rate        50%           WHAT FRACTION WERE WINNERS?          Dropped from v2      ✗
P/L Ratio       0.95          AVG WIN vs AVG LOSS                  Slight improvement   →
PSR             0.132%        IS SHARPE STATISTICALLY REAL?        Below 95% = NOISE    ✗
```

**Reading:**
```
Tightening the stop reduced the size of losing trades (P/L improved 0.85 → 0.95).
But it also cut winning trades early:
  Price would hit the 0.5% stop before the reversion completed.
  Net return got WORSE, not better.

This is the classic stop-loss trap:
  Tighter stop  →  exits losers faster     ✓
  Tighter stop  →  also exits winners early ✗
  Net effect    →  changes WHEN you lose, not WHETHER you lose

If the signal does not have edge, the stop only changes the shape of the loss.
It does not create a path to profitability.
```

**Decision:** Try faster RSI to capture more extreme readings.

---

## v4 — RSI period shortened (14 → 7)

**One change made:** RSI window from 14 bars to 7 bars. Faster indicator, more extreme readings.

**THE FIVE NUMBERS:**
```
                               WHAT IT MEASURES                     VERDICT
Gross Return    −21.30%       DOES THE SIGNAL HAVE EDGE?           Much worse — anti-predictive ✗
Total Costs     $3,214        WHAT DID FEES COST?                  More trades = more fees      ✗
Net Return      −21.30%       REAL RESULT AFTER ALL COSTS?         Worst result so far          ✗
Win Rate        —             WHAT FRACTION WERE WINNERS?          Not recorded                 —
PSR             0.000%        IS SHARPE STATISTICALLY REAL?        Zero — purely random         ✗
```

**Reading:**
```
Faster RSI fired on noise.

RSI(14) on 5-minute bars  →  measures 70 minutes of movement  →  meaningful exhaustion
RSI(7)  on 5-minute bars  →  measures 35 minutes of movement  →  a normal intraday wiggle

A 35-minute move reaching extreme RSI is NOT exhaustion.
It is price doing what it normally does in the morning.
The result: more signals, more costs, and predictions pointing the wrong direction.

Key lesson: more signals ≠ better signals.
Frequency and quality are independent properties of a signal.
Making the indicator faster made the signal WORSE because the time horizon
of the indicator no longer matched the time horizon of the phenomenon.
```

**Decision:** Extend to IWM — test whether the hypothesis works on ETFs.

---

## v5a / v5b — IWM (out-of-sample instrument test)

**What changed:** Applied best AAPL settings to IWM (Russell 2000 ETF) as out-of-sample test.

**THE FIVE NUMBERS — v5a (RSI 7, stop 0.75%):**
```
                               VERDICT
Net Return      −22.53%       LOSS — worse than every AAPL version   ✗
PSR             0.001%        Zero — pure noise                       ✗
```

**THE FIVE NUMBERS — v5b (RSI 14, stop 1.0% — best AAPL settings):**
```
                               VERDICT
Net Return      −12.22%       LOSS — worse than AAPL v2 (−2.33%)     ✗
PSR             0.005%        Near zero — not real                    ✗
```

**Why IWM structurally underperformed:**
```
AAPL intraday moves = IDIOSYNCRATIC events
  Earnings, product news, analyst upgrades, short squeezes
  → create dislocations LOCAL to one stock
  → when the event resolves, price returns to VWAP
  → mean reversion can work here

IWM intraday moves = MACRO FLOWS
  Federal Reserve decisions, risk-on/risk-off rotation, sector fund flows
  → when macro drives IWM down, it stays down until the event resolves
  → there is no local reversion pulling it back
  → mean reversion cannot work here

SAME signal. DIFFERENT instrument. STRUCTURALLY different result.

This is not noise. This is a category mismatch:
VWAP mean reversion requires idiosyncratic noise to fade.
ETF moves are driven by systematic macro forces that persist.
```

**VERDICT:**
```
Hypothesis 1 closed.
PSR < 1% across all 6 versions.
No statistically detectable edge on AAPL or IWM.
Documented and archived. Research continues.
```

---

# CHAPTER 5 — WHY THE HYPOTHESIS FAILED

Four root causes, not one.

**Failure 1 — Wrong market environment:**
```
Mean reversion works in range-bound, choppy markets.
Jan 2020–Jun 2024 included:
  COVID crash (2020)     →  extreme trending down
  Tech rally (2021)      →  extreme trending up
  Rate-hike bear (2022)  →  extreme trending down
  AI bull run (2023–24)  →  extreme trending up

More than half the period was strongly trending.
The regime filter helped, but could not remove all trending periods.
The environment was fundamentally wrong for this signal.
```

**Failure 2 — Negative expected value per trade:**
```
P/L ratio < 1.0 across every version.
Average losing trade exceeded average winning trade.

EV = (win_rate × avg_win) + (loss_rate × avg_loss)
   = (53% × +0.28%) + (47% × −0.33%)
   = −0.007% per trade

No amount of filtering or parameter tuning fixes negative EV.
This is a signal design problem.
The wins are too small and the losses are too large.
```

**Failure 3 — VWAP+RSI is publicly known:**
```
This signal appears in every retail trading tutorial.
If thousands of traders run the same signal at the same time,
the edge is arbitraged away.

The dislocation gets filled before any individual trader can act.
Real edge requires signals others are not running.
Publicly documented signals have diminishing returns as more participants use them.
```

**Failure 4 — RSI measures momentum, not exhaustion:**
```
RSI below 30 means: recent selling has dominated.
It does NOT mean: selling is finished.

In a downtrend, RSI can stay below 30 for weeks.

Using RSI as an exhaustion signal in a trending market is a category error.
The tool is being applied in the wrong context.
RSI identifies momentum extremes — not reversal points.
```

---

# CHAPTER 6 — WHAT THE RESEARCH PRODUCED

Closing a hypothesis is not failure. It is the research process working correctly.

**Finding 1 — Regime filter is essential:**
```
The ATR + SMA200 filter was the only structural improvement.
It reduced fees 65% and cut gross loss from −14.4% to −2.33%.
This filter was carried forward into every subsequent signal.
It reappears in ORB (v2) and ML Ridge (regime awareness).
```

**Finding 2 — Gross must be positive before anything else matters:**
```
v1 had negative gross. Stop losses and regime filters could not fix it.

Rule established for all future research:
  If gross < 0  →  close hypothesis immediately.
  Never optimise a losing signal. Redesign it.
```

**Finding 3 — Win rate without P/L ratio is meaningless:**
```
53% win rate sounds good.
P/L ratio of 0.85 makes it negative.

EV  =  (53% × +0.28%)  +  (47% × −0.33%)  =  negative

Both numbers must be read together.
Win rate alone tells you nothing about profitability.
```

**Finding 4 — ETF mean reversion is structurally different:**
```
Single stocks  →  idiosyncratic dislocations that revert
ETFs           →  macro flows that do not revert on intraday time scales

Same signal. Different instrument. Different result. Structural, not random.
```

**Finding 5 — Faster indicators generate noise, not signal:**
```
RSI(7) on 5-minute bars fires on 35-minute wiggles.

The speed of the indicator should match the speed of the phenomenon.
Intraday exhaustion takes more than 35 minutes to develop.
```

**Finding 6 — PSR is the test that matters:**
```
Sharpe ratio alone is not enough.
A Sharpe ratio from 4.5 years of data can still be noise.

PSR asks: what is the probability the true Sharpe is above zero?
Every version of VWAP+RSI produced PSR below 0.3%.
That is the correct way to close a hypothesis.
```

---

# CHAPTER 7 — THE FIVE NUMBERS AT CLOSE

Final state of the research at hypothesis close:

```
Version:     v2 (best result — regime filter)
Instrument:  AAPL
Period:      Jan 2020 – Jun 2024 (4.5 years)

──────────────────────────────────────────────────────
FIVE NUMBERS           VALUE        THRESHOLD   STATUS
──────────────────────────────────────────────────────
Gross Return           −2.33%       > 0         FAIL
Total Costs            $1,626       minimise    OK
Net Return             −2.33%       > 0         FAIL
Win Rate               53%          —           noted
P/L Ratio              0.85         > 1.0       FAIL
Expected Value         −0.007%      > 0         FAIL
PSR                    0.295%       > 95%       FAIL
──────────────────────────────────────────────────────

VERDICT: CLOSED. Zero metrics passed threshold.
         No version of this hypothesis produced statistical significance.
         Documented and archived. Research continues.
```

---

# CHAPTER 8 — WHAT CAME NEXT

Closing a hypothesis is not failure. It produces the question for the next one.

```
LESSON FROM VWAP+RSI:

Mean reversion works AGAINST the trend.
Institutional money moves WITH the trend.

If institutions are buying AAPL in the morning, price will drift above VWAP
and stay there. A mean reversion signal will short every dip and lose.

The next hypothesis inverted the direction:
  What if we FOLLOW the institutions instead of fading them?
  What if the signal is momentum, not reversion?
  What if we enter AFTER confirmation, not before it?

That became Hypothesis 2: Opening Range Breakout (ORB).
See: ORB_ALPHA_RESEARCH_MEMO.md
```

---

*Bullseye Alpha | Systematic Equity Research | bullseyealpha.com*
