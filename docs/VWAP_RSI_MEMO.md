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

```
HYPOTHESIS
──────────
Intraday prices dislocate from VWAP (intraday fair value) and mean-revert.
RSI extremes identify the moment of maximum dislocation.

Entry: price below VWAP AND RSI below 25  →  buy, expect recovery
       price above VWAP AND RSI above 75  →  sell, expect decline

Exit:  price crosses back through VWAP  →  reversion complete

Instrument: AAPL (first), IWM (out-of-sample)
Frequency:  5-minute bars
Platform:   QuantConnect LEAN, Interactive Brokers cost model
Window:     Jan 2020 – Jun 2024 (4.5 years, $100,000 capital)
```

---

# CHAPTER 2 — WHAT EACH TOOL DOES

## What is VWAP?

```
VWAP = Volume-Weighted Average Price
     = the average price every market participant paid today,
       weighted by the number of shares they traded

Formula:
  VWAP = Σ(typical_price × volume) / Σ(volume)
  typical_price = (high + low + close) / 3

Resets at 9:30am every trading day. Reflects only that day's activity.
By 4pm it represents the full day's consensus.

VWAP is intraday fair value.
If price is BELOW VWAP → stock is cheap relative to today's average.
If price is ABOVE VWAP → stock is expensive relative to today's average.

This is why market makers and institutional traders reference VWAP as a
benchmark: beating VWAP on a buy means you paid less than average.
```

## What is RSI?

```
RSI = Relative Strength Index (0 to 100)
    = how extreme recent price moves have been

Formula:
  avg_gain = rolling mean of positive returns over N bars
  avg_loss = rolling mean of negative returns over N bars (absolute)
  RS        = avg_gain / avg_loss
  RSI       = 100 - (100 / (1 + RS))

RSI < 30  →  oversold   →  recent losses dominate, move may be exhausted
RSI > 70  →  overbought →  recent gains dominate, move may be exhausted

RSI is the CONFIRMATION.
VWAP tells you WHERE price is relative to fair value.
RSI tells you HOW EXTREME the current move has been.
Both must agree before entering.
```

## Why combine them?

```
VWAP alone: price is below fair value. So what? It could keep going lower.
RSI alone:  readings are extreme. On which asset? In which direction?

Together:
  Price is below VWAP (dislocated from fair value)
  AND RSI is below 25 (the selling pressure is exhausted)
  = This is the best setup for a snap-back.

The combination reduces false signals.
One condition is noise. Two conditions pointing the same direction are signal.
```

---

# CHAPTER 3 — THE LEVERS

Before running a single version, identify what can be tuned and WHY.

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

NOT a logical lever for mean reversion:
  Volume filter  →  volume tells you about momentum, not reversion
  Time window    →  mean reversion can happen at any time of day
  These are momentum levers. Borrowed from ORB. Do not mix frameworks.
```

---

# CHAPTER 4 — RUN HISTORY (FIVE NUMBERS EACH VERSION)

Every run changes ONE variable. Read the five numbers in sequence.

## The Five Numbers Framework

```
1. Gross Return   →  does the signal have edge before fees?
2. Total Costs    →  what is the fee drag?
3. Net Return     →  gross minus costs — the real result
4. Win Rate       →  what fraction of trades were winners?
5. PSR            →  is the Sharpe ratio statistically real?

Reading order: always gross first. If gross is negative, stop.
               Costs, stops, and filters cannot fix a negative-gross signal.
```

---

## v1 — Baseline (no filter)

```
Change:      Pure VWAP+RSI signal. No regime filter. No stop loss. All signals traded.

Gross Return    −14.40%     ✗   Signal has no edge — gross is negative
Total Costs     $4,667      ✗   High fees (many trades, small account)
Net Return      −14.40%     ✗   Worse than doing nothing
Win Rate        —           ✗   Not recorded
PSR             0.107%      ✗   Not real — pure statistical noise

Reading:
The signal lost money before costs. Regime did not matter —
VWAP mean reversion fired in trending markets and faded in the
wrong direction. Adding filters to a gross-negative signal does
not help: there is no edge to uncover.

Decision: add regime filter. Hypothesis not closed — v1 could be
the wrong environment, not a wrong signal.
```

---

## v2 — Regime filter added (ATR volatility + SMA200 trend)

```
Change:      Added two conditions:
             ATR/Price < 2.5%  →  only trade when volatility is calm
             Price within 3% of SMA200  →  only trade when market is not in a strong trend

Gross Return    −2.33%      ✗   Still negative, but materially better (12pp improvement)
Total Costs     $1,626      →   65% fee reduction — far fewer trades
Net Return      −2.33%      ✗   Negative but in a different regime
Win Rate        53%         →   More than half of trades were winners
P/L Ratio       0.85        ✗   Average loss > average win — negative EV
PSR             0.295%      ✗   Still noise — not statistically real

Expected Value per trade:
  (53% × +0.28%) + (47% × −0.33%) = −0.007% per trade
  Strategy loses a fraction of a percent on every trade on average.

Reading:
The regime filter was the most important structural improvement.
It removed 65% of fees by preventing trades in hostile environments.
It lifted net return from −14.4% to −2.33%. But the core problem
remained: average losing trade (−0.33%) was larger than average
winning trade (+0.28%). A 53% win rate with a 0.85 P/L ratio is
a losing strategy. No stop loss or filter can fix negative EV.

Decision: tighten stop loss to reduce average loss size.
```

---

## v3 — Stop loss tightened (1.0% → 0.5%)

```
Change:      Stop loss reduced from 1.0% to 0.5%. Exit if trade loses 0.5%.

Gross Return    −4.32%      ✗   Worse than v2 (stop cutting winners short)
Total Costs     $1,788      →   Similar to v2
Net Return      −4.32%      ✗   Negative
Win Rate        50%         →   Win rate dropped (stop catches some reversals early)
P/L Ratio       0.95        →   Slight improvement over v2 (0.85 → 0.95)
PSR             0.132%      ✗   Not real

Reading:
Tightening the stop reduced the size of losing trades (P/L improved
to 0.95). But it also cut winning trades early — price would hit the
stop before the reversion completed. Net return got worse, not better.

This is the classic stop-loss trap: tightening a stop is not the same
as having a better signal. If the signal does not have edge, the stop
only changes WHEN you lose, not whether you lose.

Decision: try faster RSI to generate more extreme readings.
```

---

## v4 — RSI period shortened (14 → 7)

```
Change:      RSI window from 14 bars to 7 bars. Faster indicator, more extreme readings.

Gross Return    −21.30%     ✗   Much worse — signal became anti-predictive
Total Costs     $3,214      ✗   More trades = more fees
Net Return      −21.30%     ✗   Worst result so far
Win Rate        —           ✗   Not recorded
PSR             0.000%      ✗   Zero — purely random result

Reading:
Faster RSI fired on noise. A 7-bar RSI reaches extreme readings after
just 7 five-minute bars — 35 minutes of movement. That is not exhaustion.
That is a normal intraday wiggle. The result was more trades, more costs,
and predictions pointing in the wrong direction.

Lesson: more signals ≠ better signals. Frequency and quality are
independent properties of a signal.

Decision: extend to IWM — test whether the hypothesis works on ETFs.
```

---

## v5a/v5b — IWM (out-of-sample instrument test)

```
v5a — RSI(7) + stop 0.75% on IWM:
  Net Return    −22.53%     ✗   Worse than AAPL equivalent
  PSR           0.001%      ✗   Zero

v5b — RSI(14) + stop 1.0% on IWM (best AAPL settings):
  Net Return    −12.22%     ✗   Worse than AAPL v2 (−2.33%)
  PSR           0.005%      ✗   Near zero

Reading:
IWM underperformed AAPL on every metric. This confirmed a structural
difference, not just noise:

AAPL intraday moves are driven by IDIOSYNCRATIC events:
  earnings, product news, analyst upgrades, short squeezes.
  These create dislocations that are localised to one stock.
  Price moves away from VWAP and can return.

IWM intraday moves are driven by MACRO FLOWS:
  Federal Reserve decisions, risk-on/risk-off rotation,
  sector fund flows, economic data releases.
  These are systematic — when macro drives IWM down,
  it stays down until the macro event resolves.
  There is no local reversion pulling it back.

VWAP mean reversion requires idiosyncratic noise to fade.
ETFs do not have that. The same signal does not transfer.

Decision: close hypothesis. No version produced statistical significance.
```

---

# CHAPTER 5 — WHY THE HYPOTHESIS FAILED

```
ROOT CAUSE ANALYSIS:

Failure 1 — Wrong market environment
  Mean reversion works in range-bound, choppy markets.
  Jan 2020–Jun 2024 included COVID crash (2020), tech rally (2021),
  rate-hike bear market (2022), AI bull run (2023–24).
  More than half the period was strongly trending.
  The regime filter helped, but could not remove all trending periods.

Failure 2 — Negative expected value per trade
  P/L ratio < 1.0 across every version.
  Average losing trade exceeded average winning trade.
  No amount of filtering or parameter tuning fixes negative EV.
  This is a signal design problem, not a parameter problem.

Failure 3 — VWAP+RSI is publicly known
  This signal appears in every retail trading tutorial.
  If thousands of traders run the same signal at the same time,
  the edge is arbitraged away. The dislocation gets filled before
  any single trader can capture the reversion.
  Real edge requires signals others are not running.

Failure 4 — RSI measures momentum, not exhaustion
  RSI below 30 means recent selling has dominated.
  It does not mean selling is finished.
  In a downtrend, RSI can stay below 30 for weeks.
  Using RSI as an exhaustion signal in a trending market is a
  category error — the tool is being applied in the wrong context.
```

---

# CHAPTER 6 — WHAT THE RESEARCH PRODUCED

Even a closed hypothesis produces value. Everything learned here
was a direct input to the next signal.

```
FINDING 1: Regime filter is essential
  The ATR + SMA200 filter was the only structural improvement.
  It reduced fees 65% and cut gross loss from −14.4% to −2.33%.
  This filter was carried forward into every subsequent signal.
  It reappears in ORB (v2) and ML Ridge (regime awareness).

FINDING 2: Gross must be positive before anything else matters
  v1 had negative gross. Stop losses and regime filters could not fix it.
  Rule established for all future research:
  if gross < 0, close hypothesis immediately.
  Never optimise a losing signal. Redesign it.

FINDING 3: Win rate without P/L ratio is meaningless
  53% win rate sounds good. P/L ratio of 0.85 means it is not.
  Expected value = (53% × 0.28%) + (47% × −0.33%) = negative.
  Both numbers must be read together. Neither alone tells you anything.

FINDING 4: ETF mean reversion is structurally different
  Single stocks have idiosyncratic dislocations that revert.
  ETFs reflect macro flows that do not revert on intraday time scales.
  Same signal, different instrument = different strategy, different result.

FINDING 5: Faster indicators generate noise, not signal
  RSI(7) on 5-minute bars fires on 35-minute wiggles.
  The speed of an indicator should match the speed of the phenomenon
  you are trying to detect. Intraday exhaustion takes more than 35 minutes.

FINDING 6: PSR is the test that matters
  Sharpe ratio alone is not enough.
  A Sharpe ratio from 4.5 years of data can be noise.
  PSR asks: what is the probability the true Sharpe is above zero?
  Every version of VWAP+RSI produced PSR below 0.3%.
  This is the correct way to close a hypothesis.
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

Closing a hypothesis is not failure. It is the research process working correctly.

```
LESSON FROM VWAP+RSI:

Mean reversion works against the trend.
Institutional money moves WITH the trend.

If institutions are buying AAPL in the morning, price will drift above VWAP
and stay there. A mean reversion signal will short every dip and lose.

The next hypothesis inverted the direction:
  What if we follow the institutions instead of fading them?
  What if the signal is momentum, not reversion?
  What if we enter AFTER confirmation, not before it?

That became Hypothesis 2: Opening Range Breakout (ORB).
See: ORB_ALPHA_RESEARCH_MEMO.md
```

---

*Bullseye Alpha | Systematic Equity Research | bullseyealpha.com*
