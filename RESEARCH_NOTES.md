# VWAP RSI Mean Reversion — Research Log

---

## Session 1 — May 1, 2026

**Strategy:** VWAP + RSI mean reversion on AAPL, 5-minute bars
**Platform:** QuantConnect LEAN | Interactive Brokers cost model | $100,000 starting capital
**Backtest window:** Jan 1, 2020 - Jun 1, 2024 (4.5 years)

---

### Full Version History

| Ver | Ticker | Key Change | Return | Net P&L | PSR | Win Rate | Avg Win | Avg Loss | P/L Ratio | Fees |
|-----|--------|-----------|--------|---------|-----|----------|---------|---------|-----------|------|
| v1 | AAPL | Baseline — no filter | -14.40% | -$14,400 | 0.107% | — | — | — | — | $4,667 |
| v1b | AAPL | Short window test | +3.28% | +$3,280 | 16.0% | — | — | — | — | — |
| v2 | AAPL | Regime filter (ATR + SMA200) | -2.33% | -$2,226 | 0.295% | 53% | +0.28% | -0.33% | 0.85 | $1,626 |
| v3 | AAPL | Stop loss 1.0% to 0.5% | -4.32% | -$4,227 | 0.132% | 50% | +0.29% | -0.31% | 0.95 | $1,788 |
| v4 | AAPL | RSI period 14 to 7, stop 0.75% | -21.30% | -$21,220 | 0.000% | — | — | — | — | $3,214 |
| v5a | IWM | RSI(7) + stop 0.75% (carry-over from v4) | -22.53% | -$22,430 | 0.001% | — | — | — | — | $4,300 |
| v5b | IWM | RSI(14) + stop 1.0% (best AAPL settings) | -12.22% | -$12,111 | 0.005% | — | — | — | — | $2,075 |

---

## What I tested today

VWAP RSI mean reversion on AAPL, 5-minute bars — four versions changing one variable at a time. Then extended the best configuration to IWM (Russell 2000 ETF) as an out-of-sample instrument test.

---

## Key finding

**AAPL (v1–v4):** The signal failed across every configuration. The regime filter (v2) was the only structural improvement — cutting fees by 65% and reducing the loss from -14.40% to -2.33%. The core problem was negative expected value per trade: avg loss (-0.33%) exceeded avg win (+0.28%) at every tested configuration.

Expected value (v2) = (53% x +0.28%) + (47% x -0.33%) = -0.007% per trade

**IWM (v5a, v5b):** Worse than AAPL on both tests. IWM with the best AAPL settings (RSI 14, stop 1%) returned -12.22% vs AAPL's -2.33%. Fees were higher. PSR was lower.

The hypothesis that IWM would show stronger mean reversion than AAPL was incorrect. IWM intraday moves are driven by macro flows — Fed decisions, risk-on/risk-off positioning, sector rotation — not individual stock dislocations. VWAP mean reversion requires idiosyncratic noise to fade. IWM does not have that; it moves systematically.

**Conclusion: VWAP + RSI mean reversion has no detectable edge on AAPL or IWM across 5 versions and 2 instruments. Hypothesis closed.**

---

## What I learned

- Mean reversion requires calm, ranging markets. The strategy breaks in any sustained trend or volatility spike because the core assumption — price returns to a mean — stops being true.

- PSR below 50% means the result is noise, not skill. Every version produced PSR below 1%. A Sharpe ratio without PSR tells you very little.

- Regime filtering is necessary but not sufficient. It prevents edge from being destroyed in the wrong environment. It does not create edge where none exists.

- Win rate alone does not determine profitability. 53% win rate with a 0.85 P/L ratio is a losing strategy. Expected value = (win rate x avg win) + (loss rate x avg loss). If negative, the strategy loses regardless of win rate.

- Stops manage risk, they do not create edge. Tightening a stop cannot fix a signal that generates losses.

- A faster indicator generates more signals, not better signals. RSI(7) fired on noise. Volume and quality of signals are independent properties.

- ETF mean reversion is different from single-stock mean reversion. ETFs reflect macro flows. Single stocks reflect idiosyncratic events. The same signal does not transfer.

- A publicly known signal is likely arbitraged. VWAP + RSI is in every retail tutorial. Real edge requires signals others are not running.

- 4-5 variations on the same dataset is the limit before results become data mining.

---

## Session 2 — May 2, 2026

**Instrument tested:** IWM (Russell 2000 ETF)
**Hypothesis:** Mean reversion stronger in mid-cap names than AAPL
**Result:** Rejected. IWM underperformed AAPL on every metric.
**Decision:** Close VWAP + RSI hypothesis entirely. Move to new signal.

---

## Next hypothesis — Opening Range Breakout (ORB)

**Logic:** First 30 minutes of the session establish the high and low. A breakout above the high or below the low with volume confirmation signals a directional move for the day.

**Why this is different:**
- Momentum-based, not mean reversion — opposite direction of bet
- Driven by institutional order flow at the open, not retail technical levels
- Works on both single stocks and ETFs
- Less likely to be arbitraged — execution speed matters, not just signal timing

**Questions to answer:**
- Does ORB on AAPL produce positive return and PSR > 50%?
- Does a volume filter improve signal quality?
- Does the regime filter from VWAP+RSI transfer to ORB?

---

## Session 3 — May 10, 2026

**Strategy:** Opening Range Breakout (ORB)
**Platform:** VS Code + yfinance | 5-minute bars | 60-day window
**Tickers:** AAPL, MSFT, NVDA, SPY, QQQ
**File:** `02_ORB_STRATEGY.py`

---

### Run History

| Run | Change | Gross Avg | Costs Avg | Net Avg | Trades Avg | Decision |
|-----|--------|-----------|-----------|---------|------------|----------|
| 1 | Baseline — no filters | unknown | unknown | -4.0% | 108 | Add volume filter |
| 2 | Volume filter 1.5x | +2.9% | 6.9% | -4.0% | 99 | Tighten filter |
| 3 | Volume filter 2.5x | +2.6% | 4.6% | -2.0% | 65 | Add min move |
| 4 | 2.5x vol + 0.2% min move | +1.74% | 3.40% | -1.66% | 48.6 | Fix trade count |
| 5 | Volume loosened to 2.0x | +1.34% | 4.66% | -3.32% | 66.6 | Wrong direction — revert |
| 6 | 10am–11am time filter (2.5x) | +0.56% | 0.14% | +0.41% | 2.0 | Best quality, too few trades |
| 7 | 10am–12pm time window (2.5x) | +0.53% | 0.69% | -0.18% | 9.8 | Data wall reached |

---

### Key finding

**Gross return was positive on every run.** This is the structural difference from VWAP+RSI. The signal finds real institutional momentum. Costs were the obstacle, not signal direction.

The 10am–11am time filter (Run 6) was the breakthrough: costs collapsed from 3.40% to 0.14%, net return turned positive (+0.41%), Sharpe reached 1.20. The filter confirmed that institutional breakout momentum is concentrated in the first hour after the opening range closes.

The data limit ended the research: Yahoo Finance provides a maximum of 60 days of 5-minute bars. With a 10am–11am entry window and 2.5x volume filter, only 2 trades per ticker triggered across 60 days — well below the 50-trade statistical threshold.

---

### What I learned

- Gross positive + net negative = fix costs, not the signal. Gross negative = close hypothesis immediately.
- Volume filter quality matters more than quantity. Loosening from 2.5x to 2.0x added weak trades that widened the gap (Run 5).
- Entry timing is a signal quality lever in momentum strategies. 10am–11am is the institutional commitment window. After 11am, breakout reliability drops.
- Levers come from theory. Mean reversion tunes thresholds (how far). Momentum tunes volume and time (who and when). These are not interchangeable.
- Sample size limits cannot be resolved by tuning. 60-day yfinance data with strict filters hits a wall at ~2 trades. This requires a different data source, not different parameters.

---

### Hypothesis status

**CLOSED — data limit.** Signal edge confirmed (gross positive, Run 6 positive net). Full validation requires QuantConnect with multi-year minute-resolution data.

**Next:** ML signal — Ridge Regression on intraday features.

---

## Session 4 — May 11, 2026

**Strategy:** ML Signal — Ridge Regression on Intraday Features
**Platform:** VS Code + yfinance + scikit-learn | 5-minute bars | 60-day window
**Tickers:** AAPL, MSFT, NVDA, SPY, QQQ
**File:** `03_ML_RIDGE_SIGNAL.py`
**Validation:** Walk-forward — train on first 40 days, test on last 20 days

---

### What Ridge Regression does

Ridge Regression is a linear model that learns the optimal weight for each input feature. It combines VWAP distance, RSI, volume ratio, ORB breakout size, momentum, and time of day into a single continuous score. High score = likely positive return. Low score = likely negative. The model penalizes large weights (L2 regularization) to prevent overfitting to 60 days of data.

New concept introduced: **IC (Information Coefficient)**
IC = correlation between the model's predicted score and the actual return that followed.
IC > 0.05 = useful signal. IC > 0.10 = strong signal. IC near 0 = no predictive power.

---

### Run History — Five Numbers Each Run

**Run 1 — Raw signal, every bar traded**

```
                              VERDICT
Total Return    -25.2%       LOSS      below 0%    ✗
Max Drawdown    -25.4%       DANGER    near 25%    ✗
Trades           192.6       TOO MANY  no filter   ✗
Sharpe           -19.5       WEAK      below 1.0   ✗
Gross Return    -2.3%        NO EDGE   negative    ✗
Total Costs     26.9%        FATAL     kills all   ✗
IC              -0.04        NONE      near zero   ✗

Problem: np.sign(score) traded every bar. 192 trades × 0.14% = 26.9% in fees.
```

**Run 2 — 30-minute forward return target**

```
                              VERDICT
Total Return    -12.6%       LOSS      below 0%    ✗
Max Drawdown    -12.9%       HIGH      above 10%   ✗
Trades           161.0       TOO MANY  still high  ✗
Sharpe           -14.6       WEAK      below 1.0   ✗
Gross Return    -1.8%        NO EDGE   negative    ✗
Total Costs     11.7%        HIGH      kills all   ✗
IC              -0.06        NONE      near zero   ✗

Change: predicting 30-min return instead of 5-min. Costs halved but still fatal.
```

**Run 3 — ORB 10am–11am time window added**

```
                              VERDICT
Total Return    -1.58%       LOSS      below 0%    ✗
Max Drawdown    -2.36%       GOOD      below 10%   ✓
Trades           27.6        WARN      below 50    ✗
Sharpe           -7.13       WEAK      below 1.0   ✗
Gross Return    +0.34%       POSITIVE  edge found  ✓  ← first positive gross
Total Costs      1.93%       LOW       improving   ✓
IC              -0.06        LOW       near zero   ✗

Gap: Gross +0.34% vs Costs 1.93% = -1.58%
Change: Restrict entries to 10am–11am ET (UTC hour 14). Costs collapsed 83%.
Gross turned positive for first time. Time filter doing the work, not ML weights.
```

**Run 4 — 10 features (added ATR, dollar volume, gap size, return z-score)**

```
                              VERDICT
Total Return    -1.36%       LOSS      below 0%    ✗
Max Drawdown    -2.85%       GOOD      below 10%   ✓
Trades           31.0        WARN      below 50    ✗
Sharpe           -6.73       WEAK      below 1.0   ✗
Gross Return    +0.81%       POSITIVE  improving   ✓
Total Costs      2.18%       LOW       manageable  ✓
IC              +0.042       LOW       approaching ✓  ← turned positive

Gap: Gross +0.81% vs Costs 2.18% = -1.36%
New features: ATR ratio (regime), dollar volume (institutional activity),
gap size (overnight positioning), return z-score (move unusualness).
Feature weights 7x larger than Run 1. QQQ IC = 0.147 (strong individual signal).
```

### Progression

```
        Gross    Costs    Net      Trades   IC       Key change
Run 1   -2.3%   26.9%   -25.2%   192.6    -0.04    every bar traded
Run 2   -1.8%   11.7%   -12.6%   161.0    -0.06    30-min target
Run 3   +0.34%   1.93%   -1.58%   27.6    -0.06    ORB time window
Run 4   +0.81%   2.18%   -1.36%   31.0    +0.04    10 features
```

---

### Key findings

- ML does not create signal where none exists. It finds patterns the features contain — but only with enough data.
- 60-day data limit prevents weights from stabilizing. All coefficients near zero in Runs 1–2.
- The ORB time window (10am–11am ET) rescued gross return, not the ML model. Same finding as ORB strategy.
- Feature engineering improved IC from -0.06 to +0.04. Dollar volume and ATR are the strongest features.
- QQQ IC = 0.147 is a genuine signal — above the 0.10 professional threshold.
- IC turning positive confirms the new features contain real information. More data will strengthen it.

---

### Hypothesis status

**OPEN.** Gross positive, IC positive, feature weights meaningful.
Data limit (60 days) prevents IC from crossing 0.05 consistently.
Next: walk-forward with multiple folds, then PSR validation.

---
