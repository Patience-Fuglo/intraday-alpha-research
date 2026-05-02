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
