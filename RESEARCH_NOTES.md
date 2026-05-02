# VWAP RSI Mean Reversion — Research Log

---

## Session 1 — May 1, 2026

**Strategy:** VWAP + RSI mean reversion on AAPL, 5-minute bars
**Platform:** QuantConnect LEAN | Interactive Brokers cost model | $100,000 starting capital
**Backtest window:** Jan 1, 2020 - Jun 1, 2024 (4.5 years)

---

### Version History

| Ver | Key Change | Return | Net P&L | PSR | Win Rate | Avg Win | Avg Loss | P/L Ratio | Fees |
|-----|-----------|--------|---------|-----|----------|---------|---------|-----------|------|
| v1 | Baseline — no filter | -14.40% | -$14,400 | 0.107% | — | — | — | — | $4,667 |
| v1b | Short window test | +3.28% | +$3,280 | 16.0% | — | — | — | — | — |
| v2 | Regime filter (ATR + SMA200) | -2.33% | -$2,226 | 0.295% | 53% | +0.28% | -0.33% | 0.85 | $1,626 |
| v3 | Stop loss 1.0% to 0.5% | -4.32% | -$4,227 | 0.132% | 50% | +0.29% | -0.31% | 0.95 | $1,788 |
| v4 | RSI period 14 to 7, stop 0.75% | -21.30% | -$21,220 | 0.000% | — | — | — | — | $3,214 |

---

## What I tested today

VWAP RSI mean reversion on AAPL, 5-minute bars.

I entered long when price was more than 0.3% below VWAP and RSI dropped below 25 — meaning the stock was oversold and displaced below its intraday fair value. I entered short when the reverse was true. I used a 1% stop loss and exited when price crossed back through VWAP.

I ran four versions, changing one variable at a time: the baseline (v1), a regime filter to block volatile and trending periods (v2), a tighter stop to reduce average loss (v3), and a faster RSI to improve entry timing (v4).

---

## Key finding

The strategy appeared profitable when tested on a short, calm window (v1b: +3.28%) but failed across the full 4.5-year period in every version.

The regime filter (v2) was the single biggest improvement — it cut fees by 65% and reduced the loss from -14.40% to -2.33% by blocking the COVID crash and the 2021-2022 tech rally. During those periods, AAPL trended far from its VWAP for days at a time. The mean reversion assumption broke entirely. Price was not displaced — it had genuinely changed direction.

Even after filtering those regimes out, the core signal still lost money. The reason is a negative expected value per trade:

Expected value (v2) = (53% x +0.28%) + (47% x -0.33%) = +0.148% - 0.155% = -0.007% per trade

Every trade, on average, lost a small amount. Across hundreds of trades that compounds into steady losses. For the strategy to be profitable, average win must exceed average loss (P/L ratio above 1.0) or win rate must be high enough to overcome the imbalance. Neither condition was met.

Attempts to fix the P/L ratio made things worse. Tightening the stop (v3) caused whipsaw — the stop triggered on normal noise before the reversion had time to complete, increasing orders from 779 to 853 and lowering win rate from 53% to 50%. Using a faster RSI (v4) generated so many low-quality signals that fees alone destroyed the account, falling from -4.32% to -21.30%.

---

## What I learned

- Mean reversion requires calm, ranging markets. The strategy breaks in any sustained trend or volatility spike because the core assumption — that price will return to a mean — stops being true.

- PSR below 50% means the result is noise, not skill. Every version in this session produced PSR below 1%. That means there is less than a 1% probability the strategy has genuine positive expected value. A Sharpe ratio without PSR tells you very little.

- Regime filtering is not optional — it is necessary for survival. Without it, the strategy trades through every crash and trend. The filter does not create edge; it prevents edge from being destroyed by trading in the wrong environment.

- Win rate alone does not determine profitability. 53% win rate with a 0.85 P/L ratio is a losing strategy. Expected value is what matters: (win rate x avg win) + (loss rate x avg loss). If that number is negative, the strategy loses money regardless of how often it wins.

- Stops manage risk, they do not create edge. Tightening a stop cannot fix a signal that is generating losses. It can only change the size of individual losses. If the signal itself is wrong, the stop just makes the wrong trades smaller.

- A faster indicator generates more signals, not better signals. RSI(7) fired on every noise spike. The volume of trades increased sharply and quality collapsed. Signal speed and signal quality are independent properties.

- A publicly known signal is likely arbitraged. VWAP + RSI is in every retail trading tutorial and quant textbook. If thousands of traders run the same signal on AAPL, the edge disappears because everyone acts at the same moment. Real edge requires either proprietary data, a faster execution advantage, or a signal that most participants are not running.

- Running more than 4-5 variations on the same dataset becomes data mining. At that point, you are fitting to the specific noise pattern of the data, not discovering a real relationship. The right response after 4 failed variations is to close the hypothesis, write the conclusion, and move to a different idea.

---

## Questions to answer next session

- Does VWAP + RSI show edge on a less efficient instrument, such as IWM (Russell 2000) where mean reversion is historically stronger than in large-cap single stocks?
- What is the minimum PSR threshold that indicates a hypothesis is worth continuing to develop?
- What does a signal with genuine institutional logic look like versus a retail signal? What makes the difference?

---
