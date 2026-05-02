# Intraday Alpha Research — VWAP + RSI Mean Reversion

End-to-end intraday alpha research framework built on a single hypothesis: **intraday prices dislocate from VWAP and mean-revert**. The codebase progresses from a single-ticker backtest to a production-grade cross-sectional portfolio with regime filtering, execution modeling, and statistical validation.

---

## Repository Structure

| File | Level | Description |
|------|-------|-------------|
| `01_ENTRY_BEGINNER.py` | Entry | Single ticker VWAP+RSI signal, basic backtest, cost model, multi-ticker sweep |
| `02_JUNIOR_TO_ADVANCED.py` | Junior → Senior | Master cheat sheet covering the full research loop from entry to production |
| `03_INTERMEDIATE_ADVANCED_FIXES.py` | Intermediate | Correct daily VWAP reset, Wilder RSI, regime detection, earnings filter, IC decay |
| `04_SENIOR_LEVEL.py` | Senior | ML signals (Ridge, Lasso, RF, XGBoost), walk-forward validation, portfolio optimization, TAQ microstructure, execution simulation |
| `05_SENIOR_FINAL_BOOST.py` | Senior/Production | Deflated Sharpe Ratio, bootstrapped CI, purged walk-forward with embargo, Monte Carlo P&L, regime-conditional sizing |
| `06_TRADINGVIEW_FULL_STRATEGY.pine` | TradingView | Full VWAP+RSI strategy with regime filter for live chart visualization |
| `07_TRADINGVIEW_TRAINING.pine` | TradingView | Training version with annotated signal logic |
| `QUANTCONNECT_VWAP_RSI.py` | QuantConnect | Production backtest in LEAN framework — Jan 2020 to Jun 2024, Interactive Brokers cost model |
| `RESEARCH_NOTES.md` | Research Log | Session-by-session backtest results, version history, findings, and next hypothesis |

---

## Research Loop

Every file follows the same research loop:

```
Idea → Data → Features → Signal → Backtest → Metrics → Robustness
```

---

## Signal Logic

**Long entry:** `close < VWAP` AND `RSI < 25` AND `distance from VWAP > 0.3%`  
**Short entry:** `close > VWAP` AND `RSI > 75` AND `distance from VWAP > 0.3%`  
**Exit:** price crosses back through VWAP (reversion complete) or stop loss  
**Regime filter:** skip when `ATR/Price > 2.5%` or `price > 15% from 200-day SMA`

---

## Backtest Results (QuantConnect, AAPL, Jan 2020 – Jun 2024)

| Version | Change | Return | PSR | Win Rate | P/L Ratio | Fees |
|---------|--------|--------|-----|----------|-----------|------|
| v1 | Baseline | -14.40% | 0.107% | — | — | $4,667 |
| v2 | Regime filter | -2.33% | 0.295% | 53% | 0.85 | $1,626 |
| v3 | Stop loss 1.0% → 0.5% | -4.32% | 0.132% | 50% | 0.95 | $1,788 |
| v4 | RSI period 14 → 7 | -21.30% | 0.000% | — | — | $3,214 |

**Key finding:** The regime filter was the only change that improved performance. VWAP+RSI on AAPL 5-minute bars shows no statistically significant edge (PSR < 1% across all versions). Next step: test on IWM where mean reversion is historically stronger in mid-cap names.

---

## Key Concepts Covered

- VWAP daily reset and volume-weighted fair value
- Wilder's RSI (EWM) vs. simple rolling RSI
- Look-ahead bias prevention via `position.shift(1)`
- Transaction costs: commission + slippage + spread + market impact
- Regime filtering: ATR volatility + 200-day SMA trend
- PSR (Probabilistic Sharpe Ratio) and Deflated Sharpe Ratio
- Cross-sectional long/short portfolio construction
- Sector neutralization and volatility scaling
- Kyle lambda (price impact per unit of signed order flow)
- Purged walk-forward validation with embargo
- Monte Carlo P&L simulation and bootstrapped Sharpe CI

---

## Stack

- **Python:** pandas, numpy, matplotlib, scikit-learn, scipy, yfinance
- **QuantConnect:** LEAN framework, minute-resolution data
- **TradingView:** Pine Script v5

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib scikit-learn scipy xgboost
```

Run each file in order. Start with `01_ENTRY_BEGINNER.py`, then progress through the levels.

---

*Bullseye Alpha — Systematic equity research*
