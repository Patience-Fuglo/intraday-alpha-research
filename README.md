# Intraday Alpha Research — VWAP + RSI Mean Reversion

A systematic, end-to-end alpha research framework built around a single testable hypothesis: **intraday prices dislocate from VWAP and mean-revert**. The codebase documents the full research process — from signal construction to regime filtering, portfolio construction, execution modeling, and statistical validation — following the same workflow used at systematic equity funds.

---

## What This Demonstrates

- Rigorous signal research with honest, documented results — including failed hypotheses
- Full research loop: Idea → Data → Features → Signal → Backtest → Metrics → Robustness
- Production-grade methodology: look-ahead bias prevention, walk-forward validation, multiple testing correction
- Execution awareness: transaction costs, market impact (Kyle lambda), slippage modeling
- Statistical credibility: PSR, Deflated Sharpe Ratio, bootstrapped confidence intervals
- Live platform implementation: QuantConnect LEAN (Python) and TradingView (Pine Script)

---

## Repository Structure

```
├── README.md
├── research/          — yfinance signal research (entry → senior progression)
├── quantconnect/      — QuantConnect LEAN production backtests
├── tradingview/       — TradingView Pine Script strategies
└── docs/              — alpha research memos and session notes
```

### research/

| File | Level | Description |
|------|-------|-------------|
| `01_ENTRY_BEGINNER.py` | Entry | Single-ticker VWAP+RSI signal, basic backtest, cost model, multi-ticker sweep |
| `02_ORB_STRATEGY.py` | Entry–Intermediate | Opening Range Breakout — volume filter, time window, gross vs net analysis |
| `02_JUNIOR_TO_ADVANCED.py` | Junior → Senior | Complete research cheat sheet — full pipeline from signal to production |
| `03_INTERMEDIATE_ADVANCED_FIXES.py` | Intermediate | Daily VWAP reset, Wilder RSI, regime detection, earnings filter, IC decay |
| `03_ML_RIDGE_SIGNAL.py` | Intermediate–Senior | ML Ridge signal — walk-forward, IC, PSR, conviction threshold |
| `04_SENIOR_LEVEL.py` | Senior | ML signals (Ridge/Lasso/RF/XGBoost), portfolio optimization, TAQ microstructure |
| `05_SENIOR_FINAL_BOOST.py` | Production | DSR, purged walk-forward with embargo, Monte Carlo P&L, pre-trade risk checklist |

### quantconnect/

| File | Description |
|------|-------------|
| `QUANTCONNECT_VWAP_RSI.py` | VWAP+RSI production backtest — Jan 2020–Jun 2024, IB cost model |
| `QUANTCONNECT_ML_RIDGE.py` | ML Ridge production backtest — 10 features, rolling walk-forward, IC+PSR output |

### tradingview/

| File | Description |
|------|-------------|
| `06_TRADINGVIEW_FULL_STRATEGY.pine` | Full VWAP+RSI strategy with regime filter |
| `07_TRADINGVIEW_TRAINING.pine` | Signal logic with annotated entry/exit conditions |

### docs/

| File | Description |
|------|-------------|
| `RESEARCH_NOTES.md` | Session-by-session backtest log, findings, and next hypothesis |
| `ORB_ALPHA_RESEARCH_MEMO.md` | Full ORB memo — 12 chapters, all runs, levers theory, final conclusion |
| `ML_ALPHA_RESEARCH_MEMO.md` | Full ML Ridge memo — walk-forward, IC, PSR, five numbers framework |

---

## Signal Logic

```
Long  (+1): close < VWAP  AND  RSI < 25  AND  distance from VWAP > 0.3%
Short (-1): close > VWAP  AND  RSI > 75  AND  distance from VWAP > 0.3%
Exit      : price crosses back through VWAP (reversion complete) or stop loss triggered
Regime    : skip when ATR/Price > 2.5% (volatile) or price > 15% from 200-day SMA (trending)
```

---

## Backtest Results — QuantConnect, AAPL, 5-min bars, Jan 2020 – Jun 2024

| Ver | Ticker | Change | Return | PSR | Win Rate | P/L Ratio | Fees |
|-----|--------|--------|--------|-----|----------|-----------|------|
| v1 | AAPL | Baseline — no filter | -14.40% | 0.107% | — | — | $4,667 |
| v2 | AAPL | + Regime filter (ATR + SMA200) | -2.33% | 0.295% | 53% | 0.85 | $1,626 |
| v3 | AAPL | Stop loss 1.0% → 0.5% | -4.32% | 0.132% | 50% | 0.95 | $1,788 |
| v4 | AAPL | RSI period 14 → 7 | -21.30% | 0.000% | — | — | $3,214 |
| v5a | IWM | RSI(7) + stop 0.75% | -22.53% | 0.001% | — | — | $4,300 |
| v5b | IWM | RSI(14) + stop 1.0% (best settings) | -12.22% | 0.005% | — | — | $2,075 |

**Research conclusion:** VWAP + RSI mean reversion shows no statistically significant edge on AAPL or IWM across 5 versions and 2 instruments (PSR < 1% throughout). The regime filter was the only structural improvement — cutting fees 65% and reducing the COVID crash impact. IWM underperformed AAPL because ETF intraday moves are driven by macro flows, not the idiosyncratic stock noise that VWAP mean reversion requires. Hypothesis closed. Full methodology and findings documented in `RESEARCH_NOTES.md`.

---

## Key Concepts Implemented

**Signal Research**
- VWAP daily reset — volume-weighted intraday fair value
- Wilder RSI (exponential smoothing) vs. simple rolling RSI
- Look-ahead bias prevention via `position.shift(1)`
- Composite alpha score: VWAP distance + RSI + return z-score reversal

**Risk & Execution**
- Transaction costs: commission + bid-ask spread + market impact
- Kyle lambda — price impact per unit of signed order flow
- TWAP and VWAP execution algorithms
- Almgren-Chriss impact model reference
- Pre-trade risk checklist

**Portfolio Construction**
- Cross-sectional long/short ranking
- Volatility scaling and sector neutralization
- Beta-neutral portfolio construction
- Risk parity allocation

**Statistical Validation**
- PSR (Probabilistic Sharpe Ratio)
- Deflated Sharpe Ratio — Lopez de Prado (2014) multiple testing correction
- Bootstrapped Sharpe confidence intervals
- Purged walk-forward cross-validation with embargo
- Monte Carlo P&L path simulation
- Information coefficient (IC) decay analysis

---

## Stack

- **Python:** pandas, numpy, matplotlib, scikit-learn, scipy, xgboost, yfinance
- **Platform:** QuantConnect LEAN (live backtest environment)
- **Charting:** TradingView Pine Script v5
- **Methodology:** Interactive Brokers cost model, minute-resolution equity data

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib scikit-learn scipy xgboost
```

Each file is self-contained and executable independently. Progress from `01` through `05` to build the full framework layer by layer.

---

*[Bullseye Alpha](https://bullseyealpha.com) — Systematic equity research*
