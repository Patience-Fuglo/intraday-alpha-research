# Intraday Alpha Research — Signal Construction, Walk-Forward Validation & Statistical Testing

A systematic, end-to-end alpha research framework documenting the **complete research lifecycle** across two intraday hypotheses — from idea to feature construction, ML signal development, walk-forward validation, PSR/DSR statistical testing, and production backtesting in QuantConnect LEAN.

Built to the standard of a systematic equity fund research process. Every hypothesis is documented — including the failures.

---

## What This Demonstrates

- **Full research lifecycle** — Idea → Data → Features → Signal → Backtest → Metrics → Robustness
- **Two complete alpha hypotheses** — VWAP+RSI mean reversion (closed) and ML Ridge momentum (open)
- **ML signal development** — Ridge Regression on 10 intraday features with rolling walk-forward retraining
- **Production-grade validation** — IC (Information Coefficient), PSR, DSR (Deflated Sharpe Ratio)
- **Multiple testing correction** — DSR penalises Sharpe for the number of strategies searched
- **QuantConnect LEAN implementation** — 4.5-year backtests with Interactive Brokers cost model
- **Honest documentation** — failed hypotheses recorded with full analysis, not discarded

---

## Repository Structure

```
├── README.md
├── signals/     — signal construction and research (entry → production-grade)
├── backtests/   — production platform backtests (QuantConnect LEAN)
├── charts/      — live visualization and monitoring (TradingView Pine Script)
└── docs/        — alpha research memos, session notes, findings
```

### signals/

| File | Level | Description |
|------|-------|-------------|
| `01_ENTRY_BEGINNER.py` | Entry | Single-ticker VWAP+RSI signal, backtest, cost model, multi-ticker sweep |
| `02_ORB_STRATEGY.py` | Entry–Intermediate | Opening Range Breakout — volume filter, time window, gross vs net analysis |
| `02_JUNIOR_TO_ADVANCED.py` | Junior → Senior | Complete research cheat sheet — full pipeline from signal to production |
| `03_INTERMEDIATE_ADVANCED_FIXES.py` | Intermediate | Daily VWAP reset, Wilder RSI, regime detection, earnings filter, IC decay |
| `03_ML_RIDGE_SIGNAL.py` | Intermediate–Senior | ML Ridge — 10 features, walk-forward (3 folds), IC, PSR, DSR |
| `04_SENIOR_LEVEL.py` | Senior | ML signals (Ridge/Lasso/RF/XGBoost), portfolio optimization, TAQ microstructure |
| `05_SENIOR_FINAL_BOOST.py` | Production | DSR, purged walk-forward with embargo, Monte Carlo P&L, pre-trade risk checklist |

### backtests/

| File | Description |
|------|-------------|
| `QUANTCONNECT_VWAP_RSI.py` | VWAP+RSI — Jan 2020–Jun 2024, 5 versions, IB cost model, PSR validation |
| `QUANTCONNECT_ML_RIDGE.py` | ML Ridge — 10 features, quarterly rolling walk-forward, IC+PSR output, NVDA+MSFT |

### charts/

| File | Description |
|------|-------------|
| `06_TRADINGVIEW_FULL_STRATEGY.pine` | Full VWAP+RSI strategy with regime filter — live chart |
| `07_TRADINGVIEW_TRAINING.pine` | Signal logic with annotated entry/exit conditions |

### docs/

| File | Description |
|------|-------------|
| `RESEARCH_NOTES.md` | Session-by-session log — hypothesis, run results, findings, next steps |
| `ORB_ALPHA_RESEARCH_MEMO.md` | ORB research memo — 12 chapters, all runs, levers theory, conclusion |
| `ML_ALPHA_RESEARCH_MEMO.md` | ML Ridge memo — walk-forward, IC, PSR, DSR, five numbers framework |

---

## Hypothesis 1 — VWAP + RSI Mean Reversion

**Hypothesis:** Intraday prices dislocate from VWAP and mean-revert. RSI extremes identify the entry point.

**Status: CLOSED — no statistically significant edge found (PSR < 1% across all versions)**

| Ver | Ticker | Change | Return | PSR | Win Rate | Fees |
|-----|--------|--------|--------|-----|----------|------|
| v1 | AAPL | Baseline | -14.40% | 0.107% | — | $4,667 |
| v2 | AAPL | + Regime filter (ATR + SMA200) | -2.33% | 0.295% | 53% | $1,626 |
| v3 | AAPL | Stop loss 1.0% → 0.5% | -4.32% | 0.132% | 50% | $1,788 |
| v4 | AAPL | RSI period 14 → 7 | -21.30% | 0.000% | — | $3,214 |
| v5a | IWM | RSI(7) + stop 0.75% | -22.53% | 0.001% | — | $4,300 |
| v5b | IWM | RSI(14) + stop 1.0% | -12.22% | 0.005% | — | $2,075 |

**Key findings:** The regime filter was the only structural improvement — cutting fees 65% and reducing COVID crash exposure. IWM underperformed AAPL because ETF intraday moves are driven by macro flows, not idiosyncratic stock noise that VWAP mean reversion requires. Hypothesis closed with full documentation.

---

## Hypothesis 2 — ML Ridge Intraday Momentum

**Hypothesis:** A Ridge Regression model combining 10 intraday features can predict 30-minute forward returns on NVDA and MSFT during the 10am–11am ET institutional momentum window.

**Status: OPEN — gross edge confirmed, signal needs refinement**

### yfinance 60-day Walk-Forward Results (3 folds, 5 tickers)

| Ticker | Avg IC | Avg Sharpe | Gross Positive Folds | PSR | DSR |
|--------|--------|------------|----------------------|-----|-----|
| NVDA | +0.054 | +1.03 | 3/3 | 47.4% | <50% |
| MSFT | +0.031 | +0.82 | 3/3 | 17.1% | <50% |
| AAPL | -0.031 | -0.41 | 0/3 | 0.3% | <50% |

**Key finding:** NVDA gross positive 3/3 folds with IC above threshold. PSR and DSR below confirmation level — data limit (60 days, ~20 trades per fold) prevents statistical confirmation. DSR benchmark SR* ≈ 1.77 at k=15 trials; NVDA best Sharpe 1.44 < SR* — result within range of selection bias. QuantConnect needed.

### QuantConnect 4.5-year Backtest (Jan 2020 – Jun 2024)

| Metric | Value |
|--------|-------|
| Net Return | +21.24% |
| Compounding Annual Return | +4.46% |
| Total Fees | $4,217 |
| Sharpe Ratio | 0.127 |
| Max Drawdown | 27.3% |
| Win Rate | 51% |
| Total Orders | 4,000 |
| IC (NVDA) | -0.047 |
| IC (MSFT) | +0.034 |

**Key finding:** Gross edge confirmed over 4.5 years. IC NVDA negative (model predicts wrong direction on NVDA); IC MSFT positive but below 0.05 threshold. Position sizing (45% per ticker in high-volatility stocks) collapses Sharpe despite positive net return. Signal needs feature redesign and reduced position concentration.

---

## The Five Numbers — Research Framework

Every run is read in this order:

| # | Metric | Question | Threshold |
|---|--------|----------|-----------|
| 1 | **Gross Return** | Does the signal have edge before fees? | > 0 |
| 2 | **Total Costs** | What is the fee gap to close? | As low as possible |
| 3 | **Net Return** | What is the result after all costs? | > 0 |
| 4 | **IC** | Do predictions track actual returns? | > 0.05 |
| 5 | **PSR** | Is the Sharpe statistically real? | > 95% |
| + | **DSR** | Does Sharpe survive multiple testing? | > 95% |

**DSR (Deflated Sharpe Ratio)** — the senior-level addition: when you run k strategies, the expected max Sharpe by chance is SR* ≈ 1.77 (k=15). DSR = probability your Sharpe exceeds SR*, not just zero. This is the correct standard for production signal approval.

---

## Key Concepts Implemented

**Signal Research**
- VWAP daily reset — volume-weighted intraday fair value
- Wilder RSI (exponential smoothing) vs. simple rolling RSI
- Opening Range Breakout — institutional momentum confirmation
- ML Ridge Regression — 10-feature intraday signal with L2 regularisation
- Look-ahead bias prevention via `position.shift(1)`

**ML & Validation**
- Walk-forward validation — train on past, test on unseen future
- Rolling walk-forward (production) — quarterly retraining window
- IC (Information Coefficient) — Pearson correlation, predictions vs. actual returns
- Conviction threshold — trade only top 30% strongest predictions
- StandardScaler normalisation before Ridge training

**Statistical Credibility**
- PSR (Probabilistic Sharpe Ratio) — Lopez de Prado (2014)
- DSR (Deflated Sharpe Ratio) — multiple testing correction
- Bootstrapped Sharpe confidence intervals
- Purged walk-forward with embargo — prevents data leakage in ML validation
- Monte Carlo P&L path simulation
- IC decay analysis

**Risk & Execution**
- Transaction costs: commission + bid-ask spread + market impact
- Kyle lambda — price impact per unit of signed order flow
- TWAP and VWAP execution algorithms
- Almgren-Chriss impact model reference
- Pre-trade risk checklist

**Portfolio Construction**
- Cross-sectional long/short ranking
- Volatility scaling and sector neutralisation
- Beta-neutral portfolio construction
- Risk parity allocation

---

## Stack

- **Python:** pandas, numpy, matplotlib, scikit-learn, scipy, xgboost, yfinance
- **Platform:** QuantConnect LEAN (production backtest environment)
- **Charting:** TradingView Pine Script v5
- **Methodology:** Interactive Brokers cost model, minute-resolution equity data

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib scikit-learn scipy xgboost
```

Each file in `signals/` is self-contained and runs independently. Progress from `01` through `05` to build the full framework layer by layer. Run `backtests/` files in QuantConnect LEAN.

---

*[Bullseye Alpha](https://bullseyealpha.com) — Systematic equity research*
