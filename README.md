# Systematic Alpha Research
### Signal Construction · Walk-Forward Validation · Statistical Testing · Institutional Risk Framework

A complete, end-to-end quantitative research framework built across four progressive levels — from first signal to production-grade validation. Every hypothesis is tested, documented, and either confirmed or killed with evidence. Every failure is kept.

Built to the standard of a systematic equity fund research process.

---

## What This Demonstrates

- **Complete research lifecycle** — Idea → Features → Signal → Backtest → Statistical Testing → Execution Modelling → Production Gate
- **Four research levels** — Entry through Senior, each building on the last with a concrete reason for every upgrade
- **Two full alpha hypotheses** — VWAP+RSI mean reversion (closed, edge not found) and ML Ridge momentum (open, gross edge confirmed)
- **Seven standalone senior research studies** — each with its own hypothesis, data pipeline, five numbers scorecard, and chart
- **Production-grade validation** — Purged walk-forward with embargo, IC, PSR, DSR multiple testing correction
- **Institutional execution framework** — TWAP, VWAP, Almgren-Chriss impact model, pre-trade go/no-go gate
- **QuantConnect LEAN** — 4.5-year backtests with Interactive Brokers cost model on both hypotheses
- **Honest documentation** — failed hypotheses recorded with full analysis, not discarded

---

## Repository Structure

```
├── signals/        Levels 1–3: pipeline techniques within a hypothesis
├── research/       Level 4: standalone senior research studies (own hypothesis per file)
├── backtests/      QuantConnect LEAN production backtests
├── charts/         Research charts — five numbers scorecards, equity fans, factor plots
└── docs/           Alpha research memos, session notes, findings
```

---

## signals/ — Levels 1 to 3

| File | Level | Hypothesis / Purpose |
|------|-------|----------------------|
| `01_ENTRY_BEGINNER.py` | 1 — Entry | VWAP+RSI mean reversion · cost model · multi-ticker sweep · win rate · P/L ratio · expected value |
| `02_ORB_STRATEGY.py` | 2 — Junior | Opening Range Breakout · volume filter · 10am–11am time window · gross vs net analysis |
| `02_JUNIOR_TO_ADVANCED.py` | 2 — Junior | Full research cheat sheet · regime filter · walk-forward · Sortino · parameter sweep |
| `03_INTERMEDIATE_ADVANCED_FIXES.py` | 3 — Intermediate | Daily VWAP reset · Wilder RSI · volatility regime detection · earnings filter · IC decay |
| `03_ML_RIDGE_SIGNAL.py` | 3–4 — Intermediate/Senior | ML Ridge on 10 features · walk-forward 3 folds · IC · PSR · DSR · purged WF · Monte Carlo · pre-trade checklist |
| `04_SENIOR_LEVEL.py` | 4 — Senior | Ridge/Lasso/RF/XGBoost comparison · portfolio optimization · TAQ microstructure reference |
| `05_SENIOR_FINAL_BOOST.py` | 4 — Senior | DSR deep dive · Monte Carlo P&L paths · pre-trade risk checklist · production deployment notes |

---

## research/ — Level 4 Standalone Research Studies

Each file is a self-contained research study: own hypothesis, own data pipeline, own five numbers scorecard, own chart.

| File | Research Question |
|------|------------------|
| `04_PURGED_WALK_FORWARD.py` | Does removing label leakage (purge + embargo) materially change the signal's measured IC and Sharpe? |
| `05_MONTE_CARLO.py` | Is the NVDA gross edge real, or is the equity curve the product of lucky trade sequencing? |
| `06_PORTFOLIO_OPTIMIZATION.py` | Does smarter allocation (Min-Variance, Max-Sharpe, Risk Parity) beat equal-weighting two high-vol tech stocks? |
| `07_FACTOR_MODELING.py` | Is the strategy return genuine alpha, or disguised exposure to market, momentum, and volatility factors? |
| `08_MICROSTRUCTURE.py` | How large is the true cost of execution on NVDA vs MSFT — spread, illiquidity, OFI signal quality? |
| `09_EXECUTION_MODELS.py` | For a $500,000 position, does VWAP execution reduce impact cost by ≥20% vs naive single-shot execution? |
| `10_PRETRADE_CHECKLIST.py` | Does the NVDA ML Ridge signal pass the 8-item production gate (5 critical + 3 advisory)? |

---

## backtests/ — QuantConnect LEAN

| File | Description |
|------|-------------|
| `QUANTCONNECT_VWAP_RSI.py` | VWAP+RSI · Jan 2020–Jun 2024 · 5 versions · IB cost model · PSR validation |
| `QUANTCONNECT_ML_RIDGE.py` | ML Ridge · 10 features · quarterly rolling walk-forward · IC+PSR output · NVDA+MSFT |

---

## The Five Numbers Framework

Every run is read in this order — no exceptions:

| # | Metric | Question | Threshold |
|---|--------|----------|-----------|
| 1 | **Gross Return** | Does the signal have edge before fees? | > 0 |
| 2 | **Total Costs** | What is the fee gap to close? | minimise |
| 3 | **Net Return** | What is the result after all costs? | > 0 |
| 4 | **IC** | Do predictions track actual returns? | > 0.05 |
| 5 | **PSR** | Is the Sharpe statistically real? | > 95% |
| + | **DSR** | Does Sharpe survive multiple testing? | > 95% |

---

## Hypothesis 1 — VWAP + RSI Mean Reversion

**Hypothesis:** Intraday prices dislocate from VWAP and mean-revert. RSI extremes identify the entry point.

**Status: CLOSED — no statistically significant edge found (PSR < 1% across all versions)**

| Ver | Ticker | Change | Net Return | PSR | Win Rate | Fees |
|-----|--------|--------|-----------|-----|----------|------|
| v1 | AAPL | Baseline | -14.40% | 0.107% | — | $4,667 |
| v2 | AAPL | + Regime filter (ATR + SMA200) | -2.33% | 0.295% | 53% | $1,626 |
| v3 | AAPL | Stop loss 1.0% → 0.5% | -4.32% | 0.132% | 50% | $1,788 |
| v4 | AAPL | RSI period 14 → 7 | -21.30% | 0.000% | — | $3,214 |
| v5a | IWM | RSI(7) + stop 0.75% | -22.53% | 0.001% | — | $4,300 |
| v5b | IWM | RSI(14) + stop 1.0% | -12.22% | 0.005% | — | $2,075 |

**Key findings:** Regime filter was the only structural improvement — cutting fees 65% and reducing COVID crash exposure. IWM underperformed AAPL because ETF intraday moves are driven by macro flows, not the idiosyncratic stock noise VWAP mean reversion requires. Hypothesis closed with full documentation.

---

## Hypothesis 2 — ML Ridge Intraday Momentum

**Hypothesis:** A Ridge Regression model combining 10 intraday features can predict 30-minute forward returns on NVDA and MSFT during the 10am–11am ET institutional momentum window.

**Status: OPEN — gross edge confirmed, statistical confirmation requires longer history**

### yfinance Walk-Forward Results (3 folds, 5 tickers)

| Ticker | Avg IC | Avg Sharpe | Gross Positive Folds | PSR | DSR |
|--------|--------|------------|----------------------|-----|-----|
| NVDA | +0.054 | +1.03 | 3/3 | 47.4% | <50% |
| MSFT | +0.031 | +0.82 | 3/3 | 17.1% | <50% |
| AAPL | -0.031 | -0.41 | 0/3 | 0.3% | <50% |

**Key finding:** NVDA gross positive 3/3 folds with IC above threshold. PSR and DSR below confirmation level — 60 days (~20 trades per fold) is insufficient for statistical confirmation. DSR benchmark SR* ≈ 1.77 at k=15 trials; NVDA best Sharpe 1.44 < SR* — within range of selection bias. QuantConnect validation required.

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

**Key finding:** Gross edge confirmed over 4.5 years. IC NVDA negative (model predicts wrong direction post-2022 regime change). Position sizing at 45% per ticker in high-volatility stocks collapses Sharpe despite positive net return. Signal needs feature redesign and reduced position concentration.

---

## Level 4 Research Charts

### Purged Walk-Forward — Leakage Test
Standard vs purged validation: does removing label overlap change the Sharpe?

![Purged Walk-Forward](charts/five_numbers_nvda_purged_wf.png)

---

### Monte Carlo P&L — 1,000 Bootstrap Paths
Is the equity curve the product of skill or lucky trade sequencing?

![Monte Carlo](charts/monte_carlo_paths.png)

---

### Portfolio Optimization — Equal Weight vs Risk Parity
Does smarter allocation improve portfolio Sharpe on NVDA + MSFT?

![Portfolio Optimization](charts/portfolio_optimization.png)

---

### Factor Modeling — Alpha Attribution
Is the strategy return genuine alpha or disguised market / momentum exposure?

![Factor Modeling](charts/factor_modeling.png)

---

### Market Microstructure — Spread, Liquidity, Order Flow
Roll spread, Amihud illiquidity, OFI Z-score, Corwin-Schultz estimator on NVDA and MSFT.

![Microstructure](charts/microstructure.png)

---

### Execution Models — TWAP vs VWAP vs Almgren-Chriss
$500,000 position — which execution method minimises market impact?

![Execution Models](charts/execution_models.png)

---

### Pre-Trade Checklist — Go / No-Go Gate
8-item production gate: 5 critical stops + 3 advisory monitors.

![Pre-Trade Checklist](charts/pretrade_checklist.png)

---

## Key Concepts Implemented

**Signal Construction**
- VWAP daily reset — volume-weighted intraday fair value, restarted each trading date
- Wilder RSI — exponential smoothing vs simple rolling RSI
- Opening Range Breakout — institutional momentum confirmation with volume gate
- ML Ridge Regression — 10-feature intraday signal with L2 regularisation
- Look-ahead bias prevention — `position.shift(1)` enforced at every level

**Validation**
- Walk-forward validation — train on past, test on unseen future
- Purged walk-forward with embargo — removes label leakage at fold boundaries
- Rolling walk-forward — quarterly retraining window for production deployment
- IC (Information Coefficient) — Pearson correlation, predictions vs actual returns
- Conviction threshold — trade only top 30% strongest predictions

**Statistical Credibility**
- PSR (Probabilistic Sharpe Ratio) — Lopez de Prado 2014, non-normal correction
- DSR (Deflated Sharpe Ratio) — multiple testing correction at k=15 trials, SR* ≈ 1.77
- Bootstrapped Monte Carlo — 1,000 equity paths, P5/P50/P95, P(ruin < -20%)
- IC decay analysis — signal degradation over forward return horizon

**Portfolio Construction**
- Equal weight, Min-Variance, Max-Sharpe (Tangency Portfolio), Risk Parity
- Diversification ratio — portfolio vol vs weighted average asset vol
- Beta-neutral construction reference

**Execution & Microstructure**
- Roll's spread estimate — bid-ask proxy from return autocorrelation
- Amihud illiquidity ratio — price impact per dollar of volume
- Order Flow Imbalance (OFI) Z-score — institutional order direction signal
- Corwin-Schultz spread estimator — high-low based, no tick data required
- TWAP and VWAP execution algorithms — time-uniform vs volume-proportional slicing
- Almgren-Chriss impact model — permanent vs temporary impact, optimal trajectory
- Implementation shortfall — decision price to average execution price gap

**Risk & Production**
- Pre-trade checklist — 5 critical gates + 3 advisory monitors, GO/WATCH/NO-GO verdict
- Kyle lambda — price impact per unit of signed order flow
- Factor regression — OLS alpha attribution, market/momentum/vol betas, R², t-stat
- Win rate, P/L ratio, expected value — level 1 baseline metrics

---

## Stack

- **Python** — pandas, numpy, matplotlib, scikit-learn, scipy, xgboost, yfinance
- **Platform** — QuantConnect LEAN (production backtest environment)
- **Charting** — TradingView Pine Script v5
- **Methodology** — Interactive Brokers cost model, minute-resolution equity data

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib scikit-learn scipy xgboost
```

`signals/` files are self-contained and run independently.
Progress from `01` through `05_SENIOR_FINAL_BOOST.py` for the full pipeline.
`research/` files each run independently as standalone research studies.
`backtests/` files run in QuantConnect LEAN (free account at quantconnect.com).

---

## Research Progression

```
Level 1 — Entry          01_ENTRY_BEGINNER.py
                         VWAP · RSI · cost model · multi-ticker · win rate · EV

Level 2 — Junior         02_ORB_STRATEGY.py  +  02_JUNIOR_TO_ADVANCED.py
                         Regime filter · ORB · walk-forward · QuantConnect

Level 3 — Intermediate   03_INTERMEDIATE_ADVANCED_FIXES.py  +  03_ML_RIDGE_SIGNAL.py
                         ML Ridge · 10 features · IC · PSR · DSR

Level 4 — Senior         research/  (7 standalone studies)
                         Purged WF · Monte Carlo · Portfolio Opt · Factor Model
                         Microstructure · Execution Models · Pre-Trade Gate
```

---

*[Bullseye Alpha](https://bullseyealpha.com) — Systematic equity research*
