# Alpha Research Memo
## Research Session Log — Full Programme Timeline
### Bullseye Alpha | Patience Fuglo | May 2026
#### Four levels · Two hypotheses · Nine sessions · All findings recorded

---

## How to read this log

Each session maps to a research memo. The memo contains the full hypothesis,
run history, five numbers, and conclusions. This log is the timeline — what
happened when, what changed, and where to find the full analysis.

```
MEMO INDEX
──────────────────────────────────────────────────────────────────
Hypothesis 1  VWAP + RSI         →  VWAP_RSI_MEMO.md
Hypothesis 2  Opening Range ORB  →  ORB_ALPHA_RESEARCH_MEMO.md
Hypothesis 3  ML Ridge Signal    →  ML_ALPHA_RESEARCH_MEMO.md
Level 4       Senior Research    →  SENIOR_RESEARCH_MEMO.md
──────────────────────────────────────────────────────────────────
```

---

## Session 1 — May 1, 2026 | Hypothesis 1 begins

**Strategy:** VWAP + RSI mean reversion
**Platform:** QuantConnect LEAN | IB cost model | $100k capital | Jan 2020 – Jun 2024
**File:** `backtests/QUANTCONNECT_VWAP_RSI.py`

**What ran:** 4 versions on AAPL changing one variable at a time — baseline, regime filter, stop loss, RSI period.

| Version | Key Change | Net Return | PSR |
|---------|-----------|-----------|-----|
| v1 | Baseline | −14.40% | 0.107% |
| v2 | Regime filter (ATR + SMA200) | −2.33% | 0.295% |
| v3 | Stop loss 1.0% → 0.5% | −4.32% | 0.132% |
| v4 | RSI period 14 → 7 | −21.30% | 0.000% |

**Outcome:** Regime filter was the only structural improvement. All versions PSR < 1%.

→ Full analysis: [VWAP_RSI_MEMO.md](VWAP_RSI_MEMO.md)

---

## Session 2 — May 2, 2026 | Hypothesis 1 closed

**What ran:** Extended best AAPL configuration to IWM (out-of-sample instrument).

| Version | Ticker | Net Return | PSR |
|---------|--------|-----------|-----|
| v5a | IWM | −22.53% | 0.001% |
| v5b | IWM | −12.22% | 0.005% |

**Finding:** IWM worse than AAPL on every metric. ETF intraday moves driven by macro flows,
not idiosyncratic stock noise that VWAP mean reversion requires.

**Decision:** Hypothesis 1 closed. PSR < 1% across all 6 versions. No edge detected.

→ Full analysis: [VWAP_RSI_MEMO.md](VWAP_RSI_MEMO.md)

---

## Session 3 — May 10, 2026 | Hypothesis 2 — Opening Range Breakout

**Strategy:** Opening Range Breakout (ORB)
**Platform:** VS Code + yfinance | 5-minute bars | 60-day window
**Tickers:** AAPL, MSFT, NVDA, SPY, QQQ
**File:** `signals/02_ORB_STRATEGY.py`

**What changed from Hypothesis 1:** Direction inverted — momentum, not mean reversion.
Volume filter and time window (10am–11am ET) as the core levers.

| Run | Key Change | Gross | Net | Decision |
|-----|-----------|-------|-----|----------|
| 1 | Baseline | — | −4.0% | Add volume filter |
| 2 | Volume 1.5× | +2.9% | −4.0% | Tighten |
| 3 | Volume 2.5× | +2.6% | −2.0% | Add min move |
| 4 | 2.5× + 0.2% move | +1.74% | −1.66% | Fix trade count |
| 5 | Volume 2.0× | +1.34% | −3.32% | Reverted |
| 6 | 10am–11am window | +0.56% | +0.41% | ← best run |
| 7 | 10am–12pm window | +0.53% | −0.18% | Data wall |

**Finding:** Gross positive every run. Net positive first time with 10am–11am filter.
Data limit: 60 days produces ~2 trades/ticker with strict filters — below 50-trade threshold.

**Decision:** Open. Edge confirmed (gross), statistical confirmation requires QuantConnect.

→ Full analysis: [ORB_ALPHA_RESEARCH_MEMO.md](ORB_ALPHA_RESEARCH_MEMO.md)

---

## Session 4 — May 11, 2026 | Hypothesis 3 — ML Ridge Signal

**Strategy:** Ridge Regression on 10 intraday features
**Platform:** VS Code + yfinance + scikit-learn | 5-minute bars | 60-day window
**Tickers:** AAPL, MSFT, NVDA, SPY, QQQ
**File:** `signals/03_ML_RIDGE_SIGNAL.py`

**New concept introduced:** IC (Information Coefficient) — correlation between
model predictions and actual returns. IC > 0.05 = useful signal.

| Run | Key Change | Gross | IC | Net | Decision |
|-----|-----------|-------|-----|-----|----------|
| 1 | Raw signal, every bar | −2.3% | −0.04 | −25.2% | Add conviction filter |
| 2 | 30-min forward target | −1.8% | −0.06 | −12.6% | Add time window |
| 3 | 10am–11am window | +0.34% | −0.06 | −1.58% | Expand features |
| 4 | 10 features | — | +0.028 | — | Add walk-forward |
| 5 | Walk-forward 3 folds | NVDA +0.054 | varies | — | Run multi-ticker |
| 6 | 5 tickers × 3 folds | NVDA IC +0.054 | PSR 47% | — | QC needed |

**Finding:** NVDA gross positive 3/3 folds. IC above threshold.
PSR and DSR below confirmation — 60 days insufficient (20 trades/fold).

→ Full analysis: [ML_ALPHA_RESEARCH_MEMO.md](ML_ALPHA_RESEARCH_MEMO.md)

---

## Session 5 — May 11, 2026 | QuantConnect ML Validation

**Strategy:** ML Ridge signal validated on 4.5-year production backtest
**Platform:** QuantConnect LEAN | IB cost model | Jan 2020 – Jun 2024
**File:** `backtests/QUANTCONNECT_ML_RIDGE.py`

| Metric | Value |
|--------|-------|
| Net Return | +21.24% |
| Compounding Annual Return | +4.46% |
| Total Fees | $4,217 |
| Sharpe Ratio | 0.127 |
| Max Drawdown | −27.3% |
| Win Rate | 51% |
| IC (NVDA) | −0.047 |
| IC (MSFT) | +0.034 |

**Finding:** Gross edge confirmed. IC turned negative on NVDA post-2022 (regime change).
Position sizing at 45%/ticker in high-vol stocks collapses Sharpe. Feature redesign required.

→ Full analysis: [ML_ALPHA_RESEARCH_MEMO.md](ML_ALPHA_RESEARCH_MEMO.md)

---

## Session 6 — May 11, 2026 | DSR — Multiple Testing Correction

**New concept introduced:** DSR (Deflated Sharpe Ratio) — Lopez de Prado 2014.
Running k=15 strategy trials, expected max Sharpe by chance = SR* ≈ 1.77.
NVDA best Sharpe 1.44 < SR* 1.77 → DSR < 50% → result within chance range.

**Finding:** NVDA signal does not pass multiple testing correction at yfinance data scale.
Requires QuantConnect history to accumulate enough trades for DSR confirmation.

→ Full analysis: [ML_ALPHA_RESEARCH_MEMO.md](ML_ALPHA_RESEARCH_MEMO.md) — Chapter 12

---

## Session 7 — May 11, 2026 | Senior Level Research Topics

All items run on NVDA and MSFT. Full results in `SENIOR_RESEARCH_MEMO.md`.

| Topic | File | Key Finding |
|-------|------|-------------|
| Microstructure | `research/08_MICROSTRUCTURE.py` | NVDA half-spread ~6bp — net IC borderline |
| TWAP/VWAP/AC | `research/09_EXECUTION_MODELS.py` | VWAP reduces impact vs TWAP on NVDA |
| Portfolio Optimization | `research/06_PORTFOLIO_OPTIMIZATION.py` | Risk parity corrects hidden 85/15 risk split |
| Factor Modeling | `research/07_FACTOR_MODELING.py` | β_mkt ≈ 0.03, R² < 0.02 — idiosyncratic |
| Monte Carlo | `research/05_MONTE_CARLO.py` | P(ruin) high at 60-day scale — sizing issue |
| Pre-trade Checklist | `research/10_PRETRADE_CHECKLIST.py` | WATCH — all critical pass, DSR advisory fail |
| Paper trading | See below | Instructions written |

### Paper Trading Setup — QuantConnect

Paper trading runs the live algorithm on real-time market data with simulated capital.
No real money at risk. Purpose: verify backtest edge persists in live conditions.

**Why paper trading before live:**
```
Backtests assume:      perfect fills at bar close price
Live trading has:      latency, partial fills, real spread, order book depth

Paper trading reveals the gap before it costs money.
```

**How to deploy `QUANTCONNECT_ML_RIDGE.py` in paper mode:**

1. Log into quantconnect.com
2. Open the project containing `QUANTCONNECT_ML_RIDGE.py`
3. Click **Deploy Live** (top right)
4. Under **Brokerage**, select **QuantConnect Paper Trading**
5. Set **Cash** to $100,000 (matches backtest starting capital)
6. Click **Deploy**

**What to monitor after deployment:**

| Metric | Backtest value | Alert if |
|--------|---------------|----------|
| IC (NVDA) | −0.047 | Stays below 0 for 4+ consecutive weeks |
| IC (MSFT) | +0.034 | Drops below 0 consistently |
| Win rate | 51% | Falls below 45% for 30+ days |
| Fee drag | ~$80/month | Exceeds $150/month (signal overtrading) |
| Drawdown | −27.3% max | Approaches −15% (reduce size immediately) |
| Model retrain | Every 63 days | Confirm log entry each retrain |

**Decision rule after 30 days:**
```
If MSFT IC holds positive  →  reduce position from 45% to 25%/ticker → redeploy
If both ICs negative       →  feature redesign required, pause trading
If drawdown > −15%         →  halt paper trading, review signal
```

→ Full senior analysis: [SENIOR_RESEARCH_MEMO.md](SENIOR_RESEARCH_MEMO.md)

---

## Session 8 — May 12, 2026 | Standalone Files + Repo Finalisation

**What changed:**
- Seven Level 4 research files moved to `research/` folder (each standalone)
- Level 1 fix: VWAP proper daily reset + win rate / P/L ratio / expected value added
- `VWAP_RSI_MEMO.md` written (this session) — Hypothesis 1 properly documented
- README rewritten: embedded charts, five numbers, research progression map
- Repo description updated on GitHub

**Repository final structure:**
```
signals/     Levels 1–3 — pipeline techniques within a hypothesis
research/    Level 4 — 7 standalone research studies, own hypothesis each
backtests/   QuantConnect LEAN production backtests
charts/      All research charts — five numbers scorecards, equity fans
docs/        Four research memos + this session log
```

**Docs folder:**
```
VWAP_RSI_MEMO.md          Hypothesis 1 — VWAP + RSI (closed)
ORB_ALPHA_RESEARCH_MEMO.md Hypothesis 2 — ORB (open, data limit)
ML_ALPHA_RESEARCH_MEMO.md  Hypothesis 3 — ML Ridge (open, QC validated)
SENIOR_RESEARCH_MEMO.md    Level 4 — 7 standalone studies
RESEARCH_NOTES.md          This session log
```

---

## Next Steps

```
1. Paper trading    Deploy QUANTCONNECT_ML_RIDGE.py in QC paper mode
                    Monitor IC, win rate, drawdown weekly for 30 days

2. Interview prep   Level 1 → Level 4, story thread training
                    One concept at a time, full absorption before moving on
                    Starting from VWAP+RSI and building to senior research
```

---

*Bullseye Alpha | Systematic Equity Research | bullseyealpha.com*
