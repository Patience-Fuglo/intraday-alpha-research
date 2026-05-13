# Alpha Research Memo
## Programme Index — Full Research Timeline
### Bullseye Alpha | Patience Fuglo | May 2026
#### Four levels · Two hypotheses · Nine sessions · Navigation map to all memos

---

## Memo Index

| Memo | Hypothesis / Topic | Status |
|------|--------------------|--------|
| [VWAP_RSI_MEMO.md](VWAP_RSI_MEMO.md) | Hypothesis 1 — VWAP + RSI mean reversion | Closed — no edge |
| [ORB_ALPHA_RESEARCH_MEMO.md](ORB_ALPHA_RESEARCH_MEMO.md) | Hypothesis 2 — Opening Range Breakout | Open — data limit |
| [ML_ALPHA_RESEARCH_MEMO.md](ML_ALPHA_RESEARCH_MEMO.md) | Hypothesis 3 — ML Ridge intraday momentum | Open — QC validated |
| [SENIOR_RESEARCH_MEMO.md](SENIOR_RESEARCH_MEMO.md) | Level 4 — Seven standalone research studies | Complete |

---

## Session Timeline

### Session 1 — May 1, 2026
**Hypothesis 1: VWAP + RSI Mean Reversion**
Platform: QuantConnect LEAN · IB cost model · $100k capital · Jan 2020–Jun 2024 (4.5 years)

```
HYPOTHESIS
When intraday price dislocates from VWAP (fair value) and RSI confirms the move
is extreme, price will snap back. We bet on the reversion.

Long:  price below VWAP AND RSI < 25  →  buy, expect recovery
Short: price above VWAP AND RSI > 75  →  sell, expect decline
Exit:  price crosses back through VWAP →  reversion complete

The bet: "Price has moved too far. It will come back."
```

**v1 — Baseline**
```
Gross Return    −14.40%     FAIL   signal is anti-predictive — gross negative
Total Costs     $4,667      HIGH   many trades, small account
Net Return      −14.40%     FAIL   costs compounding a bad signal
PSR             0.107%      FAIL   pure statistical noise

Reading: gross is negative. The signal fires in trending markets where
price does not revert — it keeps going. Stop loss and filters cannot fix
a gross-negative signal. The market environment is wrong.
Decision: add regime filter to prevent trading in trending conditions.
```

**v2 — Regime filter (ATR volatility + SMA200 trend)**
```
Gross Return    −2.33%      FAIL   still negative but 12pp improvement
Total Costs     $1,626      →      65% fee reduction — far fewer trades
Net Return      −2.33%      FAIL   negative
Win Rate        53%         →      more than half of trades are winners
P/L Ratio       0.85        FAIL   average loss (0.33%) > average win (0.28%)
PSR             0.295%      FAIL   still noise

Expected value: (53% × +0.28%) + (47% × −0.33%) = −0.007% per trade
The regime filter was the biggest structural improvement in the research.
It cut fees 65% and reduced loss from −14.4% to −2.33%. But EV per trade
is negative. A 53% win rate with a 0.85 P/L ratio loses money.
Decision: tighten stop loss to reduce average losing trade size.
```

**v3 — Stop loss tightened (1.0% → 0.5%)**
```
Gross Return    −4.32%      FAIL   worse than v2 — stop cuts winners short
Net Return      −4.32%      FAIL
Win Rate        50%         →      dropped (stop triggers on valid reversions)
P/L Ratio       0.95        →      slight improvement but still below 1.0
PSR             0.132%      FAIL

Reading: tightening a stop changes WHEN you lose, not WHETHER you lose.
The stop exited winners before reversion completed. Net return got worse.
Lesson: stops manage risk. They do not create edge where none exists.
Decision: try faster RSI to capture more extreme readings.
```

**v4 — RSI period shortened (14 → 7)**
```
Gross Return    −21.30%     FAIL   worst result — signal became anti-predictive
Total Costs     $3,214      HIGH   more trades = more fees
Net Return      −21.30%     FAIL
PSR             0.000%      FAIL   pure random — zero statistical meaning

Reading: RSI(7) on 5-minute bars fires after 35 minutes of movement.
That is not exhaustion — it is a normal intraday wiggle. More signals
is not better signals. Frequency and quality are independent.
Decision: close AAPL. Extend to IWM as out-of-sample test.
```

→ Full analysis: [VWAP_RSI_MEMO.md](VWAP_RSI_MEMO.md)

---

### Session 2 — May 2, 2026
**Hypothesis 1 closed: IWM out-of-sample test**

```
QUESTION: Does VWAP mean reversion work on ETFs?
The best AAPL settings (RSI 14, stop 1.0%) applied to IWM (Russell 2000 ETF).
```

**v5a — IWM, RSI(7) + stop 0.75%**
```
Net Return    −22.53%     FAIL   worse than every AAPL version
PSR           0.001%      FAIL   effectively zero
```

**v5b — IWM, RSI(14) + stop 1.0% (best AAPL settings)**
```
Net Return    −12.22%     FAIL   worse than AAPL v2 (−2.33%)
PSR           0.005%      FAIL   near zero
```

```
WHY IWM UNDERPERFORMED AAPL
AAPL intraday moves = idiosyncratic events (earnings, news, upgrades)
                      → local dislocations that fade back to fair value

IWM intraday moves  = macro flows (Fed decisions, risk-on/risk-off rotation)
                      → systematic moves that do not revert intraday

VWAP mean reversion requires idiosyncratic noise to fade.
ETF moves are driven by macro forces that persist.
Same signal. Different instrument. Structurally different result.

VERDICT: Hypothesis 1 closed.
PSR < 1% across all 6 versions. No statistically detectable edge
on AAPL or IWM. Documented and archived. Research continues.
```

→ Full analysis: [VWAP_RSI_MEMO.md](VWAP_RSI_MEMO.md)

---

### Session 3 — May 10, 2026
**Hypothesis 2 begins: Opening Range Breakout**
Platform: VS Code + yfinance · 5-min bars · 60-day window · AAPL, MSFT, NVDA, SPY, QQQ

| Run | Change | Gross | Net | Decision |
|-----|--------|-------|-----|----------|
| 1 | Baseline | — | −4.0% | Add volume filter |
| 2 | Volume 1.5× | +2.9% | −4.0% | Tighten |
| 3 | Volume 2.5× | +2.6% | −2.0% | Add min move |
| 4 | 2.5× + 0.2% move | +1.74% | −1.66% | Fix trade count |
| 5 | Volume 2.0× | +1.34% | −3.32% | Reverted |
| 6 | 10am–11am window | +0.56% | +0.41% | ← best |
| 7 | 10am–12pm window | +0.53% | −0.18% | Data wall |

Outcome: Gross positive every run. Net positive with 10am–11am filter. 60-day data limit: ~2 trades/ticker with strict filters. QuantConnect required.

→ Full analysis: [ORB_ALPHA_RESEARCH_MEMO.md](ORB_ALPHA_RESEARCH_MEMO.md)

---

### Session 4 — May 11, 2026
**Hypothesis 3 begins: ML Ridge signal on 10 intraday features**
Platform: VS Code + yfinance + scikit-learn · 5-min bars · 60-day window

| Run | Change | Gross | IC | Net |
|-----|--------|-------|-----|-----|
| 1 | Raw signal, every bar | −2.3% | −0.04 | −25.2% |
| 2 | 30-min forward target | −1.8% | −0.06 | −12.6% |
| 3 | 10am–11am window | +0.34% | −0.06 | −1.58% |
| 4 | 10 features | — | +0.028 | — |
| 5 | Walk-forward 3 folds | NVDA IC +0.054 | PSR 47% | — |

Outcome: NVDA gross positive 3/3 folds, IC above threshold. PSR and DSR below confirmation — 20 trades/fold insufficient. QuantConnect required.

→ Full analysis: [ML_ALPHA_RESEARCH_MEMO.md](ML_ALPHA_RESEARCH_MEMO.md)

---

### Session 5 — May 11, 2026
**QuantConnect ML validation: 4.5-year production backtest**
File: `backtests/QUANTCONNECT_ML_RIDGE.py`

| Metric | Value |
|--------|-------|
| Net Return | +21.24% |
| Sharpe | 0.127 |
| Max Drawdown | −27.3% |
| Win Rate | 51% |
| IC (NVDA) | −0.047 |
| IC (MSFT) | +0.034 |

Outcome: Gross edge confirmed. IC negative on NVDA post-2022 (regime change). Feature redesign required.

→ Full analysis: [ML_ALPHA_RESEARCH_MEMO.md](ML_ALPHA_RESEARCH_MEMO.md)

---

### Session 6 — May 11, 2026
**DSR — Multiple testing correction introduced**

At k=15 trials, SR* ≈ 1.77. NVDA best Sharpe 1.44 < SR* → DSR < 50%. Result within chance range of selection bias. Needs QuantConnect history for confirmation.

→ Full analysis: [ML_ALPHA_RESEARCH_MEMO.md](ML_ALPHA_RESEARCH_MEMO.md) — Chapter 12

---

### Session 7 — May 11, 2026
**Level 4 senior research topics**

| Study | File | Key Finding |
|-------|------|-------------|
| Purged walk-forward | `research/04_` | IC honest after leakage removed |
| Monte Carlo | `research/05_` | P(ruin) high at 60-day scale — sizing issue |
| Portfolio optimization | `research/06_` | Risk parity corrects hidden 85/15 risk split |
| Factor modeling | `research/07_` | β_mkt ≈ 0.03, R² < 0.02 — idiosyncratic |
| Microstructure | `research/08_` | NVDA half-spread ~6bp — net IC borderline |
| Execution models | `research/09_` | VWAP reduces impact vs TWAP on NVDA |
| Pre-trade checklist | `research/10_` | WATCH — all critical pass, DSR advisory fail |
| Paper trading | — | Instructions in SENIOR_RESEARCH_MEMO.md |

→ Full analysis: [SENIOR_RESEARCH_MEMO.md](SENIOR_RESEARCH_MEMO.md)

---

### Session 8 — May 12, 2026
**Repo finalisation**

- Level 4 files moved to `research/` folder
- `VWAP_RSI_MEMO.md` written — Hypothesis 1 properly documented as memo
- `RESEARCH_NOTES.md` replaced by `PROGRAMME_INDEX.md` (this file)
- README rewritten with embedded charts, five numbers, research progression map
- Paper trading instructions added to `SENIOR_RESEARCH_MEMO.md`
- GitHub pushed and repo description updated

---

## Repository Structure

```
signals/     Levels 1–3 — pipeline techniques within a hypothesis
research/    Level 4 — 7 standalone research studies, own hypothesis each
backtests/   QuantConnect LEAN production backtests
charts/      All research charts — five numbers scorecards, equity fans
docs/        Four research memos + this index
```

---

## Next Steps

```
1. Paper trading    Deploy QUANTCONNECT_ML_RIDGE.py in QC paper mode
                    Monitor IC, win rate, drawdown weekly — 30-day window
                    Decision rules in SENIOR_RESEARCH_MEMO.md

2. Interview prep   Level 1 → Level 4, story thread training
                    One concept per session, full absorption before moving on
```

---

*Bullseye Alpha | Systematic Equity Research | bullseyealpha.com*
