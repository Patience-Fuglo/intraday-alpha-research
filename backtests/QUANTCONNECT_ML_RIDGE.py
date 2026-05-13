# ==============================================================================
# QUANTCONNECT — ML RIDGE REGRESSION SIGNAL
# Tickers: NVDA + MSFT | Jan 2020 – Jun 2024 | Interactive Brokers
# Paste this entire file into QuantConnect main.py
# ==============================================================================
#
# *** RESEARCH WORKFLOW ***
#   Idea → Data → Features → Signal → Backtest → Metrics → Robustness
#
# Hypothesis:
#   A Ridge Regression model combining 10 intraday features can predict
#   30-minute forward returns on NVDA and MSFT during the 10am-11am ET
#   institutional momentum window.
#
# Signal validated in yfinance 60-day test (03_ML_RIDGE_SIGNAL.py):
#   NVDA: avg IC +0.054, avg Sharpe +1.03, gross positive 3/3 folds
#   MSFT: gross positive 3/3 folds
#   PSR:  NVDA 47.4% on ~20 trades per fold
#         Needs 5-8x more data for PSR > 95% confirmation
#
# What this file adds vs yfinance:
#   4.5 years of minute-resolution data (vs 60 days)
#   Interactive Brokers cost model (commission + slippage)
#   Rolling walk-forward: retrain quarterly on last 2 years
#   Sufficient trade count for PSR and IC to be statistically meaningful
#
# The Five Numbers — read in this order every run:
#   1. Gross Return  — signal edge before fees? (QC panel: Gross Profit)
#   2. Total Costs   — fee gap to close?        (QC panel: Total Fees)
#   3. Total Return  — net result after costs   (QC panel: Net Profit / logged)
#   4. IC            — do predictions track actual returns? threshold 0.05
#   5. PSR           — is the Sharpe result statistically real? threshold 95%
#   + Max Drawdown, Trades, Sharpe (all in QC Statistics panel)
#
# Prediction before running:
#   Gross stays positive on NVDA over 4.5 years.
#   PSR climbs toward 95% with ~500 trades vs ~20 in 60-day test.
#   Basis: gross positive 3/3 yfinance folds + PSR scaling from 47.4% at 20 trades.
#
# ==============================================================================

from AlgorithmImports import *
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, norm


# The 10 features learned by the model
# Each asks a different question about the current bar
FEATURE_COLS = [
    "vwap_distance",  # is price cheap or expensive vs fair value?
    "rsi",            # how extreme is the recent move?
    "volume_ratio",   # is institutional money active?
    "orb_breakout",   # how far has price broken the opening range?
    "momentum_5",     # what direction has price moved in last 25 minutes?
    "time_of_day",    # how far into the session are we?
    "atr_ratio",      # is this a volatile or calm regime?
    "dollar_vol",     # how much real money is flowing through this bar?
    "gap_size",       # how much did price gap from yesterday's close?
    "ret_zscore",     # is this bar's move unusual vs recent history?
]


class MLRidgeSignal(QCAlgorithm):

    def Initialize(self):

        # ── BACKTEST WINDOW ────────────────────────────────────────────────
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 6, 1)
        self.SetCash(100000)

        # ── TICKERS ────────────────────────────────────────────────────────
        # NVDA: primary signal ticker — avg IC +0.054, Sharpe +1.03 in 60-day test
        # MSFT: secondary signal ticker — gross positive 3/3 walk-forward folds
        # AAPL excluded: gross negative all 3 folds, PSR 0.3% (confirmed dead)
        self.tickers  = ["NVDA", "MSFT"]
        self.sym      = {}

        for t in self.tickers:
            eq = self.AddEquity(t, Resolution.Minute)
            eq.SetDataNormalizationMode(DataNormalizationMode.Raw)
            self.sym[t] = eq.Symbol

        # ── COST MODEL ─────────────────────────────────────────────────────
        # Interactive Brokers: commission + slippage
        # Same cost model used in QUANTCONNECT_VWAP_RSI.py
        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        # ── WARM-UP ────────────────────────────────────────────────────────
        # timedelta form uses the subscription resolution (Minute), not Daily.
        # Resolution.Daily warm-up feeds 6.5-hour bars into the 5-min consolidator,
        # which throws "can not consolidate bars of higher period" at runtime.
        # 60 calendar days x 78 five-min bars/day = ~4,680 bars — enough to
        # initialize all indicators and fill the feature buffer for first training.
        self.SetWarmUp(timedelta(days=60))

        # ── INDICATORS (per ticker) ────────────────────────────────────────
        self.rsi_ind  = {}
        self.vwap_ind = {}
        self.atr_ind  = {}

        for t, s in self.sym.items():
            self.rsi_ind[t]  = self.RSI(s, 14, MovingAverageType.Wilders, Resolution.Minute)
            self.vwap_ind[t] = self.VWAP(s)
            self.atr_ind[t]  = self.ATR(s, 14, MovingAverageType.Wilders, Resolution.Minute)

        # ── ROLLING WINDOWS ────────────────────────────────────────────────
        # Stores last 25 closes and volumes per ticker
        # More efficient than calling History() on every bar
        self.close_win  = {t: RollingWindow[float](25) for t in self.tickers}
        self.volume_win = {t: RollingWindow[float](25) for t in self.tickers}

        # ── 5-MINUTE BAR CONSOLIDATION ─────────────────────────────────────
        # Each ticker gets its own handler — avoids Python closure bug in loops
        for t, s in self.sym.items():
            self.Consolidate(s, timedelta(minutes=5), self._make_handler(t))

        # ── ML MODEL STATE (per ticker) ────────────────────────────────────
        self.model    = {t: None  for t in self.tickers}
        self.scaler   = {t: None  for t in self.tickers}
        self.trained  = {t: False for t in self.tickers}

        # Feature buffer: list of dicts, used for training
        # Grows as the backtest runs — rolling window trims at retraining
        self.feat_buf = {t: [] for t in self.tickers}

        # IC tracker: list of {pred, actual, close}
        # Prediction stored when signal fires, actual filled 6 bars later
        self.ic_data  = {t: [] for t in self.tickers}

        # ── ORB STATE (resets daily per ticker) ────────────────────────────
        self.orb_high   = {t: None  for t in self.tickers}
        self.orb_low    = {t: None  for t in self.tickers}
        self.orb_done   = {t: False for t in self.tickers}
        self.bar_idx    = {t: 0     for t in self.tickers}
        self.prev_close = {t: 0.0   for t in self.tickers}
        self.gap_size   = {t: 0.0   for t in self.tickers}

        # ── POSITION STATE ─────────────────────────────────────────────────
        self.pos      = {t: 0   for t in self.tickers}
        self.entry_px = {t: 0.0 for t in self.tickers}

        # ── PERFORMANCE TRACKING ───────────────────────────────────────────
        self.trade_count = 0

        # ── RETRAINING SCHEDULE ────────────────────────────────────────────
        # Rolling walk-forward: retrain every 63 trading days (~quarterly)
        # Each training window = last 2 trading years of stored features
        # This is the production-grade version of walk_forward_multi_fold()
        self.days_since_retrain = 0
        self.Schedule.On(
            self.DateRules.EveryDay("NVDA"),
            self.TimeRules.AfterMarketOpen("NVDA", 35),
            self.TryRetrain
        )

        self.Log("=" * 60)
        self.Log("ML RIDGE SIGNAL INITIALIZED")
        self.Log("Tickers   : NVDA + MSFT")
        self.Log("Period    : Jan 2020 - Jun 2024  (4.5 years)")
        self.Log("Signal    : Ridge Regression, 10 features, 10am-11am ET")
        self.Log("Costs     : Interactive Brokers brokerage model")
        self.Log("Training  : Quarterly rolling walk-forward, 2-year window")
        self.Log("=" * 60)

    # ──────────────────────────────────────────────────────────────────────
    # CONSOLIDATOR HANDLER FACTORY
    # Creates a separate callback for each ticker
    # Required: Python closures in loops do not capture loop variables
    # ──────────────────────────────────────────────────────────────────────
    def _make_handler(self, ticker):
        def handler(bar):
            self.OnFiveMinBar(bar, ticker)
        return handler

    # ──────────────────────────────────────────────────────────────────────
    # DAILY RESET
    # ORB high/low and bar count reset at start of each new day
    # All positions flattened at end of day (intraday strategy)
    # ──────────────────────────────────────────────────────────────────────
    def OnEndOfDay(self, symbol):
        t = symbol.Value
        if t not in self.tickers:
            return

        # Record yesterday's close for gap feature on next open
        px = self.Securities[self.sym[t]].Price
        if px > 0:
            self.prev_close[t] = px

        # Reset daily ORB state
        self.orb_high[t] = None
        self.orb_low[t]  = None
        self.orb_done[t] = False
        self.bar_idx[t]  = 0

        # Flatten any open positions
        if self.pos[t] != 0:
            self.Liquidate(self.sym[t])
            self.pos[t] = 0

    # ──────────────────────────────────────────────────────────────────────
    # QUARTERLY RETRAINING
    # Called daily but only retrains every 63 trading days
    # This is the rolling walk-forward: model always trained on recent data
    # ──────────────────────────────────────────────────────────────────────
    def TryRetrain(self):
        if self.IsWarmingUp:
            return
        self.days_since_retrain += 1
        if self.days_since_retrain < 63:
            return
        self.days_since_retrain = 0
        for t in self.tickers:
            self._train(t)

    def _train(self, ticker):
        """
        Ridge Regression training on the feature buffer.

        Walk-forward logic:
          Always train on the MOST RECENT 2 years of stored features.
          As time passes, older data falls out, newer data comes in.
          The model continuously adapts to the current regime.

          This is the production-grade version of the 3-fold
          walk_forward_multi_fold() from 03_ML_RIDGE_SIGNAL.py.
          Instead of 3 discrete windows, the window rolls continuously.
        """
        buf = self.feat_buf[ticker]
        if len(buf) < 100:
            return

        # 2 trading years = ~504 trading days x ~78 bars/day = ~39,312 bars
        # Cap at 39312 for rolling window, else use all available data
        train = buf[-39312:] if len(buf) > 39312 else buf

        # Only use rows where forward return has been filled (not NaN)
        rows = [r for r in train if not np.isnan(r.get("fwd_ret", np.nan))]
        if len(rows) < 50:
            return

        X = np.array([[r[f] for f in FEATURE_COLS] for r in rows])
        y = np.array([r["fwd_ret"] for r in rows])

        # Remove any remaining NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 50:
            return

        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)

        mdl = Ridge(alpha=1.0)
        mdl.fit(X_scaled, y)

        self.model[ticker]   = mdl
        self.scaler[ticker]  = sc
        self.trained[ticker] = True

    # ──────────────────────────────────────────────────────────────────────
    # FEATURE COMPUTATION
    # All 10 features computed from indicators and rolling windows
    # No History() call per bar — uses rolling windows for speed
    # ──────────────────────────────────────────────────────────────────────
    def _get_features(self, bar, ticker):
        t     = ticker
        close = bar.Close
        vol   = bar.Volume

        if close <= 0:
            return None

        # Push latest bar into rolling windows
        self.close_win[t].Add(float(close))
        self.volume_win[t].Add(float(vol))

        if not self.close_win[t].IsReady:
            return None

        # Extract arrays (index 0 = most recent)
        n_close = self.close_win[t].Count
        n_vol   = self.volume_win[t].Count
        closes  = np.array([self.close_win[t][i]  for i in range(n_close)])[::-1]
        volumes = np.array([self.volume_win[t][i]  for i in range(n_vol)])[::-1]

        # ── Feature 1: VWAP distance ───────────────────────────────────────
        if not self.vwap_ind[t].IsReady:
            return None
        vwap_val      = float(self.vwap_ind[t].Current.Value)
        vwap_distance = (close - vwap_val) / vwap_val if vwap_val > 0 else 0.0

        # ── Feature 2: RSI ─────────────────────────────────────────────────
        if not self.rsi_ind[t].IsReady:
            return None
        rsi = float(self.rsi_ind[t].Current.Value)

        # ── Feature 3: Volume ratio ────────────────────────────────────────
        avg_vol      = volumes[:-1].mean() if len(volumes) > 1 else float(vol)
        volume_ratio = float(vol) / avg_vol if avg_vol > 0 else 1.0

        # ── Feature 4: ORB breakout ────────────────────────────────────────
        if self.orb_high[t] is not None and self.orb_low[t] is not None:
            orb_mid      = (self.orb_high[t] + self.orb_low[t]) / 2.0
            orb_range    = self.orb_high[t] - self.orb_low[t]
            orb_breakout = (close - orb_mid) / orb_range if orb_range > 0 else 0.0
        else:
            orb_breakout = 0.0

        # ── Feature 5: Momentum (5-bar = 25 minutes) ──────────────────────
        momentum_5 = ((closes[-1] - closes[-6]) / closes[-6]
                      if len(closes) >= 6 and closes[-6] > 0 else 0.0)

        # ── Feature 6: Time of day ─────────────────────────────────────────
        # 0 = market open (9:30am), 1 = market close (4pm)
        # 78 bars per day on 5-min resolution
        time_of_day = self.bar_idx[t] / 78.0

        # ── Feature 7: ATR ratio ───────────────────────────────────────────
        if not self.atr_ind[t].IsReady:
            return None
        atr_ratio = float(self.atr_ind[t].Current.Value) / close if close > 0 else 0.0

        # ── Feature 8: Dollar volume ───────────────────────────────────────
        dv_bar   = float(vol) * close
        dv_hist  = (volumes[:-1] * closes[:-1]).mean() if len(closes) > 1 else dv_bar
        dollar_vol = dv_bar / dv_hist if dv_hist > 0 else 1.0

        # ── Feature 9: Gap size ────────────────────────────────────────────
        # Set once per day at first bar after ORB
        gap_size = self.gap_size[t]

        # ── Feature 10: Return z-score ─────────────────────────────────────
        if len(closes) >= 20:
            rets       = np.diff(closes[-20:]) / closes[-20:-1]
            r_now      = (close - closes[-2]) / closes[-2] if closes[-2] > 0 else 0.0
            r_std      = rets.std()
            ret_zscore = (r_now - rets.mean()) / r_std if r_std > 0 else 0.0
        else:
            ret_zscore = 0.0

        return {
            "vwap_distance": vwap_distance,
            "rsi":           rsi,
            "volume_ratio":  volume_ratio,
            "orb_breakout":  orb_breakout,
            "momentum_5":    momentum_5,
            "time_of_day":   time_of_day,
            "atr_ratio":     atr_ratio,
            "dollar_vol":    dollar_vol,
            "gap_size":      gap_size,
            "ret_zscore":    ret_zscore,
            "close":         close,
            "fwd_ret":       np.nan   # filled in 6 bars later
        }

    # ──────────────────────────────────────────────────────────────────────
    # MAIN SIGNAL LOGIC
    # Fires at the close of every 5-minute bar
    # ──────────────────────────────────────────────────────────────────────
    def OnFiveMinBar(self, bar, ticker):
        if self.IsWarmingUp:
            return

        t = ticker
        self.bar_idx[t] += 1

        # ── ORB: build opening range from first 6 bars ─────────────────
        # Bars 1-6 = 9:30am to 10:00am = do not trade
        if self.bar_idx[t] <= 6:
            if self.orb_high[t] is None:
                self.orb_high[t] = bar.High
                self.orb_low[t]  = bar.Low
                # Gap size: today's open vs yesterday's close
                if self.prev_close[t] > 0:
                    self.gap_size[t] = (bar.Open - self.prev_close[t]) / self.prev_close[t]
            else:
                self.orb_high[t] = max(self.orb_high[t], bar.High)
                self.orb_low[t]  = min(self.orb_low[t],  bar.Low)
            return

        self.orb_done[t] = True

        # ── FILL FORWARD RETURNS in feature buffer ──────────────────────
        # 6 bars = 30 minutes — now we know what happened after bar (i-6)
        buf = self.feat_buf[t]
        if len(buf) >= 6 and not np.isnan(buf[-6]["close"]):
            ref_close = buf[-6]["close"]
            if ref_close > 0:
                buf[-6]["fwd_ret"] = (bar.Close - ref_close) / ref_close

        # ── COMPUTE FEATURES ────────────────────────────────────────────
        feat = self._get_features(bar, t)
        if feat is None:
            return
        buf.append(feat)

        # ── TRAIN ON FIRST AVAILABLE DATA if not yet trained ────────────
        if not self.trained[t] and len(buf) >= 100:
            self._train(t)

        # ── FILL IC ACTUALS — runs on every bar, not just 10am window ──
        # Bug fix: if placed inside the time filter, entries logged at
        # 10:30am-10:55am never get their actuals filled (6 bars later = 11am+)
        ic_buf = self.ic_data[t]
        if len(ic_buf) >= 6:
            ref = ic_buf[-6]
            if np.isnan(ref["actual"]) and not np.isnan(ref["close"]) and ref["close"] > 0:
                ref["actual"] = (bar.Close - ref["close"]) / ref["close"]

        # ── TIME FILTER: 10am–11am ET only ──────────────────────────────
        if self.Time.hour != 10:
            return

        # ── EOD FLATTEN ─────────────────────────────────────────────────
        if self.pos[t] != 0 and self.Time.hour >= 15:
            self.Liquidate(self.sym[t])
            self.pos[t] = 0
            return

        # ── HOLD if already in position ──────────────────────────────────
        if self.pos[t] != 0:
            return

        # ── MODEL PREDICTION ─────────────────────────────────────────────
        if not self.trained[t]:
            return

        fvec = np.array([[feat[f] for f in FEATURE_COLS]])
        if np.isnan(fvec).any():
            return

        fvec_sc    = self.scaler[t].transform(fvec)
        prediction = float(self.model[t].predict(fvec_sc)[0])

        # Store prediction for IC — actual filled by the pre-filter block above
        self.ic_data[t].append({
            "pred":   prediction,
            "actual": np.nan,
            "close":  bar.Close
        })

        # ── CONVICTION THRESHOLD ─────────────────────────────────────────
        # Only trade top 30% strongest predictions
        # Same threshold logic as 03_ML_RIDGE_SIGNAL.py
        recent = [r["pred"] for r in ic_buf[-200:]]
        if len(recent) < 10:
            return
        threshold = float(np.percentile(np.abs(recent), 70))

        # ── ENTRY SIGNAL ─────────────────────────────────────────────────
        # 45% per ticker — allows both NVDA and MSFT simultaneous positions
        # IB fees are per-share with minimums, not % of size — reducing to 25%
        # cut gross returns in half while fees stayed flat. 45% is correct.
        if prediction > threshold:
            self.SetHoldings(self.sym[t], 0.45)
            self.pos[t]      = 1
            self.entry_px[t] = bar.Close
            self.trade_count += 1

        elif prediction < -threshold:
            self.SetHoldings(self.sym[t], -0.45)
            self.pos[t]      = -1
            self.entry_px[t] = bar.Close
            self.trade_count += 1

    def OnData(self, data):
        pass

    # ──────────────────────────────────────────────────────────────────────
    # IC — INFORMATION COEFFICIENT
    # Pearson correlation between predictions and actual 30-min returns
    # IC > 0.05 = useful signal | IC > 0.10 = strong signal
    # ──────────────────────────────────────────────────────────────────────
    def _compute_ic(self, ticker):
        records = [r for r in self.ic_data[ticker]
                   if not np.isnan(r["actual"]) and not np.isnan(r["pred"])]
        if len(records) < 10:
            return np.nan, 0
        preds   = [r["pred"]   for r in records]
        actuals = [r["actual"] for r in records]
        ic, _   = pearsonr(preds, actuals)
        return float(ic), len(records)

    # ──────────────────────────────────────────────────────────────────────
    # PSR — PROBABILISTIC SHARPE RATIO (Lopez de Prado, 2014)
    # PSR = probability the true Sharpe ratio is above zero
    # Accounts for: sample size (T) and non-normality (skewness, fat tails)
    #
    # Formula:
    #   PSR = Phi [ (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + (g4+2)/4 * SR^2) ]
    #   g3 = skewness, g4 = excess kurtosis
    #
    # PSR > 95% = confirmed real
    # PSR > 50% = some evidence
    # PSR < 50% = noise / sample too small
    # ──────────────────────────────────────────────────────────────────────
    def _compute_psr(self, returns):
        import pandas as pd
        arr = np.array([r for r in returns if not np.isnan(r)])
        n   = len(arr)
        if n < 5 or arr.std() == 0:
            return np.nan
        sr_hat   = arr.mean() / arr.std()
        skew     = float(pd.Series(arr).skew())
        exc_kurt = float(pd.Series(arr).kurt())
        denom_sq = 1 - skew * sr_hat + ((exc_kurt + 2) / 4) * sr_hat ** 2
        if denom_sq <= 0:
            return np.nan
        z = sr_hat * np.sqrt(n - 1) / np.sqrt(denom_sq)
        return float(norm.cdf(z))

    # ──────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY — THE FIVE NUMBERS + IC + PSR
    # Read in order: Gross → Costs → Net → IC → PSR
    # ──────────────────────────────────────────────────────────────────────
    def OnEndOfAlgorithm(self):
        final_eq     = self.Portfolio.TotalPortfolioValue
        total_return = (final_eq - 100000) / 100000
        total_profit = self.Portfolio.TotalProfit

        self.Log("")
        self.Log("=" * 60)
        self.Log("BACKTEST COMPLETE — ML RIDGE REGRESSION SIGNAL")
        self.Log("NVDA + MSFT | Jan 2020 - Jun 2024")
        self.Log("=" * 60)
        self.Log("")
        self.Log("=== FIVE NUMBERS: Gross(stats) | Costs(stats) | Net | IC | PSR ===")
        self.Log(f"  NET RETURN : {total_return:+.2%}  |  P&L: ${total_profit:,.2f}  |  Equity: ${final_eq:,.2f}")
        self.Log(f"  TRADES     : {self.trade_count}")
        self.Log("")
        self.Log("  IC — correlation predictions vs actual 30-min returns (>0.05 useful)")

        for t in self.tickers:
            pairs = [(r["pred"], r["actual"]) for r in self.ic_data[t]
                     if not np.isnan(r["pred"]) and not np.isnan(r["actual"])]
            if len(pairs) >= 10:
                preds = [p[0] for p in pairs]
                acts  = [p[1] for p in pairs]
                ic    = float(np.corrcoef(preds, acts)[0, 1])
                verdict = ("STRONG" if ic > 0.10 else "USEFUL" if ic > 0.05 else "LOW")
                self.Debug(f"IC {t}: {ic:.4f}  {verdict}  ({len(pairs)} pairs)")
                self.Log(f"     {t} IC = {ic:.4f}  {verdict}  ({len(pairs)} pairs)")
            else:
                self.Debug(f"IC {t}: n/a  (only {len(pairs)} pairs filled)")
                self.Log(f"     {t} IC = n/a  ({len(pairs)} pairs filled)")

        self.Log("")
        self.Log("  PSR — P(true Sharpe > 0), >95% confirmed, <50% noise")
        for t in self.tickers:
            actuals = [r["actual"] for r in self.ic_data[t] if not np.isnan(r["actual"])]
            psr     = self._compute_psr(actuals)
            psr_str = f"{psr:.1%}" if not np.isnan(psr) else "n/a"
            verdict = ("CONFIRMED" if (not np.isnan(psr) and psr > 0.95) else
                       "SOME EVIDENCE" if (not np.isnan(psr) and psr > 0.50) else "NOISE")
            self.Debug(f"PSR {t}: {psr_str}  {verdict}  ({len(actuals)} obs)")
            self.Log(f"     {t} PSR = {psr_str}  {verdict}  ({len(actuals)} obs)")

        self.Log("")
        self.Log("  DECISION: IC>0.05 + PSR>95% + Gross>0 = CONFIRMED")
        self.Log("            IC>0.05 + PSR<95% + Gross>0 = NEEDS MORE DATA")
        self.Log("            Gross<0 = DEAD")
        self.Log("=" * 60)
# Results: Statistics panel (right) for Gross/Net/Fees/Sharpe/Drawdown
# Log panel (bottom, scroll to end) for IC and PSR per ticker
