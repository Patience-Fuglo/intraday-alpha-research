# ==============================================================================
# QUANTCONNECT VWAP + RSI Mean Reversion v4 - FASTER RSI + ADJUSTED STOP
# Paste this entire file into QuantConnect main.py - NO emojis, clean ASCII
#
# What changed from v3:
#   RSI period 14 -> 7: faster signal, reacts to current 35-min window not 70-min
#   Stop loss 0.5% -> 0.75%: 0.5% caused whipsawing (orders went UP 779->853)
#   v3 lesson: stop tighter than intraday noise = more churn, not less loss
#   Target: win rate recovers above 50%, P/L ratio holds above 0.95
# ==============================================================================

from AlgorithmImports import *


class VWAPRSIMeanReversion(QCAlgorithm):

    def Initialize(self):

        # --- BACKTEST WINDOW ---
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 6, 1)
        self.SetCash(100000)

        # --- TICKER ---
        self.ticker = "AAPL"
        equity = self.AddEquity(self.ticker, Resolution.Minute)
        equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
        self.symbol = equity.Symbol

        # --- COST MODEL ---
        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        # --- INDICATORS (5-min resolution) ---
        # Wilder RSI - period 7 for faster intraday response (35-min lookback on 5-min bars)
        self.rsi = self.RSI(self.symbol, 7, MovingAverageType.Wilders, Resolution.Minute)
        # VWAP - resets daily automatically
        self.vwap = self.VWAP(self.symbol)

        # --- REGIME INDICATORS (daily resolution) ---
        # ATR: measures daily volatility - high ATR = volatile = stay flat
        self.atr = self.ATR(self.symbol, 14, MovingAverageType.Wilders, Resolution.Daily)
        # 200-day SMA: measures trend direction
        self.sma_slow = self.SMA(self.symbol, 200, Resolution.Daily)
        # 20-day SMA: short-term trend
        self.sma_fast = self.SMA(self.symbol, 20, Resolution.Daily)

        # --- SIGNAL PARAMETERS ---
        self.long_rsi       = 25
        self.short_rsi      = 75
        self.distance_pct   = 0.003    # 0.30% min distance from VWAP
        self.stop_loss_pct  = 0.0075   # 0.75% stop loss - midpoint, v3 0.5% caused whipsawing
        self.vol_threshold  = 0.025    # ATR/Price > 2.5% = high vol, stay flat

        # --- 5-MINUTE BAR CONSOLIDATION ---
        self.Consolidate(self.symbol, timedelta(minutes=5), self.OnFiveMinuteBar)

        # --- STATE ---
        self.position    = 0
        self.entry_price = 0.0
        self.trade_count = 0
        self.skipped_vol = 0   # count how many signals filtered by vol regime
        self.skipped_trend = 0 # count how many signals filtered by trend regime

        self.Log("Initialized: VWAP RSI Mean Reversion v4 - RSI(7) + Stop 0.75%")

    # --------------------------------------------------------------------------
    # REGIME CHECK - called before every trade entry
    # Returns True if conditions are safe to trade, False to stay flat
    # Equivalent to apply_regime_filter() in 03_INTERMEDIATE_ADVANCED_FIXES.py
    # --------------------------------------------------------------------------
    def IsGoodRegime(self):
        # Need indicators warmed up
        if not self.atr.IsReady or not self.sma_slow.IsReady:
            return False

        price = self.Securities[self.symbol].Price
        if price <= 0:
            return False

        # Volatility regime: skip if ATR/Price > threshold (market too wild)
        # This filters out COVID crash, flash crashes, earnings gaps
        vol_ratio = self.atr.Current.Value / price
        if vol_ratio > self.vol_threshold:
            self.skipped_vol += 1
            return False

        # Trend regime: skip if price is far from 200 SMA (strong trend)
        # Mean reversion only works in ranging/sideways markets
        sma200 = self.sma_slow.Current.Value
        trend_distance = abs(price - sma200) / sma200
        if trend_distance > 0.15:  # price >15% away from 200 SMA = trending
            self.skipped_trend += 1
            return False

        return True

    # --------------------------------------------------------------------------
    # MAIN LOGIC - fires at close of every 5-minute bar
    # --------------------------------------------------------------------------
    def OnFiveMinuteBar(self, bar):

        if not self.rsi.IsReady or not self.vwap.IsReady:
            return

        close    = bar.Close
        vwap_val = self.vwap.Current.Value
        rsi_val  = self.rsi.Current.Value
        distance = (close - vwap_val) / vwap_val

        # --- STOP LOSS CHECK (always runs, even in bad regime) ---
        if self.position == 1 and self.entry_price > 0:
            loss = (close - self.entry_price) / self.entry_price
            if loss < -self.stop_loss_pct:
                self.Liquidate(self.symbol)
                self.position = 0
                self.Log("STOP LOSS LONG  | loss=" + str(round(loss*100,2)) + "%")
                return

        if self.position == -1 and self.entry_price > 0:
            loss = (self.entry_price - close) / self.entry_price
            if loss < -self.stop_loss_pct:
                self.Liquidate(self.symbol)
                self.position = 0
                self.Log("STOP LOSS SHORT | loss=" + str(round(loss*100,2)) + "%")
                return

        # --- VWAP EXIT (reversion complete) ---
        if self.position == 1 and close > vwap_val:
            self.Liquidate(self.symbol)
            self.position = 0
            self.Log("EXIT LONG  | close=" + str(round(close,2)) + " vwap=" + str(round(vwap_val,2)))
            return

        if self.position == -1 and close < vwap_val:
            self.Liquidate(self.symbol)
            self.position = 0
            self.Log("EXIT SHORT | close=" + str(round(close,2)) + " vwap=" + str(round(vwap_val,2)))
            return

        # --- ENTRY (only in good regime) ---
        if self.position != 0:
            return

        # Signal conditions
        long_signal  = (close < vwap_val and
                        rsi_val < self.long_rsi and
                        distance < -self.distance_pct)

        short_signal = (close > vwap_val and
                        rsi_val > self.short_rsi and
                        distance > self.distance_pct)

        if not (long_signal or short_signal):
            return

        # Regime gate: only enter if market is calm and ranging
        if not self.IsGoodRegime():
            return

        if long_signal:
            self.SetHoldings(self.symbol, 0.95)
            self.position    = 1
            self.entry_price = close
            self.trade_count += 1
            self.Log("ENTER LONG  | rsi=" + str(round(rsi_val,1)) + " dist=" + str(round(distance,4)))

        elif short_signal:
            self.SetHoldings(self.symbol, -0.50)
            self.position    = -1
            self.entry_price = close
            self.trade_count += 1
            self.Log("ENTER SHORT | rsi=" + str(round(rsi_val,1)) + " dist=" + str(round(distance,4)))

    # --------------------------------------------------------------------------
    # FLATTEN at end of day
    # --------------------------------------------------------------------------
    def OnEndOfDay(self, symbol):
        if symbol == self.symbol and self.position != 0:
            self.Liquidate(self.symbol)
            self.position = 0
            self.Log("EOD FLATTEN")

    def OnData(self, data):
        pass

    # --------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------
    def OnEndOfAlgorithm(self):
        self.Log("=== BACKTEST COMPLETE v4 ===")
        self.Log("Total trades    : " + str(self.trade_count))
        self.Log("Skipped vol     : " + str(self.skipped_vol))
        self.Log("Skipped trend   : " + str(self.skipped_trend))
        self.Log("Final equity    : $" + str(round(self.Portfolio.TotalPortfolioValue, 2)))
        self.Log("Net P&L         : $" + str(round(self.Portfolio.TotalProfit, 2)))

# ==============================================================================
# WHAT TO COMPARE vs v1
# ==============================================================================
# v1 result (no filter):  Return -14.40%  PSR 0.107%  Fees $4,667
# v2 target (with filter): fewer trades, lower fees, higher PSR
#
# Key lesson from 03_INTERMEDIATE_ADVANCED_FIXES.py:
#   Mean reversion only works in low-vol, ranging markets.
#   The regime filter turns the strategy off during crises and trends.
#   Fewer trades + higher quality = better Sharpe.
#
# NEXT: if PSR improves above 50%, add a second ticker (MSFT)
#       and check if the improvement holds out-of-sample.
# ==============================================================================
