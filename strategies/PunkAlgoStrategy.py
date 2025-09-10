# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from functools import reduce

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta


class PunkAlgoStrategy(IStrategy):
    """
    PunkAlgo Strategy - Freqtrade Implementation
    
    Original Pine Script logic converted to Freqtrade
    Multiple signal confirmation system with trend analysis
    """

    # Strategy interface version - allow new iterations of the strategy
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.15,   # 15% ROI
        "30": 0.08,  # 8% ROI after 30 minutes
        "90": 0.03,  # 3% ROI after 1.5 hours
        "180": 0.01  # 1% ROI after 3 hours
    }

    # Optimal stoploss designed for the strategy (optimized from 2% to 3.3%).
    stoploss = -0.033  # 3.3% stop loss (optimized value)

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # ============================================
    # OPTIMIZED PARAMETERS (Best from Optuna Trial 22 - Sharpe: 0.305)
    # ============================================
    
    # Main PunkAlgo Parameters - OPTIMIZED VALUES
    sensitivity = DecimalParameter(0.5, 3.0, default=2.8, decimals=1, space='buy')
    smooth1 = IntParameter(10, 60, default=35, space='buy')  # Optimal: 35
    smooth2 = IntParameter(20, 80, default=75, space='buy')  # Optimal: 75
    ema_period = IntParameter(50, 200, default=100, space='buy')  # Optimal: 100
    
    # EMA Parameters - OPTIMIZED VALUES
    len_ema1 = IntParameter(20, 100, default=65, space='buy')  # Optimal: 65
    len_ema2 = IntParameter(100, 300, default=280, space='buy')  # Optimal: 280
    
    # Wave Trend Parameters - OPTIMIZED VALUES
    wt_chl_len = IntParameter(5, 15, default=8, space='buy')  # Optimal: 8
    wt_avg_len = IntParameter(8, 20, default=14, space='buy')  # Optimal: 14
    wt_oversold = IntParameter(-70, -40, default=-53, space='buy')
    wt_overbought = IntParameter(40, 70, default=53, space='exit')
    
    # RSI Parameters - OPTIMIZED VALUES
    rsi_period = IntParameter(14, 30, default=28, space='buy')  # Optimal: 28
    rsi_ema_period = IntParameter(5, 15, default=13, space='buy')  # Optimal: 13
    rsi_overbought = IntParameter(65, 80, default=70, space='exit')
    rsi_oversold = IntParameter(20, 35, default=30, space='buy')
    
    # SuperTrend Parameters - OPTIMIZED VALUES
    st_period = IntParameter(5, 20, default=17, space='buy')  # Optimal: 17
    st_multiplier = DecimalParameter(1.0, 5.0, default=1.9, decimals=1, space='buy')  # Optimal: 1.9
    
    # Intelligent Trend Parameters - OPTIMIZED VALUES
    trend_multiplier = DecimalParameter(2.0, 8.0, default=7.5, decimals=1, space='buy')  # Optimal: 7.5
    trend_length = IntParameter(20, 50, default=34, space='buy')  # Optimal: 34
    trend_zone_width = DecimalParameter(0.5, 2.0, default=1.2, decimals=1, space='buy')  # Optimal: 1.2
    
    # Momentum Parameters - OPTIMIZED VALUES
    momentum_factor = DecimalParameter(0.3, 1.5, default=1.2, decimals=1, space='buy')  # Optimal: 1.2
    momentum_atr_period = IntParameter(1, 10, default=4, space='buy')  # Optimal: 4
    
    # Stop Loss - OPTIMIZED VALUE
    percent_stop = DecimalParameter(0.5, 5.0, default=3.3, decimals=1, space='buy')  # Optimal: 3.3%
    
    # Trend Filter Parameters
    use_trend_filter = BooleanParameter(default=True, space='buy')
    trend_strength = IntParameter(3, 10, default=5, space='buy')

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pairs will automatically be available in populate_indicators.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        
        # ============================================
        # PUNK ALGO CORE LOGIC
        # ============================================
        
        # Calculate range smoothing components
        dataframe = self.add_smoothrng_indicators(dataframe)
        
        # Calculate main PunkAlgo signals
        dataframe = self.add_punk_signals(dataframe)
        
        # ============================================
        # WAVE TREND OSCILLATOR
        # ============================================
        dataframe = self.add_wave_trend(dataframe)
        
        # ============================================
        # RSI CONDITIONS
        # ============================================
        dataframe = self.add_rsi_conditions(dataframe)
        
        # ============================================
        # EMA TREND (Multiple EMA System - Optimized)
        # ============================================
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.len_ema1.value)  # 65
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.len_ema2.value)  # 280
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)  # 100
        
        # Multi-EMA trend confirmation
        dataframe['ema_bull'] = (
            (dataframe['close'] > dataframe['ema_trend']) &
            (dataframe['ema_fast'] > dataframe['ema_slow'])
        )
        dataframe['ema_bear'] = (
            (dataframe['close'] < dataframe['ema_trend']) &
            (dataframe['ema_fast'] < dataframe['ema_slow'])
        )
        
        # ============================================
        # SUPERTREND (Optimized Parameters)
        # ============================================
        dataframe = self.add_supertrend(dataframe)
        
        # ============================================
        # INTELLIGENT TREND (Optimized Parameters)
        # ============================================
        dataframe = self.add_intelligent_trend(dataframe)
        
        # ============================================
        # MOMENTUM CANDLES (Optimized Parameters)
        # ============================================
        dataframe = self.add_momentum_candles(dataframe)
        
        # ============================================
        # VOLUME INDICATORS
        # ============================================
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # ============================================
        # ADDITIONAL CONFIRMATIONS
        # ============================================
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # ============================================
        # DEBUG: ADD SIMPLE FALLBACK SIGNALS
        # ============================================
        
        # Simple moving average cross as fallback
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=20)
        
        # Simple signals for debugging
        dataframe['simple_bull'] = (
            (dataframe['sma_fast'] > dataframe['sma_slow']) &
            (dataframe['close'] > dataframe['sma_fast']) &
            (dataframe['rsi'] < 70)
        )
        
        dataframe['simple_bear'] = (
            (dataframe['sma_fast'] < dataframe['sma_slow']) &
            (dataframe['close'] < dataframe['sma_fast']) &
            (dataframe['rsi'] > 30)
        )
        
        return dataframe

    def add_intelligent_trend(self, dataframe: DataFrame) -> DataFrame:
        """Add Intelligent Trend with optimized parameters"""
        
        # ATR calculation for intelligent trend
        atr = ta.ATR(dataframe, timeperiod=self.trend_length.value)  # 34
        
        # Simplified intelligent trend calculation
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        
        # Basic upper and lower bands
        upper_band = hl2 + (self.trend_multiplier.value * atr)  # 7.5
        lower_band = hl2 - (self.trend_multiplier.value * atr)
        
        # Intelligent trend line
        dataframe['intelligent_trend'] = lower_band
        dataframe['intelligent_trend_upper'] = upper_band
        
        # Trend direction
        dataframe['it_bull'] = dataframe['close'] > dataframe['intelligent_trend']
        dataframe['it_bear'] = dataframe['close'] < dataframe['intelligent_trend']
        
        # Trend zone
        zone_width = atr * self.trend_zone_width.value  # 1.2
        dataframe['it_zone_upper'] = dataframe['intelligent_trend'] + zone_width
        dataframe['it_zone_lower'] = dataframe['intelligent_trend'] - zone_width
        
        return dataframe

    def add_momentum_candles(self, dataframe: DataFrame) -> DataFrame:
        """Add Momentum Candles with optimized parameters"""
        
        # Momentum SuperTrend with optimized parameters
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        atr = ta.ATR(dataframe, timeperiod=self.momentum_atr_period.value)  # 4
        
        dataframe['momentum_upper'] = hl2 + (self.momentum_factor.value * atr)  # 1.2
        dataframe['momentum_lower'] = hl2 - (self.momentum_factor.value * atr)
        
        # Momentum trend
        dataframe['momentum_trend'] = 1
        dataframe['momentum_line'] = dataframe['momentum_lower']
        
        for i in range(1, len(dataframe)):
            if dataframe['close'].iloc[i] <= dataframe['momentum_line'].iloc[i-1]:
                dataframe.loc[dataframe.index[i], 'momentum_trend'] = -1
                dataframe.loc[dataframe.index[i], 'momentum_line'] = dataframe['momentum_upper'].iloc[i]
            else:
                dataframe.loc[dataframe.index[i], 'momentum_trend'] = 1
                dataframe.loc[dataframe.index[i], 'momentum_line'] = dataframe['momentum_lower'].iloc[i]
        
        # Momentum signals
        dataframe['momentum_bull'] = (
            (dataframe['close'] > dataframe['momentum_line']) & 
            (dataframe['close'].shift(1) <= dataframe['momentum_line'].shift(1))
        )
        dataframe['momentum_bear'] = (
            (dataframe['close'] < dataframe['momentum_line']) & 
            (dataframe['close'].shift(1) >= dataframe['momentum_line'].shift(1))
        )
        
        return dataframe

    def add_supertrend(self, dataframe: DataFrame) -> DataFrame:
        """Add SuperTrend indicator with optimized parameters"""
        
        # SuperTrend calculation with optimized parameters
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        atr = ta.ATR(dataframe, timeperiod=self.st_period.value)  # 14
        
        dataframe['st_upper'] = hl2 + (self.st_multiplier.value * atr)  # 1.6
        dataframe['st_lower'] = hl2 - (self.st_multiplier.value * atr)
        
        # SuperTrend logic
        dataframe['st_trend'] = 1
        dataframe['supertrend'] = dataframe['st_lower']
        
        for i in range(1, len(dataframe)):
            if dataframe['close'].iloc[i] <= dataframe['supertrend'].iloc[i-1]:
                dataframe.loc[dataframe.index[i], 'st_trend'] = -1
                dataframe.loc[dataframe.index[i], 'supertrend'] = dataframe['st_upper'].iloc[i]
            else:
                dataframe.loc[dataframe.index[i], 'st_trend'] = 1
                dataframe.loc[dataframe.index[i], 'supertrend'] = dataframe['st_lower'].iloc[i]
        
        # SuperTrend signals
        dataframe['st_bull'] = (
            (dataframe['close'] > dataframe['supertrend']) & 
            (dataframe['close'].shift(1) <= dataframe['supertrend'].shift(1))
        )
        dataframe['st_bear'] = (
            (dataframe['close'] < dataframe['supertrend']) & 
            (dataframe['close'].shift(1) >= dataframe['supertrend'].shift(1))
        )
        
        return dataframe

    def add_smoothrng_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Add smooth range indicators (PunkAlgo core)"""
        
        # Helper function for smooth range
        def smoothrng(source, length, multiplier):
            wper = length * 2 - 1
            avrng = ta.EMA(np.abs(source.diff()), timeperiod=length)
            return ta.EMA(avrng, timeperiod=wper) * multiplier
        
        # Calculate smooth ranges
        smrng1 = smoothrng(dataframe['close'], 27, 1.5)
        smrng2 = smoothrng(dataframe['close'], 55, self.sensitivity.value)
        dataframe['smrng'] = (smrng1 + smrng2) / 2
        
        # Range filter
        dataframe['filt'] = self.range_filter(dataframe['close'], dataframe['smrng'])
        
        return dataframe

    def range_filter(self, source: pd.Series, rang: pd.Series) -> pd.Series:
        """Range filter implementation"""
        filt = source.copy()
        
        for i in range(1, len(source)):
            if source.iloc[i] > filt.iloc[i-1]:
                filt.iloc[i] = max(filt.iloc[i-1], source.iloc[i] - rang.iloc[i])
            else:
                filt.iloc[i] = min(filt.iloc[i-1], source.iloc[i] + rang.iloc[i])
        
        return filt

    def add_punk_signals(self, dataframe: DataFrame) -> DataFrame:
        """Add main PunkAlgo signals"""
        
        # Calculate up/down conditions
        dataframe['filt_up'] = dataframe['filt'] > dataframe['filt'].shift(1)
        dataframe['filt_down'] = dataframe['filt'] < dataframe['filt'].shift(1)
        
        # Count consecutive periods
        dataframe['up_count'] = 0
        dataframe['dn_count'] = 0
        
        for i in range(1, len(dataframe)):
            if dataframe['filt_up'].iloc[i]:
                dataframe.loc[dataframe.index[i], 'up_count'] = dataframe['up_count'].iloc[i-1] + 1 if dataframe['up_count'].iloc[i-1] > 0 else 1
            elif dataframe['filt_down'].iloc[i]:
                dataframe.loc[dataframe.index[i], 'up_count'] = 0
            else:
                dataframe.loc[dataframe.index[i], 'up_count'] = dataframe['up_count'].iloc[i-1]
                
            if dataframe['filt_down'].iloc[i]:
                dataframe.loc[dataframe.index[i], 'dn_count'] = dataframe['dn_count'].iloc[i-1] + 1 if dataframe['dn_count'].iloc[i-1] > 0 else 1
            elif dataframe['filt_up'].iloc[i]:
                dataframe.loc[dataframe.index[i], 'dn_count'] = 0
            else:
                dataframe.loc[dataframe.index[i], 'dn_count'] = dataframe['dn_count'].iloc[i-1]
        
        # Bull and bear conditions
        dataframe['bull_cond'] = (
            ((dataframe['close'] > dataframe['filt']) & 
             (dataframe['close'] > dataframe['close'].shift(1)) & 
             (dataframe['up_count'] > 0)) |
            ((dataframe['close'] > dataframe['filt']) & 
             (dataframe['close'] < dataframe['close'].shift(1)) & 
             (dataframe['up_count'] > 0))
        )
        
        dataframe['bear_cond'] = (
            ((dataframe['close'] < dataframe['filt']) & 
             (dataframe['close'] < dataframe['close'].shift(1)) & 
             (dataframe['dn_count'] > 0)) |
            ((dataframe['close'] < dataframe['filt']) & 
             (dataframe['close'] > dataframe['close'].shift(1)) & 
             (dataframe['dn_count'] > 0))
        )
        
        # Last condition tracking
        dataframe['last_cond'] = 0
        for i in range(1, len(dataframe)):
            if dataframe['bull_cond'].iloc[i]:
                dataframe.loc[dataframe.index[i], 'last_cond'] = 1
            elif dataframe['bear_cond'].iloc[i]:
                dataframe.loc[dataframe.index[i], 'last_cond'] = -1
            else:
                dataframe.loc[dataframe.index[i], 'last_cond'] = dataframe['last_cond'].iloc[i-1]
        
        # Generate main signals
        dataframe['punk_bull'] = dataframe['bull_cond'] & (dataframe['last_cond'].shift(1) == -1)
        dataframe['punk_bear'] = dataframe['bear_cond'] & (dataframe['last_cond'].shift(1) == 1)
        
        # Current trigger (position)
        dataframe['punk_trigger'] = dataframe['last_cond'] > 0
        
        return dataframe

    def add_wave_trend(self, dataframe: DataFrame) -> DataFrame:
        """Add Wave Trend oscillator"""
        
        # HLC3
        hlc3 = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        
        # Wave Trend calculation
        esa = ta.EMA(hlc3, timeperiod=self.wt_chl_len.value)
        d = ta.EMA(np.abs(hlc3 - esa), timeperiod=self.wt_chl_len.value)
        ci = (hlc3 - esa) / (0.015 * d)
        dataframe['wt1'] = ta.EMA(ci, timeperiod=self.wt_avg_len.value)
        dataframe['wt2'] = ta.SMA(dataframe['wt1'], timeperiod=3)
        
        # Wave Trend signals
        dataframe['wt_cross_up'] = (
            (dataframe['wt1'] > dataframe['wt2']) & 
            (dataframe['wt1'].shift(1) <= dataframe['wt2'].shift(1)) & 
            (dataframe['wt2'] <= self.wt_oversold.value)
        )
        
        dataframe['wt_cross_down'] = (
            (dataframe['wt1'] < dataframe['wt2']) & 
            (dataframe['wt1'].shift(1) >= dataframe['wt2'].shift(1)) & 
            (dataframe['wt2'] >= self.wt_overbought.value)
        )
        
        return dataframe

    def add_rsi_conditions(self, dataframe: DataFrame) -> DataFrame:
        """Add RSI conditions"""
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['rsi_ema'] = ta.EMA(dataframe['rsi'], timeperiod=self.rsi_ema_period.value)
        
        dataframe['rsi_ob'] = (dataframe['rsi'] > self.rsi_overbought.value) & (dataframe['rsi'] > dataframe['rsi_ema'])
        dataframe['rsi_os'] = (dataframe['rsi'] < self.rsi_oversold.value) & (dataframe['rsi'] < dataframe['rsi_ema'])
        
        return dataframe

    def add_supertrend(self, dataframe: DataFrame) -> DataFrame:
        """Add SuperTrend indicator with optimized parameters"""
        
        # SuperTrend calculation with optimized parameters
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        atr = ta.ATR(dataframe, timeperiod=self.st_period.value)  # 14
        
        dataframe['st_upper'] = hl2 + (self.st_multiplier.value * atr)  # 1.6
        dataframe['st_lower'] = hl2 - (self.st_multiplier.value * atr)
        
        # SuperTrend logic
        dataframe['st_trend'] = 1
        dataframe['supertrend'] = dataframe['st_lower']
        
        for i in range(1, len(dataframe)):
            if dataframe['close'].iloc[i] <= dataframe['supertrend'].iloc[i-1]:
                dataframe.loc[dataframe.index[i], 'st_trend'] = -1
                dataframe.loc[dataframe.index[i], 'supertrend'] = dataframe['st_upper'].iloc[i]
            else:
                dataframe.loc[dataframe.index[i], 'st_trend'] = 1
                dataframe.loc[dataframe.index[i], 'supertrend'] = dataframe['st_lower'].iloc[i]
        
        # SuperTrend signals
        dataframe['st_bull'] = (
            (dataframe['close'] > dataframe['supertrend']) & 
            (dataframe['close'].shift(1) <= dataframe['supertrend'].shift(1))
        )
        dataframe['st_bear'] = (
            (dataframe['close'] < dataframe['supertrend']) & 
            (dataframe['close'].shift(1) >= dataframe['supertrend'].shift(1))
        )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        SIMPLIFIED ENTRY CONDITIONS - Less restrictive for more trades
        """
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        
        # ============================================
        # LONG ENTRY CONDITIONS
        # ============================================
        
        long_conditions = []
        
        # 1. Main PunkAlgo Bull Signal
        long_conditions.append(dataframe['punk_bull'])
        
        # 2. EMA Trend Filter (if enabled)
        if self.use_trend_filter.value:
            long_conditions.append(dataframe['ema_bull'])
        
        # 3. Not RSI Overbought
        long_conditions.append(~dataframe['rsi_ob'])
        
        # 4. Volume confirmation
        long_conditions.append(dataframe['volume_ratio'] > 0.8)
        
        # 5. Enhanced confirmations with optimized indicators
        additional_long = (
            # Wave Trend oversold cross
            dataframe['wt_cross_up'] |
            # SuperTrend bull signal with EMA support
            (dataframe['st_bull'] & dataframe['ema_bull']) |
            # Intelligent Trend support
            (dataframe['it_bull'] & dataframe['ema_bull']) |
            # Momentum confirmation
            (dataframe['momentum_bull'] & dataframe['ema_bull']) |
            # Strong momentum with trend
            (dataframe['punk_trigger'] & dataframe['ema_bull'] & (dataframe['close'] > dataframe['close'].shift(3)))
        )
        long_conditions.append(additional_long)
        
        # Combine all long conditions
        dataframe.loc[
            reduce(lambda x, y: x & y, long_conditions),
            'enter_long'] = 1

        # ============================================
        # SHORT ENTRY CONDITIONS  
        # ============================================
        
        short_conditions = []
        
        # 1. Main PunkAlgo Bear Signal
        short_conditions.append(dataframe['punk_bear'])
        
        # 2. EMA Trend Filter (enhanced with multiple EMA)
        if self.use_trend_filter.value:
            short_conditions.append(dataframe['ema_bear'])
        
        # 3. Not RSI Oversold
        short_conditions.append(~dataframe['rsi_os'])
        
        # 4. Volume confirmation
        short_conditions.append(dataframe['volume_ratio'] > 0.8)
        
        # 5. Enhanced confirmations with optimized indicators
        additional_short = (
            # Wave Trend overbought cross
            dataframe['wt_cross_down'] |
            # SuperTrend bear signal with EMA support
            (dataframe['st_bear'] & dataframe['ema_bear']) |
            # Intelligent Trend resistance
            (dataframe['it_bear'] & dataframe['ema_bear']) |
            # Momentum confirmation
            (dataframe['momentum_bear'] & dataframe['ema_bear']) |
            # Strong downward momentum
            (~dataframe['punk_trigger'] & dataframe['ema_bear'] & (dataframe['close'] < dataframe['close'].shift(3)))
        )
        short_conditions.append(additional_short)
        
        # Combine all short conditions
        dataframe.loc[
            reduce(lambda x, y: x & y, short_conditions),
            'enter_short'] = 1

        return dataframe

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        
        # ============================================
        # LONG EXIT CONDITIONS
        # ============================================
        
        long_exit_conditions = []
        
        # 1. Main bear signal
        long_exit_conditions.append(dataframe['punk_bear'])
        
        # 2. Wave Trend overbought cross
        long_exit_conditions.append(dataframe['wt_cross_down'])
        
        # 3. RSI Overbought
        long_exit_conditions.append(dataframe['rsi_ob'])
        
        # 4. SuperTrend bear signal
        long_exit_conditions.append(dataframe['st_bear'])
        
        # 5. EMA trend broken
        long_exit_conditions.append(
            ~dataframe['ema_bull'] & 
            (dataframe['close'] < dataframe['ema_trend'] * 0.99)  # 1% below EMA
        )
        
        # Combine long exit conditions (any condition triggers exit)
        dataframe.loc[
            reduce(lambda x, y: x | y, long_exit_conditions),
            'exit_long'] = 1

        # ============================================
        # SHORT EXIT CONDITIONS
        # ============================================
        
        short_exit_conditions = []
        
        # 1. Main bull signal
        short_exit_conditions.append(dataframe['punk_bull'])
        
        # 2. Wave Trend oversold cross
        short_exit_conditions.append(dataframe['wt_cross_up'])
        
        # 3. RSI Oversold
        short_exit_conditions.append(dataframe['rsi_os'])
        
        # 4. SuperTrend bull signal
        short_exit_conditions.append(dataframe['st_bull'])
        
        # 5. EMA trend reversal
        short_exit_conditions.append(
            dataframe['ema_bull'] & 
            (dataframe['close'] > dataframe['ema_trend'] * 1.01)  # 1% above EMA
        )
        
        # Combine short exit conditions (any condition triggers exit)
        dataframe.loc[
            reduce(lambda x, y: x | y, short_exit_conditions),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic with optimized 3.3% base and trailing features
        """
        
        # Get the dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe.empty:
            return self.stoploss  # -0.033 (3.3%)
        
        # Get current candle
        current_candle = dataframe.iloc[-1]
        
        # Enhanced ATR-based stop with optimization insights
        atr_stop_distance = current_candle['atr'] / current_rate
        
        # Progressive profit-based stops (optimized based on 3.3% base)
        if current_profit > 0.15:  # If profit > 15% - secure most gains
            return max(self.stoploss, -current_profit + 0.03)  # Trail with 3% buffer
        elif current_profit > 0.10:  # If profit > 10% - moderate trail
            return max(self.stoploss, -current_profit + 0.02)  # Trail with 2% buffer
        elif current_profit > 0.05:  # If profit > 5% - light trail
            return max(self.stoploss, -current_profit + 0.015) # Trail with 1.5% buffer
        elif current_profit > 0.02:  # If profit > 2% - minimal trail
            return max(self.stoploss, -0.02)  # Move to 2% stop
        else:
            # Use optimized base stop with ATR consideration
            dynamic_stop = max(self.stoploss, -atr_stop_distance * 1.5)
            return dynamic_stop

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing functions (e.g. current_time) should be preferred over technical indicators.
        """
        
        # Get current market data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe.empty:
            return False
        
        current_candle = dataframe.iloc[-1]
        
        # Don't enter during low volume periods
        if current_candle['volume_ratio'] < 0.5:
            return False
        
        # Don't enter if spread is too wide (if available)
        if hasattr(current_candle, 'spread'):
            if current_candle['spread'] > 0.001:  # 0.1% spread
                return False
        
        # Additional safety checks
        if side == 'long':
            # Don't buy at resistance levels
            if current_candle['close'] > current_candle['bb_upper']:
                return False
        elif side == 'short':
            # Don't sell at support levels  
            if current_candle['close'] < current_candle['bb_lower']:
                return False
        
        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade.
        """
        
        # Conservative leverage based on volatility
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe.empty:
            return 1.0
        
        current_candle = dataframe.iloc[-1]
        
        # Calculate volatility-based leverage
        atr_pct = current_candle['atr'] / current_candle['close']
        
        if atr_pct > 0.03:  # High volatility (>3%)
            return min(2.0, max_leverage)
        elif atr_pct > 0.02:  # Medium volatility (2-3%)
            return min(3.0, max_leverage)
        else:  # Low volatility (<2%)
            return min(5.0, max_leverage)