# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class LimitedEntryDCAStrategy(IStrategy):
    """
    Limited Entry DCA Strategy
    
    This strategy implements a "Limited Entry" philosophy where:
    1. Initial entries are high-conviction signals based on trend + oversold conditions
    2. Position adjustments (DCA) are limited and controlled
    3. Risk management is paramount with clear stop-loss and take-profit levels
    
    The strategy looks for:
    - Strong uptrend (EMA alignment)
    - Oversold conditions (RSI < 30)
    - Price near lower Bollinger Band
    - Volume confirmation
    
    Position adjustments occur only when:
    - Price drops further after initial entry
    - Maximum 2 additional entries per trade
    - Each additional entry is smaller than the previous
    """

    INTERFACE_VERSION = 3

    # Strategy parameters - easily tunable
    
    # Trend detection
    fast_ema_period = IntParameter(12, 21, default=12, space="buy")
    slow_ema_period = IntParameter(26, 50, default=26, space="buy")
    trend_ema_period = IntParameter(50, 200, default=100, space="buy")
    
    # RSI parameters
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_oversold = IntParameter(20, 35, default=30, space="buy")
    rsi_overbought = IntParameter(65, 80, default=70, space="sell")
    
    # Bollinger Bands
    bb_period = IntParameter(15, 25, default=20, space="buy")
    bb_std = DecimalParameter(1.5, 2.5, default=2.0, space="buy")
    
    # Volume confirmation
    volume_factor = DecimalParameter(1.0, 2.0, default=1.2, space="buy")
    
    # Position adjustment (DCA) parameters
    dca_trigger_pct = DecimalParameter(2.0, 8.0, default=4.0, space="buy")  # Price drop % to trigger DCA
    dca_size_factor = DecimalParameter(0.5, 1.0, default=0.7, space="buy")  # Each DCA is 70% of previous
    max_dca_entries = IntParameter(1, 3, default=2, space="buy")  # Maximum additional entries
    
    # Exit parameters
    take_profit_pct = DecimalParameter(5.0, 20.0, default=12.0, space="sell")
    trailing_stop_positive = DecimalParameter(0.01, 0.05, default=0.02, space="sell")
    trailing_stop_positive_offset = DecimalParameter(0.03, 0.08, default=0.05, space="sell")

    # ROI table - Conservative approach for limited entries
    minimal_roi = {
        "0": 0.15,   # 15% profit any time
        "30": 0.08,  # 8% after 30 minutes
        "60": 0.05,  # 5% after 1 hour
        "120": 0.02  # 2% after 2 hours
    }

    # Stoploss
    stoploss = -0.08  # 8% stop loss

    # Trailing stop
    trailing_stop = True
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Position adjustment settings (will be enforced by config)
    position_adjustment_enable = True

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all indicators needed for the strategy.
        
        This method calculates:
        - EMAs for trend detection
        - RSI for momentum/oversold conditions  
        - Bollinger Bands for price boundaries
        - Volume indicators for confirmation
        """
        
        # EMA indicators for trend detection
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.fast_ema_period.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.slow_ema_period.value)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=self.trend_ema_period.value)
        
        # RSI for momentum
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        
        # Bollinger Bands for price boundaries
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), 
                                          window=self.bb_period.value, 
                                          stds=self.bb_std.value)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # Volume indicators
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        
        # Additional trend confirmation
        dataframe['trend_bullish'] = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) & 
            (dataframe['ema_slow'] > dataframe['ema_trend']) &
            (dataframe['close'] > dataframe['ema_trend'])
        )
        
        # Price action indicators
        dataframe['price_above_bb_lower'] = dataframe['close'] > dataframe['bb_lower']
        dataframe['close_to_bb_lower'] = dataframe['bb_percent'] < 0.2  # Close to lower band
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate the entry trend for initial positions.
        
        High-conviction entry conditions:
        1. Strong bullish trend (EMA alignment)
        2. RSI oversold (< 30)
        3. Price near lower Bollinger Band
        4. Volume confirmation
        """
        
        conditions = []
        
        # Primary trend must be bullish
        conditions.append(dataframe['trend_bullish'])
        
        # RSI oversold condition
        conditions.append(dataframe['rsi'] < self.rsi_oversold.value)
        
        # Price near lower Bollinger Band (oversold bounce setup)
        conditions.append(dataframe['close_to_bb_lower'])
        conditions.append(dataframe['price_above_bb_lower'])  # But not below it
        
        # Volume confirmation - higher than average
        conditions.append(dataframe['volume'] > (dataframe['volume_sma'] * self.volume_factor.value))
        
        # Ensure we're not at extreme low (some safety)
        conditions.append(dataframe['close'] > dataframe['bb_lower'] * 1.001)  # At least 0.1% above lower band
        
        # Additional momentum filter - RSI showing some recovery
        conditions.append(dataframe['rsi'] > dataframe['rsi'].shift(1))  # RSI improving
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate the exit trend.
        
        Exit conditions:
        1. RSI overbought
        2. Price above upper Bollinger Band
        3. Bearish trend reversal
        """
        
        conditions = []
        
        # RSI overbought
        conditions.append(dataframe['rsi'] > self.rsi_overbought.value)
        
        # Price extended above upper Bollinger Band
        conditions.append(dataframe['close'] > dataframe['bb_upper'])
        
        # Trend showing weakness
        conditions.append(
            (dataframe['ema_fast'] < dataframe['ema_slow']) |
            (dataframe['close'] < dataframe['ema_fast'])
        )
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe

    def adjust_trade_position(self, trade, current_time: datetime, current_rate: float,
                            current_profit: float, min_stake: Optional[float],
                            max_stake: float, current_entry_rate: float, 
                            current_exit_rate: float, current_entry_profit: float,
                            current_exit_profit: float, **kwargs) -> Optional[float]:
        """
        Adjust trade position - implements limited DCA strategy.
        
        This method adds to positions when:
        1. Current profit is negative (price has dropped)
        2. Price has dropped by the DCA trigger percentage
        3. We haven't exceeded maximum DCA entries
        4. We have sufficient stake available
        
        Each subsequent entry is smaller than the previous to limit risk.
        """
        
        # Get the number of entries already made
        filled_entries = trade.nr_of_successful_entries
        
        # Check if we've reached maximum entries
        if filled_entries >= (1 + self.max_dca_entries.value):
            return None
            
        # Only DCA if we're at a loss
        if current_profit >= 0:
            return None
            
        # Calculate price drop percentage from average entry price
        price_drop_pct = abs(current_profit) * 100
        
        # Check if price drop meets our DCA trigger
        if price_drop_pct < self.dca_trigger_pct.value:
            return None
            
        # Calculate DCA amount - each entry gets smaller
        try:
            # Get initial stake amount from the first entry
            initial_stake = trade.stake_amount
            
            # Calculate DCA stake (progressively smaller)
            dca_multiplier = self.dca_size_factor.value ** (filled_entries - 1)
            dca_stake = initial_stake * dca_multiplier
            
            # Ensure we don't exceed max_stake and have minimum stake
            if min_stake and dca_stake < min_stake:
                dca_stake = min_stake
                
            if dca_stake > max_stake:
                dca_stake = max_stake
                
            # Additional safety check - don't DCA if it would make position too large
            total_stake_after_dca = trade.stake_amount + dca_stake
            if total_stake_after_dca > max_stake * 2:  # Conservative limit
                return None
                
            self.dp.log_info(f"DCA Entry #{filled_entries} for {trade.pair}: "
                           f"Price drop: {price_drop_pct:.2f}%, "
                           f"Adding stake: {dca_stake:.2f}")
            
            return dca_stake
            
        except Exception as e:
            self.dp.log_error(f"Error in adjust_trade_position: {e}")
            return None

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Custom exit logic - provides additional exit conditions beyond ROI/stoploss.
        """
        
        # Emergency exit if we have too many DCA entries and still losing
        if trade.nr_of_successful_entries > 2 and current_profit < -0.15:  # -15%
            return "emergency_exit_deep_loss"
            
        # Take profit at significant gains to lock in profits
        if current_profit > 0.20:  # 20% profit
            return "take_profit_20pct"
            
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                          side: str, **kwargs) -> bool:
        """
        Confirm trade entry - additional validation before entering trade.
        """
        
        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe.empty:
            return False
            
        # Get the latest data point
        last_candle = dataframe.iloc[-1]
        
        # Additional safety checks for entry
        
        # Don't enter if RSI is too extreme
        if last_candle['rsi'] < 15:  # Too oversold, might continue falling
            return False
            
        # Don't enter if we're in a strong downtrend
        if (last_candle['ema_fast'] < last_candle['ema_slow'] * 0.98):  # 2% below
            return False
            
        # Confirm volume is still adequate
        if last_candle['volume'] < last_candle['volume_sma'] * 0.8:  # Volume too low
            return False
            
        return True

# Helper function for reduce (since it's not imported by default in newer Python versions)
from functools import reduce