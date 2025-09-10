import logging
from typing import Optional, Union, Tuple, Dict, List, Any

from freqtrade.persistence import Trade
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter,
                                BooleanParameter, CategoricalParameter,
                                informative, merge_informative_pair)
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series
from freqtrade.persistence import Order, CustomDataWrapper
import datetime
from datetime import datetime as dt
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
import math
from sqlalchemy import desc
from functools import reduce


class FiboSmaEmaRsi(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True

    # Updated Minimal ROI based on provided parameters
    minimal_roi = {
        "0": 0.08,    # Take profit after 8% gain
        "60": 0.05,   # Take profit after 5% gain if trade has been open for 60 minutes
        "120": 0.03,  # Take profit after 3% gain if trade has been open for 120 minutes
        "240": 0.01   # Take profit after 1% gain if trade has been open for 240 minutes
    }

    # Stoploss
    stoploss = -0.097
    # Order types
    order_types = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    use_exit_signal = True
    exit_profit_only = False

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.008

    position_adjustment_enable = False
    debug_mode = False
    max_stake_amount = 50000  # Default value (in stake currency)
    minimum_stake_amount = 10  # Default minimum stake amount (in stake currency)

    # Leverage
    leverage_value = IntParameter(1, 20, default=19, space='buy', optimize=True)

    # Common parameters - Keep only what we need for the new strategy
    use_dynamic_sizing = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Timeframe parameters
    timeframe_parameter = CategoricalParameter(['5m', '15m', '1h', '4h', '1d', '1w'], default='5m', space='buy', optimize=True)
    
    # Multi-timeframe parameters
    use_multi_timeframe = BooleanParameter(default=True, space='buy', optimize=True)
    higher_timeframe_weight = DecimalParameter(0.1, 0.9, default=0.5, space='buy', optimize=True)
    
    # Moving Average Parameters (from Pine Script)
    ma_type1 = CategoricalParameter(['EMA', 'SMA'], default='SMA', space='buy', optimize=False)
    ma_type2 = CategoricalParameter(['EMA', 'SMA'], default='SMA', space='buy', optimize=False)
    ma_type3 = CategoricalParameter(['EMA', 'SMA'], default='EMA', space='buy', optimize=False)
    len_fast = IntParameter(5, 50, default=10, space='buy', optimize=True)
    len_slow = IntParameter(10, 100, default=30, space='buy', optimize=True)
    len_tend = IntParameter(100, 300, default=200, space='buy', optimize=True)
    show_cross = BooleanParameter(default=True, space='buy', optimize=True)
    
    # RSI Parameters (from Pine Script)
    rsi_length = IntParameter(5, 25, default=8, space='buy', optimize=True)
    rsi_buy_lower = IntParameter(20, 40, default=25, space='buy', optimize=True)
    rsi_buy_upper = IntParameter(30, 50, default=40, space='buy', optimize=True)
    rsi_sell_lower = IntParameter(50, 70, default=60, space='sell', optimize=True)
    rsi_sell_upper = IntParameter(60, 80, default=75, space='sell', optimize=True)
    
    # Risk Management Parameters (from Pine Script)
    sl_percent = DecimalParameter(0.5, 3.0, default=1.0, space='sell', optimize=True)
    rr_ratio = DecimalParameter(1.0, 5.0, default=3.0, space='sell', optimize=True)
    show_projections = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Fibonacci Parameters (from Pine Script)
    show_fib = BooleanParameter(default=True, space='buy', optimize=False)
    show_fib_labels = BooleanParameter(default=True, space='buy', optimize=False)
    use_fib_for_entry = BooleanParameter(default=True, space='buy', optimize=False)
    fib_lookback_candles = IntParameter(10, 100, default=50, space='buy', optimize=True)
    fib_line_length = IntParameter(10, 100, default=50, space='buy', optimize=False)
    
    # Stake divider
    stake_divider = IntParameter(1, 10, default=1, space='buy', optimize=True)
    
    # Entry condition switches
    use_ma_crossover = BooleanParameter(default=True, space='buy', optimize=True)
    use_rsi_filter = BooleanParameter(default=True, space='buy', optimize=True)
    use_candle_pattern = BooleanParameter(default=True, space='buy', optimize=True)
    use_trend_filter = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Exit condition switches
    use_ma_crossover_exit = BooleanParameter(default=True, space='sell', optimize=True)
    use_rsi_exit = BooleanParameter(default=True, space='sell', optimize=True)
    use_trailing_sl = BooleanParameter(default=False, space='sell', optimize=True)

    @property
    def timeframe(self):
        return self.timeframe_parameter.value

    def __init__(self, config: dict):
        super().__init__(config)
        # Set max stake amount from config if provided
        if "max_stake_amount" in config:
            self.max_stake_amount = config["max_stake_amount"]
        if "minimum_stake_amount" in config:
            self.minimum_stake_amount = config["minimum_stake_amount"]
        self.set_debug_mode(config.get('debug_mode', False))
        
        # Log that we're using simplified strategy during hyperopt/backtest
        if hasattr(self, 'dp') and hasattr(self.dp, 'runmode') and self.dp.runmode.value in ('hyperopt', 'backtest'):
            logging.info("Using simplified strategy during hyperopt/backtest")

    @property
    def protections(self):
        return [
            {
                # Lock pair after selling for 5 candles
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                # Stop trading if max drawdown threshold is reached
                "method": "MaxDrawdown",
                "lookback_period_candles": 24,  # Evaluate over the last 24 candles
                "trade_limit": 20,              # Consider at least 20 trades for this protection
                "stop_duration_candles": 4,     # Stop trading for 4 candles
                "max_allowed_drawdown": 0.2     # Maximum allowed drawdown is 20%
            },
            {
                # Guard against too many stoploss hits
                "method": "StoplossGuard",
                "lookback_period_candles": 24,  # Evaluate over the last 24 candles
                "trade_limit": 4,               # Consider at least 4 trades for this protection
                "stop_duration_candles": 2,     # Stop trading for 2 candles
                "only_per_pair": False          # Apply globally, not just per pair
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_value.value

    # Calculate custom stake amount
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        custom_stake = proposed_stake
        
        # Apply stake divider
        adjusted_stake = custom_stake / self.stake_divider.value
        
        # Calculate the maximum allowed stake considering leverage
        max_allowed_stake = self.max_stake_amount / leverage if leverage > 0 else self.max_stake_amount
        
        # Apply maximum stake limit with leverage consideration
        adjusted_stake = min(adjusted_stake, max_allowed_stake)
        
        # Use our own minimum stake amount if it's higher than the exchange's minimum
        min_stake_to_use = max(self.minimum_stake_amount, min_stake) if min_stake is not None else self.minimum_stake_amount
        
        # Ensure we don't go below minimum allowed stake
        adjusted_stake = max(adjusted_stake, min_stake_to_use)
        
        # Ensure we don't exceed maximum allowed stake from config
        adjusted_stake = min(adjusted_stake, max_stake)
        
        if self.debug_mode:
            logging.info(f"Custom stake calculation for {pair}:")
            logging.info(f"- Initial proposed stake: {proposed_stake}")
            logging.info(f"- After stake divider ({self.stake_divider.value}): {adjusted_stake}")
            logging.info(f"- Max allowed with leverage ({leverage}x): {max_allowed_stake}")
            logging.info(f"- Minimum stake amount: {min_stake_to_use}")
            logging.info(f"- Final adjusted stake: {adjusted_stake}")
        
        return adjusted_stake

    def set_max_stake_amount(self, amount: float) -> None:
        """
        Set the maximum stake amount
        :param amount: Maximum stake amount in stake currency
        """
        self.max_stake_amount = amount
        if self.debug_mode:
            logging.info(f"Set max stake amount to {amount}")

    def set_minimum_stake_amount(self, amount: float) -> None:
        """
        Set the minimum stake amount
        :param amount: Minimum stake amount in stake currency
        """
        self.minimum_stake_amount = amount
        if self.debug_mode:
            logging.info(f"Set minimum stake amount to {amount}")

    def set_debug_mode(self, enabled=True):
        """
        Enable or disable debug mode
        :param enabled: True to enable debug mode, False to disable
        """
        self.debug_mode = enabled

    def calculate_fibonacci_levels(self, dataframe: DataFrame) -> tuple:
        """
        Calculate Fibonacci levels based on the highest high and lowest low in the lookback period
        """
        # Find current swing high and low
        current_swing_high = dataframe['high'].rolling(window=self.fib_lookback_candles.value).max()
        current_swing_low = dataframe['low'].rolling(window=self.fib_lookback_candles.value).min()
        current_fib_range = current_swing_high - current_swing_low

        # Calculate Fibonacci price levels - directly matching Pine Script's levels
        price_0 = current_swing_high  # 0% (100%)
        price_382 = current_swing_high - (current_fib_range * 0.382)  # 38.2% (61.8%)
        price_5 = current_swing_high - (current_fib_range * 0.5)  # 50% (50%)
        price_618 = current_swing_high - (current_fib_range * 0.618)  # 61.8% (38.2%)
        price_65 = current_swing_high - (current_fib_range * 0.65)  # 65% (35%)
        price_8 = current_swing_high - (current_fib_range * 0.8)  # 80% (20%)
        price_88 = current_swing_high - (current_fib_range * 0.88)  # 88% (12%)
        price_1 = current_swing_low  # 100% (0%)

        return price_0, price_382, price_5, price_618, price_65, price_8, price_88, price_1

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are designed to be used in the strategy alongside the
        current timeframe and are automatically inserted into the beginning of the strategy.
        """
        # Always return empty list during hyperopt to avoid timeframe issues
        if hasattr(self, 'dp') and hasattr(self.dp, 'runmode') and self.dp.runmode.value in ('hyperopt', 'backtest'):
            return []
            
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        
        if self.use_multi_timeframe.value:
            # Dictionary of timeframe minutes for comparison
            tf_minutes = {
                '5m': 5,
                '15m': 15,
                '1h': 60,
                '4h': 240,
                '1d': 1440,
                '1w': 10080
            }
            
            # Current timeframe
            current_tf = self.timeframe
            
            # Only add higher timeframes, not lower ones
            for higher_tf in ['15m', '1h', '4h', '1d', '1w']:
                # Skip if higher_tf is not actually higher than current_tf
                if current_tf not in tf_minutes or higher_tf not in tf_minutes:
                    continue
                    
                if tf_minutes[higher_tf] <= tf_minutes[current_tf]:
                    continue
                    
                # Add all pairs with the higher timeframe
                for pair in pairs:
                    informative_pairs.append((pair, higher_tf))
        
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators to the dataframe - directly matching Pine Script indicators
        """
        # Calculate Moving Averages
        # Fast MA - matching Pine Script
        if self.ma_type1.value == 'SMA':
            dataframe['ma_fast'] = ta.SMA(dataframe['close'], timeperiod=self.len_fast.value)
        else:
            dataframe['ma_fast'] = ta.EMA(dataframe['close'], timeperiod=self.len_fast.value)
            
        # Slow MA - matching Pine Script
        if self.ma_type2.value == 'SMA':
            dataframe['ma_slow'] = ta.SMA(dataframe['close'], timeperiod=self.len_slow.value)
        else:
            dataframe['ma_slow'] = ta.EMA(dataframe['close'], timeperiod=self.len_slow.value)
            
        # Trend MA - matching Pine Script
        if self.ma_type3.value == 'SMA':
            dataframe['ma_tend'] = ta.SMA(dataframe['close'], timeperiod=self.len_tend.value)
        else:
            dataframe['ma_tend'] = ta.EMA(dataframe['close'], timeperiod=self.len_tend.value)
        
        # Calculate MA crossovers - matching Pine Script
        dataframe['cross_up'] = qtpylib.crossed_above(dataframe['ma_fast'], dataframe['ma_slow'])
        dataframe['cross_down'] = qtpylib.crossed_below(dataframe['ma_fast'], dataframe['ma_slow'])
        
        # Calculate RSI - matching Pine Script
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_length.value)
        
        # Calculate Fibonacci levels if enabled - matching Pine Script
        if self.show_fib.value:
            price_0, price_382, price_5, price_618, price_65, price_8, price_88, price_1 = self.calculate_fibonacci_levels(dataframe)
            dataframe['fib_0'] = price_0
            dataframe['fib_382'] = price_382
            dataframe['fib_5'] = price_5
            dataframe['fib_618'] = price_618
            dataframe['fib_65'] = price_65
            dataframe['fib_8'] = price_8
            dataframe['fib_88'] = price_88
            dataframe['fib_1'] = price_1
            
            # Fibonacci conditions - exact match to Pine Script
            dataframe['fib_long_condition'] = (
                (dataframe['close'] > dataframe['fib_382']) & 
                (dataframe['close'] < dataframe['fib_0']) & 
                (dataframe['low'] > dataframe['fib_5'])
            )
            
            dataframe['fib_short_condition'] = (
                (dataframe['close'] < dataframe['fib_618']) & 
                (dataframe['close'] > dataframe['fib_1']) & 
                (dataframe['high'] < dataframe['fib_5'])
            )
        
        # Calculate position projection levels for SL and TP - with leverage adjustment
        if self.show_projections.value:
            # Get current leverage value
            leverage = self.leverage_value.value
            
            # Adjust stop loss percentage based on leverage
            # For higher leverage, use a tighter stop loss to manage risk
            adjusted_sl_percent = self.sl_percent.value / leverage if leverage > 1 else self.sl_percent.value
            
            # Keep the same risk-reward ratio but apply it to the adjusted stop loss
            # This maintains the same absolute profit target while reducing risk
            dataframe['long_sl_price'] = dataframe['close'] * (1 - adjusted_sl_percent/100)
            dataframe['long_tp_price'] = dataframe['close'] * (1 + (adjusted_sl_percent * self.rr_ratio.value)/100)
            
            dataframe['short_sl_price'] = dataframe['close'] * (1 + adjusted_sl_percent/100)
            dataframe['short_tp_price'] = dataframe['close'] * (1 - (adjusted_sl_percent * self.rr_ratio.value)/100)
            
            # Store the adjusted percentages for reference
            dataframe['adjusted_sl_percent'] = adjusted_sl_percent
            dataframe['adjusted_tp_percent'] = adjusted_sl_percent * self.rr_ratio.value
            
            # Calculate effective risk-reward with leverage (for information purposes)
            # Leverage amplifies both risk and reward
            dataframe['effective_rr'] = self.rr_ratio.value * leverage
        
        # No informative merging during hyperopt/backtest
        if self.use_multi_timeframe.value and (not hasattr(self, 'dp') or not hasattr(self.dp, 'runmode') or self.dp.runmode.value not in ('hyperopt', 'backtest')):
            try:
                # Dictionary of timeframe minutes for comparison
                tf_minutes = {
                    '5m': 5,
                    '15m': 15,
                    '1h': 60,
                    '4h': 240,
                    '1d': 1440,
                    '1w': 10080
                }
                
                # Current timeframe
                current_tf = self.timeframe
                
                # Find next higher timeframe that is available
                for higher_tf in ['15m', '1h', '4h', '1d', '1w']:
                    # Make sure the higher timeframe is actually higher
                    if higher_tf not in tf_minutes or current_tf not in tf_minutes:
                        continue
                        
                    if tf_minutes[higher_tf] <= tf_minutes[current_tf]:
                        continue
                        
                    # Try to get the informative dataframe
                    try:
                        informative = self.dp.get_pair_dataframe(metadata['pair'], higher_tf)
                        
                        # Merge informative timeframe
                        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, higher_tf, ffill=True)
                        
                        # Use higher timeframe for major trend direction
                        if f"{higher_tf}_ma_fast" in dataframe.columns and f"{higher_tf}_ma_slow" in dataframe.columns:
                            dataframe['higher_tf_trend'] = (dataframe[f"{higher_tf}_ma_fast"] > dataframe[f"{higher_tf}_ma_slow"]).astype(int)
                        
                        # Optional: Combine higher timeframe RSI with current RSI (weighted average)
                        if 'rsi' in dataframe.columns and f"{higher_tf}_rsi" in dataframe.columns:
                            dataframe['combined_rsi'] = (
                                (1 - self.higher_timeframe_weight.value) * dataframe['rsi'] + 
                                self.higher_timeframe_weight.value * dataframe[f"{higher_tf}_rsi"]
                            )
                            
                        # Successfully merged, don't try more timeframes
                        break
                    except Exception as e:
                        # If failed to get this timeframe, try the next one
                        logging.info(f"Could not merge {higher_tf} for {metadata['pair']}: {e}")
                        continue
            except Exception as e:
                logging.info(f"Error in multi-timeframe merging: {e}")
                # Continue with just the current timeframe indicators
                pass
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the entry signals - exactly matching Pine Script entry conditions
        with added multi-timeframe capability
        """
        # Initialize entry tag columns
        dataframe['enter_tag'] = ''
        
        # Individual condition components
        ma_crossover_long = dataframe['ma_fast'].shift(1) > dataframe['ma_slow'].shift(1)
        ma_crossover_short = dataframe['ma_fast'].shift(1) < dataframe['ma_slow'].shift(1)
        
        rsi_filter_long = (
            (dataframe['rsi'].shift(1) > self.rsi_buy_lower.value) &
            (dataframe['rsi'].shift(1) < self.rsi_buy_upper.value)
        )
        rsi_filter_short = (
            (dataframe['rsi'].shift(1) < self.rsi_sell_lower.value) &
            (dataframe['rsi'].shift(1) > self.rsi_sell_upper.value)
        )
        
        candle_pattern_long = dataframe['close'].shift(1) > dataframe['open'].shift(1)
        candle_pattern_short = dataframe['close'].shift(1) < dataframe['open'].shift(1)
        
        trend_filter_long = True  # Default to true if not using trend filter
        trend_filter_short = True  # Default to true if not using trend filter
        
        # Only check trend if needed and columns are available
        if self.use_trend_filter.value and 'ma_tend' in dataframe.columns:
            trend_filter_long = dataframe['close'] > dataframe['ma_tend']
            trend_filter_short = dataframe['close'] < dataframe['ma_tend']

        # Build basic conditions by selectively including components based on switches
        basic_long = True
        basic_short = True
        
        if self.use_ma_crossover.value:
            basic_long = basic_long & ma_crossover_long
            basic_short = basic_short & ma_crossover_short
            
        if self.use_rsi_filter.value:
            basic_long = basic_long & rsi_filter_long
            basic_short = basic_short & rsi_filter_short
            
        if self.use_candle_pattern.value:
            basic_long = basic_long & candle_pattern_long
            basic_short = basic_short & candle_pattern_short
            
        if self.use_trend_filter.value:
            basic_long = basic_long & trend_filter_long
            basic_short = basic_short & trend_filter_short
        
        # Only use multi-timeframe confirmation outside of hyperopt/backtest
        use_multi_tf = (
            self.use_multi_timeframe.value and 
            'higher_tf_trend' in dataframe.columns and 
            (not hasattr(self, 'dp') or not hasattr(self.dp, 'runmode') or self.dp.runmode.value not in ('hyperopt', 'backtest'))
        )
        
        # Multi-timeframe confirmation if enabled and NOT in hyperopt/backtest
        if use_multi_tf:
            # For long entries, higher timeframe should be in an uptrend
            multi_tf_long_condition = (dataframe['higher_tf_trend'] > 0)
            
            # For short entries, higher timeframe should be in a downtrend
            multi_tf_short_condition = (dataframe['higher_tf_trend'] == 0)
            
            # Strengthen basic conditions with higher timeframe confirmation
            if 'combined_rsi' in dataframe.columns and self.use_rsi_filter.value:
                # Use the combined RSI if available
                basic_long = basic_long & (
                    (dataframe['combined_rsi'].shift(1) > self.rsi_buy_lower.value) &
                    (dataframe['combined_rsi'].shift(1) < self.rsi_buy_upper.value)
                )
                
                basic_short = basic_short & (
                    (dataframe['combined_rsi'].shift(1) < self.rsi_sell_lower.value) &
                    (dataframe['combined_rsi'].shift(1) > self.rsi_sell_upper.value)
                )
        
        # Apply final entry conditions with Fibonacci if enabled
        if self.show_fib.value and self.use_fib_for_entry.value and 'fib_long_condition' in dataframe.columns:
            # Long with Fibonacci
            long_fib_condition = basic_long & dataframe['fib_long_condition']
            
            # Add multi-timeframe confirmation if enabled and NOT in hyperopt/backtest
            if use_multi_tf:
                long_fib_mtf_condition = long_fib_condition & multi_tf_long_condition
                dataframe.loc[long_fib_mtf_condition, 'enter_tag'] = 'long_fib_mtf'
                dataframe.loc[long_fib_mtf_condition, 'enter_long'] = 1
                
                # Long with Fibonacci but without MTF confirmation
                long_fib_only_condition = long_fib_condition & ~multi_tf_long_condition
                dataframe.loc[long_fib_only_condition, 'enter_tag'] = 'long_fib_only'
                dataframe.loc[long_fib_only_condition, 'enter_long'] = 1
            else:
                dataframe.loc[long_fib_condition, 'enter_tag'] = 'long_fib'
                dataframe.loc[long_fib_condition, 'enter_long'] = 1
            
            # Short with Fibonacci
            short_fib_condition = basic_short & dataframe['fib_short_condition']
            
            # Add multi-timeframe confirmation if enabled and NOT in hyperopt/backtest
            if use_multi_tf:
                short_fib_mtf_condition = short_fib_condition & multi_tf_short_condition
                dataframe.loc[short_fib_mtf_condition, 'enter_tag'] = 'short_fib_mtf'
                dataframe.loc[short_fib_mtf_condition, 'enter_short'] = 1
                
                # Short with Fibonacci but without MTF confirmation
                short_fib_only_condition = short_fib_condition & ~multi_tf_short_condition
                dataframe.loc[short_fib_only_condition, 'enter_tag'] = 'short_fib_only'
                dataframe.loc[short_fib_only_condition, 'enter_short'] = 1
            else:
                dataframe.loc[short_fib_condition, 'enter_tag'] = 'short_fib'
                dataframe.loc[short_fib_condition, 'enter_short'] = 1
        else:
            # If Fibonacci is not used, rely on basic conditions
            
            # Generate tags for basic conditions
            if self.use_ma_crossover.value and self.use_rsi_filter.value and self.use_trend_filter.value:
                tag_prefix_long = 'long_ma_rsi_trend'
                tag_prefix_short = 'short_ma_rsi_trend'
            elif self.use_ma_crossover.value and self.use_rsi_filter.value:
                tag_prefix_long = 'long_ma_rsi'
                tag_prefix_short = 'short_ma_rsi'
            elif self.use_ma_crossover.value and self.use_trend_filter.value:
                tag_prefix_long = 'long_ma_trend'
                tag_prefix_short = 'short_ma_trend'
            elif self.use_rsi_filter.value and self.use_trend_filter.value:
                tag_prefix_long = 'long_rsi_trend'
                tag_prefix_short = 'short_rsi_trend'
            elif self.use_ma_crossover.value:
                tag_prefix_long = 'long_ma'
                tag_prefix_short = 'short_ma'
            elif self.use_rsi_filter.value:
                tag_prefix_long = 'long_rsi'
                tag_prefix_short = 'short_rsi'
            elif self.use_trend_filter.value:
                tag_prefix_long = 'long_trend'
                tag_prefix_short = 'short_trend'
            else:
                tag_prefix_long = 'long_basic'
                tag_prefix_short = 'short_basic'
            
            # Add candle pattern to tag if used
            if self.use_candle_pattern.value:
                tag_prefix_long += '_candle'
                tag_prefix_short += '_candle'
            
            # Add multi-timeframe confirmation if enabled and NOT in hyperopt/backtest
            if use_multi_tf:
                long_mtf_condition = basic_long & multi_tf_long_condition
                dataframe.loc[long_mtf_condition, 'enter_tag'] = f'{tag_prefix_long}_mtf'
                dataframe.loc[long_mtf_condition, 'enter_long'] = 1
                
                # Long without MTF confirmation
                long_only_condition = basic_long & ~multi_tf_long_condition
                dataframe.loc[long_only_condition, 'enter_tag'] = tag_prefix_long
                dataframe.loc[long_only_condition, 'enter_long'] = 1
                
                short_mtf_condition = basic_short & multi_tf_short_condition
                dataframe.loc[short_mtf_condition, 'enter_tag'] = f'{tag_prefix_short}_mtf'
                dataframe.loc[short_mtf_condition, 'enter_short'] = 1
                
                # Short without MTF confirmation
                short_only_condition = basic_short & ~multi_tf_short_condition
                dataframe.loc[short_only_condition, 'enter_tag'] = tag_prefix_short
                dataframe.loc[short_only_condition, 'enter_short'] = 1
            else:
                dataframe.loc[basic_long, 'enter_tag'] = tag_prefix_long
                dataframe.loc[basic_long, 'enter_long'] = 1
                
                dataframe.loc[basic_short, 'enter_tag'] = tag_prefix_short
                dataframe.loc[basic_short, 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the exit signals - using SL/TP from position projections
        """
        # Initialize exit conditions and tags
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''
        
        # MA crossover exit condition
        if self.use_ma_crossover_exit.value:
            # Exit long when fast MA crosses below slow MA
            ma_cross_long_exit = (dataframe['ma_fast'] < dataframe['ma_slow'])
            dataframe.loc[ma_cross_long_exit, 'exit_long'] = 1
            dataframe.loc[ma_cross_long_exit, 'exit_tag'] = 'long_ma_cross_exit'
            
            # Exit short when fast MA crosses above slow MA
            ma_cross_short_exit = (dataframe['ma_fast'] > dataframe['ma_slow'])
            dataframe.loc[ma_cross_short_exit, 'exit_short'] = 1
            dataframe.loc[ma_cross_short_exit, 'exit_tag'] = 'short_ma_cross_exit'
        
        # RSI exit condition
        if self.use_rsi_exit.value:
            # Exit long when RSI crosses above upper threshold
            rsi_long_exit = (dataframe['rsi'] > 70) & (dataframe['exit_long'] == 0)
            dataframe.loc[rsi_long_exit, 'exit_long'] = 1
            dataframe.loc[rsi_long_exit, 'exit_tag'] = 'long_rsi_overbought'
            
            # Exit short when RSI crosses below lower threshold
            rsi_short_exit = (dataframe['rsi'] < 30) & (dataframe['exit_short'] == 0)
            dataframe.loc[rsi_short_exit, 'exit_short'] = 1
            dataframe.loc[rsi_short_exit, 'exit_tag'] = 'short_rsi_oversold'
        
        # Only use multi-timeframe confirmation outside of hyperopt/backtest
        use_multi_tf = (
            self.use_multi_timeframe.value and 
            'higher_tf_trend' in dataframe.columns and 
            (not hasattr(self, 'dp') or not hasattr(self.dp, 'runmode') or self.dp.runmode.value not in ('hyperopt', 'backtest'))
        )
        
        # Add multi-timeframe confirmation for exits if enabled and NOT in hyperopt/backtest
        if use_multi_tf:
            # Additional exit condition when higher timeframe changes trend
            mtf_long_exit = (
                (dataframe['exit_long'] == 0) &  # Only add if not already set
                ((dataframe['higher_tf_trend'] == 0) &  # Higher timeframe turned bearish
                 (dataframe['rsi'] > 60))        # Plus RSI showing momentum loss
            )
            dataframe.loc[mtf_long_exit, 'exit_long'] = 1
            dataframe.loc[mtf_long_exit, 'exit_tag'] = 'long_mtf_trend_change'
            
            mtf_short_exit = (
                (dataframe['exit_short'] == 0) &  # Only add if not already set
                ((dataframe['higher_tf_trend'] > 0) &   # Higher timeframe turned bullish
                 (dataframe['rsi'] < 40))        # Plus RSI showing momentum loss
            )
            dataframe.loc[mtf_short_exit, 'exit_short'] = 1
            dataframe.loc[mtf_short_exit, 'exit_tag'] = 'short_mtf_trend_change'
        
        # Enable trailing stop loss if configured
        if self.use_trailing_sl.value:
            self.trailing_stop = True
            # Note: Trailing stop exits are handled by Freqtrade core, not by our exit signals
            # But we can log that we're using trailing stops
            if self.debug_mode:
                logging.info(f"Using trailing stop loss with positive offset: {self.trailing_stop_positive}")
        else:
            self.trailing_stop = False
        
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Confirm trade entry based on strategy rules.
        
        Args:
            pair (str): Pair that's about to be entered
            order_type (str): Order type (market/limit)
            amount (float): Stake amount to be used
            rate (float): Entry rate to be used
            time_in_force (str): Time in force (GTC, FOK, IOC, ...)
            current_time (datetime): Current datetime object
            entry_tag (Optional[str]): Entry tag to be used for this trade
            side (str): 'long' or 'short' - direction of the trade
            **kwargs: Additional arguments passed to strategy
            
        Returns:
            bool: Whether trade should be confirmed or not
        """
        if entry_tag:
            if self.debug_mode:
                logging.info(f"Confirming {side} trade for {pair} with tag: {entry_tag}")
                
        # Simple check for minimum stake
        if self.check_for_min_stake(pair, rate, amount):
            return True
        return False

    def check_for_min_stake(self, pair: str, rate: float, amount: float) -> bool:
        # Calculate the stake amount in the stake currency
        stake_amount = amount * rate
        # Check if it's above our minimum threshold
        return stake_amount >= self.minimum_stake_amount

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Custom exit signal logic to be used by strategy.
        
        Args:
            pair (str): Pair that's currently analyzed
            trade (Trade): Current trade object.
            current_time (datetime): Current datetime object.
            current_rate (float): Current rate of the pair.
            current_profit (float): Current profit (as ratio) of the trade.
            **kwargs: Additional arguments passed to strategy.
            
        Returns:
            str|bool|None: Exit signal name if exit should occur, True if exit should occur with 
                           standard signal_name, False|None if no exit signal.
        """
        # Example profit-based custom exit logic
        if current_profit >= 0.05:  # 5% profit
            return "high_profit_exit"
        
        # Example time-based exit for long trades
        if trade.enter_tag and 'long' in trade.enter_tag and trade.open_date_utc < current_time - datetime.timedelta(hours=48):
            return "long_timeout_exit"
            
        # Example time-based exit for short trades
        if trade.enter_tag and 'short' in trade.enter_tag and trade.open_date_utc < current_time - datetime.timedelta(hours=48):
            return "short_timeout_exit"
            
        return None  # No custom exit signal
