from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import talib.abstract as ta
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, IntParameter, DecimalParameter
from typing import Optional, Union, Dict, Tuple, List, Any
import numpy as np
import logging


# Set pandas option to resolve FutureWarnings
pd.set_option('future.no_silent_downcasting', True)

class CCDeltaHyperopt(IStrategy):
    """
    Crypto Cradle Strategy Delta - Hyperopt Compatible Version
    
    Hyperopt Parameters (using new IntParameter/DecimalParameter pattern):
    - buy_profit_target_1: First profit taking level (1.0-2.0 R:R)
    - buy_swing_detection_window: Window for swing high/low detection (8-20)
    - buy_min_trend_strength: Minimum % move for trend validation (0.001-0.005)
    - buy_lookback_period: Swing low lookback period (2-5)
    - buy_confirmation_period: Swing low confirmation period (1-3)
    - buy_time_diff: Hours to keep entry candidates (2-6)
    
    Static Parameters (Core Crypto Cradle):
    - EMA 10/20: Core moving averages (unchanged)
    - MACD 12/26/9: Core momentum indicator (unchanged)
    """
    
    INTERFACE_VERSION = 3
    process_only_new_candles = True
    startup_candle_count: int = 700
    stoploss = -0.99
    use_custom_stoploss = True
    position_adjustment_enable = True

    # Static timeframe setup (unchanged)
    timeframe = '1h'
    informative_timeframe = '4h'

    # HYPEROPT PARAMETERS - NEW PATTERN: Using IntParameter/DecimalParameter
    buy_profit_target_1 = DecimalParameter(1.0, 2.0, decimals=1, default=1.0, space="buy")
    buy_swing_detection_window = IntParameter(8, 20, default=14, space="buy")
    buy_min_trend_strength = DecimalParameter(0.001, 0.005, decimals=3, default=0.002, space="buy")
    buy_lookback_period = IntParameter(2, 5, default=3, space="buy")
    buy_confirmation_period = IntParameter(1, 3, default=1, space="buy")
    buy_time_diff = IntParameter(2, 6, default=3, space="buy")

    # STATIC PARAMETERS (Core Crypto Cradle - Not optimized)
    ema_short = 10  # STATIC: Core Crypto Cradle EMA
    ema_long = 20   # STATIC: Core Crypto Cradle EMA
    macd_fast = 12  # STATIC: Core MACD settings
    macd_slow = 26  # STATIC: Core MACD settings
    macd_signal = 9 # STATIC: Core MACD settings

    # Custom attributes to track entry candidates
    entry_candidates = {}

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, self.informative_timeframe))
        return informative_pairs

    def calculate_trend_indicators(self, dataframe: DataFrame, timeframe: str = '1h') -> DataFrame:
        """Calculate trend indicators - STATIC parameters preserved"""
        # EMAs - STATIC (Core Crypto Cradle)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=self.ema_short)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=self.ema_long)

        # MACD - STATIC (Core Crypto Cradle)
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        # Trend detection - use different periods for different timeframes
        periods = 3 if timeframe == '4h' else 2
        dataframe['ema10_slope'] = dataframe['ema10'].pct_change(periods=periods, fill_method=None)
        dataframe['ema20_slope'] = dataframe['ema20'].pct_change(periods=periods, fill_method=None)

        # HYPEROPT: min_trend_strength optimization
        dataframe['uptrend'] = (
            (dataframe['ema10'] > dataframe['ema20']) &
            (dataframe['ema10_slope'] > self.buy_min_trend_strength.value) &  # HYPEROPT
            (dataframe['ema20_slope'] > 0)
        )

        # MACD convergence (strengthening momentum)
        dataframe['macd_convergence'] = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))
        )

        return dataframe

    def calculate_cradle_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate cradle-specific indicators - preserved from Delta"""
        # Candle body analysis
        dataframe['body_size'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['total_range'] = dataframe['high'] - dataframe['low']
        dataframe['body_ratio'] = dataframe['body_size'] / dataframe['total_range'].replace(0, np.nan)

        # Upper and lower wicks
        dataframe['upper_wick'] = dataframe['high'] - np.maximum(dataframe['open'], dataframe['close'])
        dataframe['lower_wick'] = np.minimum(dataframe['open'], dataframe['close']) - dataframe['low']
        dataframe['upper_wick_ratio'] = dataframe['upper_wick'] / dataframe['total_range'].replace(0, np.nan)
        dataframe['lower_wick_ratio'] = dataframe['lower_wick'] / dataframe['total_range'].replace(0, np.nan)

        # Small candle detection (indecision) - STATIC thresholds
        dataframe['small_candle'] = dataframe['body_ratio'] < 0.3
        dataframe['small_upper_wick'] = dataframe['upper_wick_ratio'] < 0.4

        # Bullish candle detection
        dataframe['is_bullish'] = dataframe['close'] > dataframe['open']

        # Cradle zone detection (between EMAs) - STATIC logic
        dataframe['in_cradle_zone'] = (
            (dataframe['ema10'] > dataframe['ema20']) &
            (
                # Candle reaches into zone from below
                ((dataframe['low'] < dataframe['ema20']) &
                 (dataframe['high'] >= dataframe['ema20'])) |
                # Candle reaches into zone from above
                ((dataframe['high'] > dataframe['ema10']) & (dataframe['low'] <= dataframe['ema10'])) |
                # Candle entirely within the cradle zone
                ((dataframe['low'] >= dataframe['ema20']) & (dataframe['high'] <= dataframe['ema10']))
            )
        )

        return dataframe

    def identify_swing_lows_advanced(self, dataframe: DataFrame) -> Dict[str, pd.Series]:
        """
        Advanced swing low detection - HYPEROPT optimized parameters
        """
        # HYPEROPT: Use parameter values directly
        lookback_period = self.buy_lookback_period.value
        confirmation_period = self.buy_confirmation_period.value
        
        results = {
            'swing_lows_confirmed': pd.Series(index=dataframe.index, dtype=float),
            'swing_lows_provisional': pd.Series(index=dataframe.index, dtype=float),
            'swing_lows_combined': pd.Series(index=dataframe.index, dtype=float)
        }
        
        for i in range(lookback_period, len(dataframe) - confirmation_period):
            current_low = dataframe.iloc[i]['low']
            
            # Check if current low is lower than lookback period
            lookback_lows = dataframe.iloc[i-lookback_period:i]['low']
            is_lower_than_lookback = all(current_low <= low for low in lookback_lows)
            
            # Check if current low is lower than confirmation period ahead
            confirmation_lows = dataframe.iloc[i+1:i+1+confirmation_period]['low']
            is_lower_than_confirmation = all(current_low <= low for low in confirmation_lows)
            
            if is_lower_than_lookback and is_lower_than_confirmation:
                results['swing_lows_confirmed'].iloc[i] = current_low
                results['swing_lows_combined'].iloc[i] = current_low
            elif is_lower_than_lookback:
                results['swing_lows_provisional'].iloc[i] = current_low
                results['swing_lows_combined'].iloc[i] = current_low
                
        return results

    def find_swing_lows_simple(self, dataframe: DataFrame) -> pd.Series:
        """Simple wrapper for the advanced swing low detection - HYPEROPT optimized"""
        try:
            return self.identify_swing_lows_advanced(dataframe)['swing_lows_combined']
            
        except Exception as e:
            # Fallback to basic swing low detection with HYPEROPT parameters
            lookback = self.buy_lookback_period.value
            swing_lows = pd.Series(index=dataframe.index, dtype=float)
            for i in range(lookback, len(dataframe) - 1):
                current_low = dataframe.iloc[i]['low']
                
                # Check if current low is lower than lookback period lows
                lookback_lows = dataframe.iloc[i-lookback:i]['low']
                is_lower_than_lookback = all(current_low < low for low in lookback_lows)
                
                # Check if current low is lower than next candle
                next_low = dataframe.iloc[i+1]['low']
                is_lower_than_next = current_low < next_low
                
                if is_lower_than_lookback and is_lower_than_next:
                    swing_lows.iloc[i] = current_low
                    
            return swing_lows

    def find_recent_higher_low_advanced(self, dataframe: DataFrame, 
                                      swing_analysis: Dict, 
                                      lookback: int = 50) -> pd.Series:
        """Enhanced recent higher low detection - preserved logic"""
        try:
            recent_higher_low = pd.Series(index=dataframe.index, dtype=float)
            uptrend_signal = pd.Series(index=dataframe.index, dtype=bool)
            
            # Use confirmed swing lows for higher reliability
            confirmed_lows = swing_analysis['swing_lows_confirmed'].dropna()
            
            for i in range(lookback, len(dataframe)):
                lookback_start = dataframe.index[max(0, i - lookback)]
                relevant_lows = confirmed_lows[(confirmed_lows.index >= lookback_start) &
                                               (confirmed_lows.index <= dataframe.index[i])]
                
                if len(relevant_lows) >= 2:
                    # Sort by time (most recent first)
                    sorted_lows = relevant_lows.sort_index(ascending=False)
                    
                    # Get most recent and second most recent confirmed swing lows
                    most_recent_low = sorted_lows.iloc[0]
                    second_recent_low = sorted_lows.iloc[1]
                    
                    # Higher low confirmation: most recent > second recent
                    if most_recent_low > second_recent_low:
                        recent_higher_low.iloc[i] = most_recent_low
                        uptrend_signal.iloc[i] = True
                        
            # Add confirmed uptrend signal to dataframe for override
            dataframe['confirmed_uptrend_signal'] = uptrend_signal
            return recent_higher_low
            
        except Exception as e:
            # Fallback to simple higher low detection
            return pd.Series(index=dataframe.index, dtype=float)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate indicators for both timeframes"""
        # Calculate trend indicators for 1h timeframe
        dataframe = self.calculate_trend_indicators(dataframe, self.timeframe)
        
        # Calculate cradle-specific indicators
        dataframe = self.calculate_cradle_indicators(dataframe)
        
        # Get 4h timeframe data
        df_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        df_4h = self.calculate_trend_indicators(df_4h, self.informative_timeframe)
        
        # Merge 4h indicators
        dataframe = merge_informative_pair(dataframe, df_4h, self.timeframe, self.informative_timeframe, ffill=True)

        # HYPEROPT: Swing low analysis with optimized parameters
        swing_analysis = self.identify_swing_lows_advanced(dataframe)
        dataframe['swing_low'] = swing_analysis['swing_lows_combined']
        
        # Recent higher low detection
        dataframe['recent_higher_low'] = self.find_recent_higher_low_advanced(dataframe, swing_analysis)
        
        # Override uptrend with confirmed swing low signal if available
        if 'confirmed_uptrend_signal' in dataframe.columns:
            dataframe['uptrend'] = dataframe['confirmed_uptrend_signal'].fillna(False)

        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()

        # Entry candidate detection - Core Crypto Cradle logic preserved
        dataframe['entry_candidate'] = (
            # Core Crypto Cradle conditions - price must be in cradle zone
            dataframe['in_cradle_zone'].fillna(False) &
            
            # Basic trend alignment (1h and 4h)
            dataframe['uptrend'].fillna(False) &
            dataframe.get('uptrend_4h', True) &
            
            # EMA structure (bullish) - STATIC EMAs
            (dataframe['ema10'] > dataframe['ema20']) &
            (dataframe.get('ema10_4h', dataframe['ema10']) > dataframe.get('ema20_4h', dataframe['ema20'])) &
            
            # MACD momentum (strengthening)
            dataframe['macd_convergence'].fillna(False) &
            dataframe.get('macd_convergence_4h', True) &
            
            # Candle characteristics (indecision/small body)
            dataframe['is_bullish'].fillna(False) &
            dataframe['small_candle'].fillna(False) &
            dataframe['small_upper_wick'].fillna(False)
        )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry logic - HYPEROPT optimized candidate management"""
        pair = metadata['pair']

        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_tag'] = ''

        if pair not in self.entry_candidates:
            self.entry_candidates[pair] = []

        for i in range(len(dataframe)):
            current_candle = dataframe.iloc[i]
            current_time = current_candle.name

            if pd.isna(current_candle.get('ema10', np.nan)):
                continue

            # HYPEROPT: Remove stale entry candidates with optimized time_diff
            stale_candidates = []
            for candidate in self.entry_candidates[pair]:
                candidate_time = candidate['timestamp']
                # Calculate time difference in hours
                if hasattr(current_time, 'to_pydatetime') and hasattr(candidate_time, 'to_pydatetime'):
                    time_diff = (current_time.to_pydatetime() - candidate_time.to_pydatetime()).total_seconds() / 3600
                else:
                    time_diff = (current_time - candidate_time).total_seconds() / 3600 if hasattr((current_time - candidate_time), 'total_seconds') else 0
                
                # HYPEROPT: Optimized time threshold
                if time_diff >= self.buy_time_diff.value:
                    stale_candidates.append(candidate)
            
            # Remove stale candidates
            for stale in stale_candidates:
                self.entry_candidates[pair].remove(stale)

            # Track new entry candidates - CORE CRYPTO CRADLE LOGIC
            if current_candle.get('entry_candidate', False):
                candidate_info = {
                    'timestamp': current_candle.name,
                    'high': current_candle['high'],
                    'stop_loss_level': current_candle.get('recent_higher_low', current_candle.get('ema20', 0) * 0.95),
                    'ema10': current_candle.get('ema10', 0),
                    'ema20': current_candle.get('ema20', 0)
                }
                self.entry_candidates[pair].append(candidate_info)
                self.entry_candidates[pair] = self.entry_candidates[pair][-5:]

            # CRYPTO CRADLE BREAKOUT DETECTION
            for candidate in self.entry_candidates[pair].copy():
                breakout_level = candidate['high']
                current_price = current_candle['close']

                # CORE CRYPTO CRADLE: Price breaks above previous cradle high
                breakout_confirmed = (
                    current_candle['high'] > breakout_level and
                    current_price > breakout_level and
                    current_candle.get('uptrend', False) and
                    current_candle.get('uptrend_4h', True) and
                    current_candle.get('in_cradle_zone', False)
                )

                if breakout_confirmed:
                    # Signal Crypto Cradle breakout entry
                    dataframe.loc[dataframe.index[i], 'enter_long'] = 1
                    dataframe.loc[dataframe.index[i], 'enter_tag'] = 'cradle_breakout'
                    
                    # Remove this candidate to prevent duplicate entries
                    self.entry_candidates[pair].remove(candidate)
                    break

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.populate_entry_trend(dataframe, metadata)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit logic - placeholder for compatibility"""
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ''
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.populate_exit_trend(dataframe, metadata)

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float, **kwargs) -> Optional[float]:
        """Hybrid profit scaling strategy - HYPEROPT optimized first target"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        latest_candle = dataframe.iloc[-1]
        entry_price = trade.open_rate
        
        # Calculate stop loss level
        stop_loss_level = latest_candle.get('recent_higher_low', 0)
        if stop_loss_level == 0:
            stop_loss_level = latest_candle.get('ema20', entry_price) * 0.97

        if stop_loss_level > 0:
            risk = entry_price - stop_loss_level
            # HYPEROPT: Optimized first profit target
            take_profit_1to1 = entry_price + (risk * self.buy_profit_target_1.value)  # HYPEROPT
            take_profit_2to1 = entry_price + (risk * 2)  # Fixed 2:1
            
            # First partial exit: 50% at HYPEROPT optimized target
            if (current_rate >= take_profit_1to1 and 
                not hasattr(trade, 'first_partial_exit_taken')):
                
                trade.first_partial_exit_taken = True
                trade.exit_reason_tag = "take_profit_1to1_partial"
                
                logging.info(f"=== HYPEROPT FIRST PARTIAL EXIT: 50% at {self.buy_profit_target_1.value}:1 for {trade.pair} ===")
                logging.info(f"Target: {take_profit_1to1:.6f}, Current: {current_rate:.6f}")
                
                return -(trade.amount * 0.5)  # Sell 50% of position
            
            # Second partial exit: 25% at 2:1 (fixed)
            elif (current_rate >= take_profit_2to1 and
                  hasattr(trade, 'first_partial_exit_taken') and
                  not hasattr(trade, 'second_partial_exit_taken')):
                
                trade.second_partial_exit_taken = True
                trade.exit_reason_tag = "take_profit_2to1_partial"
                
                logging.info(f"=== SECOND PARTIAL EXIT: 25% at 2:1 for {trade.pair} ===")
                logging.info(f"Target: {take_profit_2to1:.6f}, Current: {current_rate:.6f}")
                
                return -(trade.amount * 0.5)  # Sell 50% of remaining (25% of original)

        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, current_time: datetime,
                          entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """Enhanced trade entry confirmation with HYPEROPT parameters"""
        return True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """Advanced trailing stop loss system - preserved logic"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        latest_candle = dataframe.iloc[-1]
        entry_price = trade.open_rate
        
        # Calculate stop loss level
        stop_loss_level = latest_candle.get('recent_higher_low', 0)
        if stop_loss_level == 0:
            stop_loss_level = latest_candle.get('ema20', entry_price) * 0.97

        if hasattr(trade, 'second_partial_exit_taken'):
            # Phase 3: Aggressive trailing for final 25%
            if stop_loss_level > 0:
                risk = entry_price - stop_loss_level
                break_even_plus = entry_price + (risk * 0.5)
                trail_level = max(break_even_plus, stop_loss_level * 0.995)
                dynamic_stop = (trail_level / current_rate) - 1
                calculated_stop = max(dynamic_stop, -0.005)
                
                if current_rate <= trail_level:
                    trade.exit_reason_tag = "advanced_trailing_stop"
                
                return calculated_stop
                
        elif hasattr(trade, 'first_partial_exit_taken'):
            # Phase 2: Breakeven protection for remaining 50%
            break_even_level = entry_price * 0.995
            break_even_stop = (break_even_level / current_rate) - 1
            calculated_stop = max(break_even_stop, self.stoploss)
            
            if current_rate <= break_even_level:
                trade.exit_reason_tag = "breakeven_trailing_stop"
            
            return calculated_stop
            
        else:
            # Phase 1: Initial position with swing-based stops
            if stop_loss_level > 0:
                stop_loss_rate = stop_loss_level * 0.99
                dynamic_stop = (stop_loss_rate / current_rate) - 1
                calculated_stop = max(dynamic_stop, self.stoploss)
                
                if current_rate <= stop_loss_rate:
                    trade.exit_reason_tag = "initial_trailing_stop"
                
                return calculated_stop
            
            # Fallback to EMA20-based stop
            current_ema20 = latest_candle.get('ema20', None)
            if current_ema20 and current_ema20 > 0:
                stop_loss_rate = current_ema20 * 0.97
                dynamic_stop = (stop_loss_rate / current_rate) - 1
                calculated_stop = max(dynamic_stop, self.stoploss)
                
                if current_rate <= stop_loss_rate:
                    trade.exit_reason_tag = "initial_trailing_stop"
                
                return calculated_stop

        return self.stoploss

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """Custom exit logic - HYPEROPT optimized profit targets"""
        # PRIORITY 1: Return stored exit tag from position adjustment
        if hasattr(trade, 'exit_reason_tag'):
            reason = trade.exit_reason_tag
            delattr(trade, 'exit_reason_tag')
            return reason
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        latest_candle = dataframe.iloc[-1]
        entry_price = trade.open_rate
        
        # Calculate stop loss level
        stop_loss_level = latest_candle.get('recent_higher_low', 0)
        if stop_loss_level == 0:
            stop_loss_level = latest_candle.get('ema20', entry_price) * 0.97

        if stop_loss_level > 0:
            risk = entry_price - stop_loss_level
            # HYPEROPT: Optimized profit targets
            take_profit_1to1 = entry_price + (risk * self.buy_profit_target_1.value)  # HYPEROPT
            take_profit_2to1 = entry_price + (risk * 2)  # Fixed 2:1
            take_profit_3to1 = entry_price + (risk * 3)  # Fixed 3:1
            
            # Stop loss scenarios
            if current_rate <= stop_loss_level:
                if hasattr(trade, 'second_partial_exit_taken'):
                    return "advanced_trailing_stop"
                elif hasattr(trade, 'first_partial_exit_taken'):
                    return "breakeven_trailing_stop"
                else:
                    return "initial_trailing_stop"
            
            # Final exit at 3:1 target
            elif (current_rate >= take_profit_3to1 and
                  hasattr(trade, 'first_partial_exit_taken') and
                  hasattr(trade, 'second_partial_exit_taken')):
                return "take_profit_3to1_final"
            
            # Fallback exits if position adjustments failed
            elif (current_rate >= take_profit_2to1 and
                  hasattr(trade, 'first_partial_exit_taken') and
                  not hasattr(trade, 'second_partial_exit_taken')):
                return "take_profit_2to1_fallback"
            
            # HYPEROPT: Optimized first profit target
            elif (current_rate >= take_profit_1to1 and
                  not hasattr(trade, 'first_partial_exit_taken')):
                return "take_profit_1to1_fallback"
            
            # Emergency exits
            elif current_profit >= 0.20:
                return "emergency_profit_exit"
            elif current_profit <= -0.10:
                return "emergency_loss_exit"

        return None