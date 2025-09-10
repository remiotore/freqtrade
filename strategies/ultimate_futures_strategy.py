from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from typing import Dict, List, Optional, Tuple

class UltimateHighProbFutures(IStrategy):
    """
    Ultimate High-Probability Futures Strategy
    Combines multiple confirmation layers for maximum win rate
    """
    
    # Core Settings
    stoploss = -0.06  # 6% maximum loss
    can_short = True
    trailing_stop = True
    trailing_stop_positive = 0.01  # Start trailing at 1%
    trailing_stop_positive_offset = 0.015  # 1.5% trailing distance
    
    timeframe = '5m'
    
    # Aggressive ROI for futures
    minimal_roi = {
        "0": 0.08,   # 8% immediate target
        "10": 0.05,  # 5% after 10 minutes
        "20": 0.03,  # 3% after 20 minutes
        "40": 0.02,  # 2% after 40 minutes
        "80": 0.01   # 1% after 80 minutes
    }
    
    # Strategy Parameters
    confirmation_layers = 5  # Minimum confirmations needed
    max_positions = 3  # Maximum concurrent trades
    
    # Dynamic parameters
    volatility_threshold = 1.2
    trend_strength_min = 0.5
    volume_spike_threshold = 1.5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Multi-layered indicator system for maximum confirmation
        """
        
        # === LAYER 1: TREND ANALYSIS ===
        
        # Multiple EMA system for trend confirmation
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_34'] = ta.EMA(dataframe, timeperiod=34)
        dataframe['ema_55'] = ta.EMA(dataframe, timeperiod=55)
        
        # Trend strength calculation
        dataframe['trend_strength'] = abs(
            (dataframe['ema_8'] - dataframe['ema_55']) / dataframe['ema_55']
        ) * 100
        
        # Trend direction (all EMAs aligned)
        dataframe['uptrend'] = (
            (dataframe['ema_8'] > dataframe['ema_13']) &
            (dataframe['ema_13'] > dataframe['ema_21']) &
            (dataframe['ema_21'] > dataframe['ema_34']) &
            (dataframe['ema_34'] > dataframe['ema_55'])
        )
        
        dataframe['downtrend'] = (
            (dataframe['ema_8'] < dataframe['ema_13']) &
            (dataframe['ema_13'] < dataframe['ema_21']) &
            (dataframe['ema_21'] < dataframe['ema_34']) &
            (dataframe['ema_34'] < dataframe['ema_55'])
        )
        
        # === LAYER 2: MOMENTUM ANALYSIS ===
        
        # Multiple RSI for momentum confirmation
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
        
        # Stochastic for additional momentum
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # MACD for trend momentum
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Momentum alignment
        dataframe['momentum_up'] = (
            (dataframe['rsi_7'] > dataframe['rsi_14']) &
            (dataframe['rsi_14'] > 40) &
            (dataframe['rsi_21'] > 45) &
            (dataframe['stoch_k'] > dataframe['stoch_d']) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macdhist'] > 0)
        )
        
        dataframe['momentum_down'] = (
            (dataframe['rsi_7'] < dataframe['rsi_14']) &
            (dataframe['rsi_14'] < 60) &
            (dataframe['rsi_21'] < 55) &
            (dataframe['stoch_k'] < dataframe['stoch_d']) &
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macdhist'] < 0)
        )
        
        # === LAYER 3: VOLATILITY & VOLUME ANALYSIS ===
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
        dataframe['volatility_ratio'] = (
            dataframe['atr_pct'] / dataframe['atr_pct'].rolling(50).mean()
        )
        
        # Volume analysis
        if 'volume' in dataframe.columns:
            dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
            dataframe['volume_spike'] = dataframe['volume_ratio'] > self.volume_spike_threshold
        else:
            dataframe['volume_ratio'] = 1
            dataframe['volume_spike'] = True
        
        # Price action volatility
        dataframe['price_volatility'] = (
            (dataframe['high'] - dataframe['low']) / dataframe['close']
        ) * 100
        
        # === LAYER 4: SUPPORT/RESISTANCE & PIVOT ANALYSIS ===
        
        # Pivot points calculation
        pivot_window = 20
        
        # Rolling pivot highs and lows
        dataframe['pivot_high'] = dataframe['high'].rolling(
            window=pivot_window, center=True
        ).max().shift(-pivot_window//2)
        
        dataframe['pivot_low'] = dataframe['low'].rolling(
            window=pivot_window, center=True
        ).min().shift(-pivot_window//2)
        
        # Identify actual pivot points
        dataframe['is_pivot_high'] = (
            dataframe['high'] == dataframe['pivot_high']
        )
        dataframe['is_pivot_low'] = (
            dataframe['low'] == dataframe['pivot_low']
        )
        
        # Dynamic support/resistance
        dataframe['resistance'] = dataframe.loc[
            dataframe['is_pivot_high'], 'high'
        ].fillna(method='ffill')
        
        dataframe['support'] = dataframe.loc[
            dataframe['is_pivot_low'], 'low'
        ].fillna(method='ffill')
        
        # Distance from S/R levels
        dataframe['dist_from_resistance'] = (
            dataframe['resistance'] - dataframe['close']
        ) / dataframe['close'] * 100
        
        dataframe['dist_from_support'] = (
            dataframe['close'] - dataframe['support']
        ) / dataframe['close'] * 100
        
        # === LAYER 5: BOLLINGER BANDS & MEAN REVERSION ===
        
        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_lower'] = bb['lowerband']
        dataframe['bb_middle'] = bb['middleband']
        
        # BB position and width
        dataframe['bb_position'] = (
            (dataframe['close'] - dataframe['bb_lower']) / 
            (dataframe['bb_upper'] - dataframe['bb_lower'])
        )
        dataframe['bb_width'] = (
            (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        ) * 100
        
        # === LAYER 6: CONFLUENCE ZONES ===
        
        # Price near key levels
        proximity_threshold = 0.5  # 0.5%
        
        dataframe['near_ema_confluence'] = (
            (abs(dataframe['close'] - dataframe['ema_21']) / dataframe['close'] * 100 < proximity_threshold) |
            (abs(dataframe['close'] - dataframe['ema_34']) / dataframe['close'] * 100 < proximity_threshold)
        )
        
        dataframe['near_support_resistance'] = (
            (dataframe['dist_from_resistance'] < proximity_threshold) |
            (dataframe['dist_from_support'] < proximity_threshold)
        )
        
        # === SIGNAL STRENGTH CALCULATION ===
        
        # Long signal strength (0-10 scale)
        dataframe['long_strength'] = 0
        
        # Trend confirmation (2 points)
        dataframe.loc[dataframe['uptrend'], 'long_strength'] += 2
        dataframe.loc[dataframe['trend_strength'] > self.trend_strength_min, 'long_strength'] += 1
        
        # Momentum confirmation (2 points)
        dataframe.loc[dataframe['momentum_up'], 'long_strength'] += 2
        
        # Volume confirmation (1 point)
        dataframe.loc[dataframe['volume_spike'], 'long_strength'] += 1
        
        # Volatility confirmation (1 point)
        dataframe.loc[
            (dataframe['volatility_ratio'] > self.volatility_threshold) & 
            (dataframe['volatility_ratio'] < 3.0),
            'long_strength'
        ] += 1
        
        # Support/Resistance confirmation (1 point)
        dataframe.loc[dataframe['close'] > dataframe['support'], 'long_strength'] += 1
        
        # Bollinger position (1 point)
        dataframe.loc[
            (dataframe['bb_position'] > 0.2) & (dataframe['bb_position'] < 0.8),
            'long_strength'
        ] += 1
        
        # RSI not extreme (1 point)
        dataframe.loc[
            (dataframe['rsi_14'] > 25) & (dataframe['rsi_14'] < 75),
            'long_strength'
        ] += 1
        
        # Short signal strength (0-10 scale)
        dataframe['short_strength'] = 0
        
        # Trend confirmation (2 points)
        dataframe.loc[dataframe['downtrend'], 'short_strength'] += 2
        dataframe.loc[dataframe['trend_strength'] > self.trend_strength_min, 'short_strength'] += 1
        
        # Momentum confirmation (2 points)
        dataframe.loc[dataframe['momentum_down'], 'short_strength'] += 2
        
        # Volume confirmation (1 point)
        dataframe.loc[dataframe['volume_spike'], 'short_strength'] += 1
        
        # Volatility confirmation (1 point)
        dataframe.loc[
            (dataframe['volatility_ratio'] > self.volatility_threshold) & 
            (dataframe['volatility_ratio'] < 3.0),
            'short_strength'
        ] += 1
        
        # Support/Resistance confirmation (1 point)
        dataframe.loc[dataframe['close'] < dataframe['resistance'], 'short_strength'] += 1
        
        # Bollinger position (1 point)
        dataframe.loc[
            (dataframe['bb_position'] > 0.2) & (dataframe['bb_position'] < 0.8),
            'short_strength'
        ] += 1
        
        # RSI not extreme (1 point)
        dataframe.loc[
            (dataframe['rsi_14'] > 25) & (dataframe['rsi_14'] < 75),
            'short_strength'
        ] += 1
        
        # === MARKET REGIME DETECTION ===
        
        # Trending vs ranging market
        dataframe['market_trending'] = (
            dataframe['trend_strength'] > 1.0
        )
        
        dataframe['market_ranging'] = (
            (dataframe['bb_width'] < dataframe['bb_width'].rolling(20).mean() * 0.8) &
            (dataframe['trend_strength'] < 0.5)
        )
        
        # High volatility periods
        dataframe['high_volatility'] = (
            dataframe['volatility_ratio'] > 2.0
        )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # === ULTRA HIGH PROBABILITY LONG ENTRIES ===
        
        long_conditions = [
            # Core signal strength requirement
            dataframe['long_strength'] >= 7,  # At least 7/10 confirmations
            
            # Price action requirements
            dataframe['close'] > dataframe['ema_8'],
            dataframe['close'] > dataframe['open'],  # Green candle
            
            # Momentum requirements
            dataframe['rsi_7'] > dataframe['rsi_14'],
            dataframe['rsi_14'] > 30,
            dataframe['rsi_14'] < 70,
            
            # Volume requirement
            dataframe['volume_ratio'] > 1.0,
            
            # Volatility in sweet spot
            dataframe['volatility_ratio'] > 0.8,
            dataframe['volatility_ratio'] < 4.0,
            
            # Not at resistance
            dataframe['dist_from_resistance'] > 1.0,
            
            # Market regime appropriate
            dataframe['market_trending'] | 
            (dataframe['market_ranging'] & (dataframe['bb_position'] < 0.3)),
            
            # No extreme conditions
            ~dataframe['high_volatility'] | 
            (dataframe['high_volatility'] & (dataframe['long_strength'] >= 8)),
        ]
        
        dataframe.loc[
            np.logical_and.reduce(long_conditions),
            'enter_long'
        ] = 1
        
        # === ULTRA HIGH PROBABILITY SHORT ENTRIES ===
        
        short_conditions = [
            # Core signal strength requirement
            dataframe['short_strength'] >= 7,  # At least 7/10 confirmations
            
            # Price action requirements
            dataframe['close'] < dataframe['ema_8'],
            dataframe['close'] < dataframe['open'],  # Red candle
            
            # Momentum requirements
            dataframe['rsi_7'] < dataframe['rsi_14'],
            dataframe['rsi_14'] > 30,
            dataframe['rsi_14'] < 70,
            
            # Volume requirement
            dataframe['volume_ratio'] > 1.0,
            
            # Volatility in sweet spot
            dataframe['volatility_ratio'] > 0.8,
            dataframe['volatility_ratio'] < 4.0,
            
            # Not at support
            dataframe['dist_from_support'] > 1.0,
            
            # Market regime appropriate
            dataframe['market_trending'] | 
            (dataframe['market_ranging'] & (dataframe['bb_position'] > 0.7)),
            
            # No extreme conditions
            ~dataframe['high_volatility'] | 
            (dataframe['high_volatility'] & (dataframe['short_strength'] >= 8)),
        ]
        
        dataframe.loc[
            np.logical_and.reduce(short_conditions),
            'enter_short'
        ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # === INTELLIGENT LONG EXITS ===
        
        long_exit_conditions = [
            # Signal strength deterioration
            dataframe['long_strength'] <= 3,
            
            # Momentum reversal
            (dataframe['rsi_7'] < dataframe['rsi_14']) &
            (dataframe['rsi_14'] > 65),
            
            # Trend breakdown
            dataframe['close'] < dataframe['ema_13'],
            
            # At resistance
            dataframe['dist_from_resistance'] < 0.5,
            
            # MACD divergence
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macdhist'] < 0),
            
            # Volatility collapse
            dataframe['volatility_ratio'] < 0.5,
            
            # Bollinger Band squeeze
            dataframe['bb_width'] < dataframe['bb_width'].rolling(10).mean() * 0.6,
        ]
        
        dataframe.loc[
            np.logical_or.reduce(long_exit_conditions),
            'exit_long'
        ] = 1
        
        # === INTELLIGENT SHORT EXITS ===
        
        short_exit_conditions = [
            # Signal strength deterioration
            dataframe['short_strength'] <= 3,
            
            # Momentum reversal
            (dataframe['rsi_7'] > dataframe['rsi_14']) &
            (dataframe['rsi_14'] < 35),
            
            # Trend breakdown
            dataframe['close'] > dataframe['ema_13'],
            
            # At support
            dataframe['dist_from_support'] < 0.5,
            
            # MACD divergence
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macdhist'] > 0),
            
            # Volatility collapse
            dataframe['volatility_ratio'] < 0.5,
            
            # Bollinger Band squeeze
            dataframe['bb_width'] < dataframe['bb_width'].rolling(10).mean() * 0.6,
        ]
        
        dataframe.loc[
            np.logical_or.reduce(short_exit_conditions),
            'exit_short'
        ] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Advanced dynamic stoploss system
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) == 0:
                return self.stoploss
            
            current_candle = dataframe.iloc[-1]
            
            # Base stoploss adjustment based on signal strength
            signal_strength = (
                current_candle['long_strength'] if not trade.is_short 
                else current_candle['short_strength']
            )
            
            # Stronger signals get looser stops
            if signal_strength >= 8:
                base_adjustment = 0.85  # 15% looser
            elif signal_strength >= 6:
                base_adjustment = 0.92  # 8% looser
            else:
                base_adjustment = 1.15  # 15% tighter
            
            # Volatility adjustment
            volatility_ratio = current_candle.get('volatility_ratio', 1.0)
            
            if volatility_ratio > 2.5:
                volatility_adjustment = 0.7  # Much tighter in high vol
            elif volatility_ratio > 1.8:
                volatility_adjustment = 0.85  # Tighter
            elif volatility_ratio < 0.6:
                volatility_adjustment = 1.2  # Looser in low vol
            else:
                volatility_adjustment = 1.0
            
            # Time-based adjustment (tighter stops over time)
            trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600  # hours
            
            if trade_duration > 4:
                time_adjustment = 0.8  # Tighter after 4 hours
            elif trade_duration > 2:
                time_adjustment = 0.9  # Slightly tighter after 2 hours
            else:
                time_adjustment = 1.0
            
            # Profit-based adjustment
            if current_profit > 0.03:  # 3% profit
                profit_adjustment = 1.3  # Allow for more pullback
            elif current_profit > 0.01:  # 1% profit
                profit_adjustment = 1.1  # Slightly more room
            else:
                profit_adjustment = 1.0
            
            # Calculate final stoploss
            adjusted_stoploss = (
                self.stoploss * 
                base_adjustment * 
                volatility_adjustment * 
                time_adjustment * 
                profit_adjustment
            )
            
            # Ensure reasonable bounds
            return max(min(adjusted_stoploss, -0.02), -0.15)  # Between 2% and 15%
            
        except Exception:
            return self.stoploss
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Intelligent leverage based on signal strength and market conditions
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) == 0:
                return 3
            
            current_candle = dataframe.iloc[-1]
            
            # Base leverage on signal strength
            if side == 'long':
                signal_strength = current_candle.get('long_strength', 5)
            else:
                signal_strength = current_candle.get('short_strength', 5)
            
            # Signal strength leverage mapping
            if signal_strength >= 9:
                base_leverage = 8  # Highest conviction
            elif signal_strength >= 8:
                base_leverage = 6  # High conviction
            elif signal_strength >= 7:
                base_leverage = 4  # Medium conviction
            else:
                base_leverage = 2  # Low conviction
            
            # Volatility adjustment
            volatility_ratio = current_candle.get('volatility_ratio', 1.0)
            
            if volatility_ratio > 3.0:
                vol_multiplier = 0.4  # Very conservative in extreme volatility
            elif volatility_ratio > 2.0:
                vol_multiplier = 0.6  # Conservative in high volatility
            elif volatility_ratio > 1.5:
                vol_multiplier = 0.8  # Slightly conservative
            else:
                vol_multiplier = 1.0  # Normal leverage
            
            # Market regime adjustment
            if current_candle.get('high_volatility', False):
                regime_multiplier = 0.7  # More conservative in volatile markets
            elif current_candle.get('market_ranging', False):
                regime_multiplier = 0.8  # Slightly lower in ranging markets
            else:
                regime_multiplier = 1.0
            
            # Calculate final leverage
            final_leverage = int(base_leverage * vol_multiplier * regime_multiplier)
            
            # Ensure reasonable bounds
            return max(min(final_leverage, 10), 2)
            
        except Exception:
            return 3
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs):
        """
        Advanced exit management system
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) == 0:
                return None
                
            current_candle = dataframe.iloc[-1]
            
            # Emergency exits for extreme conditions
            if current_candle.get('volatility_ratio', 1.0) > 5.0:
                return "extreme_volatility_exit"
            
            # Signal strength deterioration
            signal_strength = (
                current_candle.get('long_strength', 0) if not trade.is_short 
                else current_candle.get('short_strength', 0)
            )
            
            if signal_strength <= 2:
                return "signal_deterioration"
            
            # Time-based exits for stale trades
            trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
            
            if trade_duration > 8 and current_profit < 0.005:  # 8 hours, less than 0.5% profit
                return "stale_trade_exit"
            
            # Profit protection exits
            if current_profit > 0.06:  # 6% profit
                if trade.is_short and current_candle.get('rsi_14', 50) < 25:
                    return "profit_protection_short"
                elif not trade.is_short and current_candle.get('rsi_14', 50) > 75:
                    return "profit_protection_long"
            
            # Market regime change
            if current_profit > 0.01:  # Only if in profit
                if (current_candle.get('market_ranging', False) and 
                    current_candle.get('bb_width', 1.0) < 0.5):
                    return "market_regime_change"
            
            return None
            
        except Exception:
            return None
    
    def informative_pairs(self):
        """
        Additional pairs for market context (if needed)
        """
        return []
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Final confirmation before trade entry
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) == 0:
                return False
            
            current_candle = dataframe.iloc[-1]
            
            # Check signal strength one more time
            if side == 'long':
                signal_strength = current_candle.get('long_strength', 0)
            else:
                signal_strength = current_candle.get('short_strength', 0)
            
            # Only allow trades with very high signal strength
            if signal_strength < 7:
                return False
            
            # Check for extreme market conditions
            if current_candle.get('volatility_ratio', 1.0) > 4.0:
                return False
            
            # Check current open trades (position sizing)
            open_trades = len(Trade.get_open_trades())
            if open_trades >= self.max_positions:
                return False
            
            return True
            
        except Exception:
            return False