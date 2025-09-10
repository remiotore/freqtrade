from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from datetime import datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta

class VolatimImproved(IStrategy):
    # Strategy parameters
    stoploss = -0.08  # Improved from -0.1 to -0.08 for better risk management
    can_short = True
    trailing_stop = True
    trailing_stop_positive = 0.015  # 1.5%
    trailing_stop_positive_offset = 0.025  # 2.5%
    
    timeframe = '5m'  # Same as original
    
    # ROI table for systematic profit taking
    minimal_roi = {
        "0": 0.12,   # 12% initial target
        "15": 0.08,  # 8% after 15 minutes
        "30": 0.05,  # 5% after 30 minutes
        "60": 0.03,  # 3% after 1 hour
        "120": 0.015 # 1.5% after 2 hours
    }
    
    # Strategy-specific parameters
    volatility_window = 20
    returns_window = 10
    rsi_period = 14
    volume_window = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced volatility-based indicators with improved signal generation
        """
        
        # Basic price action
        dataframe['returns'] = dataframe['close'].pct_change()
        
        # Volatility calculations (core of Volatim strategy)
        dataframe['returns_roll_mean'] = dataframe['returns'].rolling(
            window=self.returns_window
        ).mean()
        
        dataframe['returns_roll_std'] = dataframe['returns'].rolling(
            window=self.returns_window
        ).std()
        
        # Volatility bands (similar to Bollinger Bands but for returns)
        dataframe['returns_roll_mean_cumsum'] = dataframe['returns_roll_mean'].cumsum()
        
        # Upper and lower volatility thresholds
        volatility_multiplier = 2.0
        dataframe['returns_roll_mean_cumsum_upper'] = (
            dataframe['returns_roll_mean_cumsum'] + 
            (volatility_multiplier * dataframe['returns_roll_std'])
        )
        dataframe['returns_roll_mean_cumsum_lower'] = (
            dataframe['returns_roll_mean_cumsum'] - 
            (volatility_multiplier * dataframe['returns_roll_std'])
        )
        
        # RSI for momentum
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        
        # Rolling min/max for volatility extremes
        dataframe['rolling_min'] = dataframe['low'].rolling(
            window=self.volatility_window
        ).min()
        dataframe['rolling_max'] = dataframe['high'].rolling(
            window=self.volatility_window
        ).max()
        
        # Local extremes detection
        dataframe['local_min'] = (
            dataframe['low'] == dataframe['rolling_min']
        ).astype(int)
        dataframe['local_max'] = (
            dataframe['high'] == dataframe['rolling_max']
        ).astype(int)
        
        # Bitcoin correlation (if trading altcoins)
        # Note: This would need actual BTC data feed in practice
        # For now, we'll simulate market correlation
        dataframe['btc_returns'] = dataframe['returns']  # Placeholder
        dataframe['btc_returns_roll_mean'] = dataframe['btc_returns'].rolling(
            window=self.returns_window
        ).mean()
        dataframe['btc_returns_roll_mean_cumsum'] = dataframe['btc_returns_roll_mean'].cumsum()
        
        # Enhanced volatility indicators
        # ATR for dynamic volatility measurement
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
        
        # Volatility ratio (current vs historical average)
        dataframe['volatility_ratio'] = (
            dataframe['atr_pct'] / dataframe['atr_pct'].rolling(50).mean()
        )
        
        # Bollinger Bands for additional volatility context
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_lower'] = bb['lowerband']
        dataframe['bb_middle'] = bb['middleband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # Volume analysis
        if 'volume' in dataframe.columns:
            dataframe['volume_sma'] = dataframe['volume'].rolling(self.volume_window).mean()
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        else:
            dataframe['volume_ratio'] = 1
        
        # Market regime detection
        dataframe['trend_strength'] = abs(
            dataframe['close'].rolling(20).mean() - dataframe['close'].rolling(50).mean()
        ) / dataframe['atr']
        
        # Volatility breakout signals
        dataframe['vol_breakout_up'] = (
            (dataframe['volatility_ratio'] > 1.2) & 
            (dataframe['close'] > dataframe['bb_middle']) &
            (dataframe['returns'] > dataframe['returns_roll_mean_cumsum_upper'])
        ).astype(int)
        
        dataframe['vol_breakout_down'] = (
            (dataframe['volatility_ratio'] > 1.2) & 
            (dataframe['close'] < dataframe['bb_middle']) &
            (dataframe['returns'] < dataframe['returns_roll_mean_cumsum_lower'])
        ).astype(int)
        
        # Mean reversion signals
        dataframe['mean_reversion_up'] = (
            (dataframe['close'] <= dataframe['bb_lower']) &
            (dataframe['rsi'] < 30) &
            (dataframe['local_min'] == 1)
        ).astype(int)
        
        dataframe['mean_reversion_down'] = (
            (dataframe['close'] >= dataframe['bb_upper']) &
            (dataframe['rsi'] > 70) &
            (dataframe['local_max'] == 1)
        ).astype(int)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Long Entry: Volatility breakout + momentum
        long_conditions = [
            # Primary volatility signal
            (dataframe['vol_breakout_up'] == 1) |
            (dataframe['mean_reversion_up'] == 1),
            
            # Momentum confirmation
            dataframe['rsi'] > 25,  # Not oversold
            dataframe['rsi'] < 75,  # Not overbought
            
            # Volatility environment
            dataframe['volatility_ratio'] > 0.8,  # Some volatility present
            dataframe['volatility_ratio'] < 3.0,  # Not extremely volatile
            
            # Volume confirmation
            dataframe['volume_ratio'] > 0.7,
            
            # Returns momentum
            dataframe['returns_roll_mean'] > dataframe['returns_roll_mean'].shift(1),
            
            # Market structure
            dataframe['close'] > dataframe['rolling_min'],
        ]
        
        dataframe.loc[
            np.logical_and.reduce(long_conditions),
            'enter_long'
        ] = 1
        
        # Short Entry: Volatility breakout down + momentum
        short_conditions = [
            # Primary volatility signal
            (dataframe['vol_breakout_down'] == 1) |
            (dataframe['mean_reversion_down'] == 1),
            
            # Momentum confirmation
            dataframe['rsi'] > 25,  # Not oversold
            dataframe['rsi'] < 75,  # Not overbought
            
            # Volatility environment
            dataframe['volatility_ratio'] > 0.8,  # Some volatility present
            dataframe['volatility_ratio'] < 3.0,  # Not extremely volatile
            
            # Volume confirmation
            dataframe['volume_ratio'] > 0.7,
            
            # Returns momentum
            dataframe['returns_roll_mean'] < dataframe['returns_roll_mean'].shift(1),
            
            # Market structure
            dataframe['close'] < dataframe['rolling_max'],
        ]
        
        dataframe.loc[
            np.logical_and.reduce(short_conditions),
            'enter_short'
        ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Long Exit: Volatility collapse or momentum reversal
        long_exit_conditions = [
            # Volatility collapse
            (dataframe['volatility_ratio'] < 0.6) |
            
            # RSI overbought
            (dataframe['rsi'] > 80) |
            
            # Returns momentum reversal
            (dataframe['returns_roll_mean'] < dataframe['returns_roll_mean'].shift(2)) |
            
            # Price at resistance
            (dataframe['close'] >= dataframe['rolling_max']) |
            
            # Bollinger Band squeeze
            (dataframe['bb_width'] < dataframe['bb_width'].rolling(10).mean() * 0.5)
        ]
        
        dataframe.loc[
            np.logical_or.reduce(long_exit_conditions),
            'exit_long'
        ] = 1
        
        # Short Exit: Volatility collapse or momentum reversal
        short_exit_conditions = [
            # Volatility collapse
            (dataframe['volatility_ratio'] < 0.6) |
            
            # RSI oversold
            (dataframe['rsi'] < 20) |
            
            # Returns momentum reversal
            (dataframe['returns_roll_mean'] > dataframe['returns_roll_mean'].shift(2)) |
            
            # Price at support
            (dataframe['close'] <= dataframe['rolling_min']) |
            
            # Bollinger Band squeeze
            (dataframe['bb_width'] < dataframe['bb_width'].rolling(10).mean() * 0.5)
        ]
        
        dataframe.loc[
            np.logical_or.reduce(short_exit_conditions),
            'exit_short'
        ] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic stoploss based on volatility
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return self.stoploss
        
        current_candle = dataframe.iloc[-1]
        
        # Volatility-adjusted stoploss
        volatility_multiplier = max(1.0, current_candle['volatility_ratio'])
        base_stoploss = self.stoploss
        
        # Tighter stops in high volatility
        if current_candle['volatility_ratio'] > 2.0:
            adjusted_stoploss = base_stoploss * 0.7  # Tighter stop
        elif current_candle['volatility_ratio'] > 1.5:
            adjusted_stoploss = base_stoploss * 0.85
        else:
            adjusted_stoploss = base_stoploss
        
        return max(adjusted_stoploss, -0.12)  # Maximum 12% loss
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Dynamic leverage based on volatility
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) == 0:
                return 3  # Conservative default
            
            current_candle = dataframe.iloc[-1]
            volatility_ratio = current_candle.get('volatility_ratio', 1.0)
            
            # Lower leverage in high volatility
            if volatility_ratio > 2.0:
                return 2  # Very conservative in high vol
            elif volatility_ratio > 1.5:
                return 3  # Moderate leverage
            elif volatility_ratio > 1.0:
                return 5  # Standard leverage
            else:
                return 7  # Higher leverage in low volatility
                
        except Exception:
            return 3  # Safe default
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs):
        """
        Enhanced exit logic for volatility-based trading
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) == 0:
                return None
                
            current_candle = dataframe.iloc[-1]
            
            # Exit on volatility collapse (major signal deterioration)
            if current_candle['volatility_ratio'] < 0.4:
                return "volatility_collapse"
            
            # Exit on extreme RSI with profit
            if current_profit > 0.02:  # At least 2% profit
                if trade.is_short and current_candle['rsi'] < 15:
                    return "rsi_extreme_short_exit"
                elif not trade.is_short and current_candle['rsi'] > 85:
                    return "rsi_extreme_long_exit"
            
            # Take profits on strong volatility spikes
            if current_profit > 0.05 and current_candle['volatility_ratio'] > 3.0:
                return "high_volatility_profit_take"
            
            return None
            
        except Exception:
            return None