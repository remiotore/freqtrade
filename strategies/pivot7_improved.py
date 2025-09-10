from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
from freqtrade.persistence import Trade
from datetime import datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta

class Pivot7Improved(IStrategy):
    # Risk Management - Much more conservative
    stoploss = -0.08  # 8% stoploss instead of 50%
    can_short = True
    trailing_stop = True
    trailing_stop_positive = 0.02  # Start trailing at 2% profit
    trailing_stop_positive_offset = 0.03  # 3% trailing distance
    
    # Timeframe optimization
    timeframe = '5m'
    
    # ROI table for taking profits
    minimal_roi = {
        "0": 0.15,   # 15% profit target
        "30": 0.08,  # 8% after 30 minutes
        "60": 0.05,  # 5% after 1 hour
        "120": 0.02  # 2% after 2 hours
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Pivot parameters
        pivot_length = 20  # Reduced for faster signals
        
        # RSI indicators with different periods
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=21)
        
        # Trend filtering with EMA
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=50)
        
        # MACD for momentum confirmation
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # ATR for volatility-based stops
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Improved pivot calculation
        dataframe['pivot_high'] = dataframe['high'].rolling(
            window=pivot_length, center=True
        ).max().shift(-pivot_length//2)
        
        dataframe['pivot_low'] = dataframe['low'].rolling(
            window=pivot_length, center=True
        ).min().shift(-pivot_length//2)
        
        # Identify actual pivot points (local extremes)
        dataframe['is_pivot_high'] = (
            (dataframe['high'] == dataframe['pivot_high']) & 
            (dataframe['high'] == dataframe['high'].rolling(pivot_length, center=True).max())
        )
        
        dataframe['is_pivot_low'] = (
            (dataframe['low'] == dataframe['pivot_low']) & 
            (dataframe['low'] == dataframe['low'].rolling(pivot_length, center=True).min())
        )
        
        # Support and resistance levels
        dataframe['resistance'] = dataframe.loc[dataframe['is_pivot_high'], 'high'].fillna(method='ffill')
        dataframe['support'] = dataframe.loc[dataframe['is_pivot_low'], 'low'].fillna(method='ffill')
        
        # Volume confirmation (if available)
        if 'volume' in dataframe.columns:
            dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        else:
            dataframe['volume_ratio'] = 1
        
        # Market structure - Higher highs, Lower lows
        dataframe['higher_high'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) & 
            (dataframe['high'].shift(1) > dataframe['high'].shift(2))
        )
        
        dataframe['lower_low'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) & 
            (dataframe['low'].shift(1) < dataframe['low'].shift(2))
        )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Long Entry Conditions
        long_conditions = [
            # RSI momentum
            qtpylib.crossed_above(dataframe['rsi_fast'], dataframe['rsi_slow']),
            # RSI not overbought
            dataframe['rsi_fast'] < 70,
            dataframe['rsi_slow'] < 65,
            # Trend alignment
            dataframe['ema_fast'] > dataframe['ema_slow'],
            dataframe['close'] > dataframe['ema_trend'],
            # MACD confirmation
            dataframe['macd'] > dataframe['macdsignal'],
            dataframe['macdhist'] > 0,
            # Price above support
            dataframe['close'] > dataframe['support'],
            # Volume confirmation
            dataframe['volume_ratio'] > 0.8,
            # Market structure
            ~dataframe['lower_low']  # Not in a downtrend
        ]
        
        dataframe.loc[
            np.logical_and.reduce(long_conditions),
            'enter_long'
        ] = 1
        
        # Short Entry Conditions
        short_conditions = [
            # RSI momentum
            qtpylib.crossed_below(dataframe['rsi_fast'], dataframe['rsi_slow']),
            # RSI not oversold
            dataframe['rsi_fast'] > 30,
            dataframe['rsi_slow'] > 35,
            # Trend alignment
            dataframe['ema_fast'] < dataframe['ema_slow'],
            dataframe['close'] < dataframe['ema_trend'],
            # MACD confirmation
            dataframe['macd'] < dataframe['macdsignal'],
            dataframe['macdhist'] < 0,
            # Price below resistance
            dataframe['close'] < dataframe['resistance'],
            # Volume confirmation
            dataframe['volume_ratio'] > 0.8,
            # Market structure
            ~dataframe['higher_high']  # Not in an uptrend
        ]
        
        dataframe.loc[
            np.logical_and.reduce(short_conditions),
            'enter_short'
        ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Long Exit Conditions
        long_exit_conditions = [
            # RSI momentum reversal
            qtpylib.crossed_below(dataframe['rsi_fast'], dataframe['rsi_slow']),
            # RSI overbought
            (dataframe['rsi_fast'] > 75) | (dataframe['rsi_slow'] > 70),
            # MACD divergence
            dataframe['macdhist'] < dataframe['macdhist'].shift(1),
            # Trend change
            dataframe['ema_fast'] < dataframe['ema_slow']
        ]
        
        dataframe.loc[
            np.logical_or.reduce(long_exit_conditions),
            'exit_long'
        ] = 1
        
        # Short Exit Conditions
        short_exit_conditions = [
            # RSI momentum reversal
            qtpylib.crossed_above(dataframe['rsi_fast'], dataframe['rsi_slow']),
            # RSI oversold
            (dataframe['rsi_fast'] < 25) | (dataframe['rsi_slow'] < 30),
            # MACD divergence
            dataframe['macdhist'] > dataframe['macdhist'].shift(1),
            # Trend change
            dataframe['ema_fast'] > dataframe['ema_slow']
        ]
        
        dataframe.loc[
            np.logical_or.reduce(short_exit_conditions),
            'exit_short'
        ] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic stoploss based on ATR
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return self.stoploss
        
        current_candle = dataframe.iloc[-1]
        atr = current_candle['atr']
        
        if trade.is_short:
            # For shorts, stoploss is above entry
            atr_stop = atr * 2 / current_rate  # 2x ATR as percentage
            return min(-0.03, -atr_stop)  # At least 3% stoploss
        else:
            # For longs, stoploss is below entry
            atr_stop = atr * 2 / current_rate  # 2x ATR as percentage
            return min(-0.03, -atr_stop)  # At least 3% stoploss
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Conservative leverage for futures trading
        """
        return 5  # Much more conservative 5x instead of 20x
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs):
        """
        Custom exit logic for additional profit protection
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return None
            
        current_candle = dataframe.iloc[-1]
        
        # Take partial profits at key levels
        if current_profit > 0.10:  # 10% profit
            return "profit_10_percent"
        
        # Exit on extreme RSI
        if trade.is_short and current_candle['rsi_fast'] < 20:
            return "rsi_oversold_exit"
        elif not trade.is_short and current_candle['rsi_fast'] > 80:
            return "rsi_overbought_exit"
        
        return None