from functools import reduce
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
from technical.util import resample_to_interval, resampled_merge

class Darq(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '3m'
    # Define minimal ROI targets
    minimal_roi = {'60': 0.1, '30': 0.12, '0': 0.08}
    # Stoploss dynamically adjusted by ATR
    stoploss = -0.02  # Reduced SL for better risk management
    can_short = True
    # Trailing stop settings for profit maximization
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    process_only_new_candles = True
    startup_candle_count: int = 25  # Increased startup candles for better indicator accuracy
    pos_entry_adx = DecimalParameter(15, 40, decimals=1, default=20.0, space='buy')
    pos_exit_adx = DecimalParameter(15, 40, decimals=1, default=18.0, space='sell')
    adx_period = IntParameter(5, 30, default=14)
    ema_short_period = IntParameter(3, 10, default=5)  # Faster EMA
    ema_long_period = IntParameter(10, 30, default=20)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate technical indicators and add them to the dataframe."""
        # Compute ADX for different periods
        for val in self.adx_period.range:
            dataframe[f'adx_{val}'] = ta.ADX(dataframe, timeperiod=val)
        # Compute EMA short and long
        for val in self.ema_short_period.range:
            dataframe[f'ema_short_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.ema_long_period.range:
            dataframe[f'ema_long_{val}'] = ta.EMA(dataframe, timeperiod=val)
        # Compute Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        # Compute additional indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)  # ATR for dynamic stop loss
        dataframe['obv'] = ta.OBV(dataframe, dataframe['volume'])  # OBV for volume trend analysis
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # RSI to filter overbought/oversold conditions
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        # Resample to higher timeframe and merge
        self.resample_interval = timeframe_to_minutes(self.timeframe) * 5
        dataframe_long = resample_to_interval(dataframe, self.resample_interval)
        dataframe_long['sma'] = ta.SMA(dataframe_long, timeperiod=50, price='close')
        dataframe = resampled_merge(dataframe, dataframe_long, fill_na=True)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions for long and short positions."""
        conditions_long = []
        conditions_short = []
        # Ensure price is above/below higher timeframe SMA
        conditions_long.append(dataframe['close'] > dataframe[f'resample_{self.resample_interval}_sma'])
        conditions_short.append(dataframe['close'] < dataframe[f'resample_{self.resample_interval}_sma'])
        # EMA crossover conditions
        conditions_long.append(qtpylib.crossed_above(dataframe[f'ema_short_{self.ema_short_period.value}'], dataframe[f'ema_long_{self.ema_long_period.value}']))
        conditions_short.append(qtpylib.crossed_below(dataframe[f'ema_short_{self.ema_short_period.value}'], dataframe[f'ema_long_{self.ema_long_period.value}']))
        # ADX filter to allow more trades
        conditions_long.append(dataframe['adx_14'] > self.pos_entry_adx.value)
        conditions_short.append(dataframe['adx_14'] > self.pos_entry_adx.value)
        # OBV filter to confirm trend direction
        conditions_long.append(dataframe['obv'] > dataframe['obv'].rolling(5).mean())
        conditions_short.append(dataframe['obv'] < dataframe['obv'].rolling(5).mean())
        # RSI filter to allow more trades
        conditions_long.append(dataframe['rsi'] < 80)
        conditions_short.append(dataframe['rsi'] > 20)
        # MACD filter to confirm momentum
        conditions_long.append(dataframe['macd'] > dataframe['macdsignal'])
        conditions_short.append(dataframe['macd'] < dataframe['macdsignal'])
        # Assign entry signals
        dataframe.loc[reduce(lambda x, y: x & y, conditions_long), 'enter_long'] = 1
        dataframe.loc[reduce(lambda x, y: x & y, conditions_short), 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions for open positions."""
        conditions_close = []
        # Exit when ADX drops below threshold
        conditions_close.append(dataframe[f'adx_{self.adx_period.value}'] < self.pos_exit_adx.value)
        # Assign exit signals
        dataframe.loc[reduce(lambda x, y: x & y, conditions_close), 'exit_long'] = 1
        dataframe.loc[reduce(lambda x, y: x & y, conditions_close), 'exit_short'] = 1
        return dataframe