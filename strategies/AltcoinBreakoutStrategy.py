import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union
import talib.abstract as ta
from technical import qtpylib
from functools import reduce

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

class AltcoinBreakoutStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False
    
    minimal_roi = {"0": 0.15, "30": 0.08, "60": 0.03, "120": 0}
    stoploss = -0.12
    trailing_stop = True
    trailing_stop_positive = 0.03

    buy_bb_width = DecimalParameter(0.02, 0.1, default=0.05, space='buy')
    buy_adx_threshold = IntParameter(25, 50, default=30, space='buy')
    buy_macd_hist_threshold = DecimalParameter(-0.01, 0.05, default=0, space='buy')
    buy_volume_factor = DecimalParameter(1.0, 3.0, default=1.5, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_width'] = (bollinger['upper'] - bollinger['lower']) / bollinger['mid']
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd_hist'] = macd['macdhist']
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Volume
        dataframe['volume_sma20'] = ta.SMA(dataframe['volume'], timeperiod=20)
        
        # EMA Ribbon
        for period in [3, 5, 8, 13]:
            dataframe[f'ema_{period}'] = ta.EMA(dataframe, timeperiod=period)
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe['bb_width'] < self.buy_bb_width.value,
            dataframe['macd_hist'] > self.buy_macd_hist_threshold.value,
            dataframe['adx'] > self.buy_adx_threshold.value,
            dataframe['volume'] > (dataframe['volume_sma20'] * self.buy_volume_factor.value),
            (dataframe['ema_3'] > dataframe['ema_5']) &
            (dataframe['ema_5'] > dataframe['ema_8']) &
            (dataframe['ema_8'] > dataframe['ema_13'])
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] >= dataframe['bb_upper']) |  # Corrected column
                (dataframe['macd_hist'] < 0)
            ),
            'exit_long'
        ] = 1
        return dataframe