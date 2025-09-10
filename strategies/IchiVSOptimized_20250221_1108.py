# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from functools import reduce
from datetime import datetime
import numpy as np
from freqtrade.strategy import merge_informative_pair, stoploss_from_open
from typing import Optional

class IchiVSOptimized(IStrategy):
    timeframe = '5m'
    startup_candle_count = 96
    process_only_new_candles = True

    can_short = True
    leverage_value = 5

    minimal_roi = {
        "0": 0.05,
        "10": 0.03,
        "30": 0.015,
        "60": 0
    }
    stoploss = -0.275

    trailing_stop = False  
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula os indicadores necessários para a estratégia.
        """
        ha = qtpylib.heikinashi(dataframe.copy())
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        
        # Calcular EMAs
        for period in [5, 15, 30, 60, 120, 240, 360, 480]:
            dataframe[f'trend_close_{period}m'] = ta.EMA(dataframe['ha_close'], timeperiod=period)

        # Fan Magnitude
        dataframe['fan_magnitude'] = dataframe['trend_close_60m'] / dataframe['trend_close_480m']
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe.fillna(0))

        return dataframe.fillna(0)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define regras de entrada para long e short.
        """
        long_conditions = [
            dataframe['trend_close_5m'] > dataframe['trend_close_60m'],
            dataframe['fan_magnitude'] > 1,
            dataframe['fan_magnitude_gain'] > 1.002,
            dataframe['atr'] > 0.0015
        ]
        
        short_conditions = [
            dataframe['trend_close_5m'] < dataframe['trend_close_60m'],
            dataframe['fan_magnitude'] < 1,
            dataframe['fan_magnitude_gain'] < 0.998,
            dataframe['atr'] > 0.0015
        ]
        
        dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 1
        dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define regras de saída.
        """
        dataframe.loc[qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe['trend_close_120m']), 'exit_long'] = 1
        dataframe.loc[qtpylib.crossed_above(dataframe['trend_close_5m'], dataframe['trend_close_120m']), 'exit_short'] = 1
        
        return dataframe
