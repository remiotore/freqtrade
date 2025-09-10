import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import IStrategy
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class BollingerMACD_V3(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True

    minimal_roi = {
        "0": 0.28,
        "22": 0.081,
        "68": 0.039,
        "90": 0
    }

    stoploss = -0.35
    
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01

    trailing_stop = True
    trailing_stop_positive = 0.12
    trailing_stop_positive_offset = 0.147
    trailing_only_offset_is_reached = True

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    @property
    def plot_config(self):
        return {
            'main_plot': {
            },
            'subplots': {
                'MACD': {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
            }
        }

    def informative_pairs(self):
        return []

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return 10.0
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband'])) &
                (dataframe['macd']       > 0) &
                (dataframe['macdsignal'] > 0) &
                (dataframe['volume']     > 0)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband'])) &
                (dataframe['macd']       < 0) &
                (dataframe['macdsignal'] < 0) &
                (dataframe['volume']     > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['bb_upperband'])) &
                (dataframe['macd']       > 0) &
                (dataframe['macdsignal'] > 0) &
                (dataframe['volume']     > 0)
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['bb_lowerband'])) &
                (dataframe['macd']       < 0) &
                (dataframe['macdsignal'] < 0) &
                (dataframe['volume']     > 0)
            ),
            'exit_short'] = 1

        return dataframe