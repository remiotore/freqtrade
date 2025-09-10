# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

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

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class BollingerMACD_V1(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    minimal_roi = {}

    stoploss = -0.99
    
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01

    trailing_stop = False

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