# By Remiotore (Jorge F. F.)
# Esper poder darte una buena vida algún día...

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
    #Trade,
    #Order,
    #PairLocks,
    #informative,
    #BooleanParameter,
    #CategoricalParameter,
    #DecimalParameter,
    #IntParameter,
    #RealParameter,
    #timeframe_to_minutes,
    #timeframe_to_next_date,
    #timeframe_to_prev_date,
    #merge_informative_pair,
    #stoploss_from_absolute,
    #stoploss_from_open,
)
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class Zaratustra(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '15m'
    can_short = True

    # ROI table:
    minimal_roi = {
        '0': 0.30,
        '109': 0.13,
        '267': 0.04,
        '428': 0.02,
        '720': 0.00,
    }

    # Stoploss:
    stoploss = -0.296

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.011
    trailing_stop_positive_offset = 0.071
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.05
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)

        dataframe['max'] = (dataframe['close'] == dataframe['close'].rolling(15).max()).astype(int)
        dataframe['min'] = (dataframe['close'] == dataframe['close'].rolling(15).min()).astype(int)

        dataframe['max_check'] = dataframe['max'].rolling(5).apply(lambda x: x.all(), raw=True).fillna(0).astype(int)
        dataframe['min_check'] = dataframe['min'].rolling(5).apply(lambda x: x.all(), raw=True).fillna(0).astype(int)

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['max_check'] == 0) &
                (dataframe['rsi'] > 70)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Max not detected + High RSI')

        dataframe.loc[
            (
                (dataframe['min_check'] == 0) &
                (dataframe['rsi'] < 30)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Min not detected + Low RSI')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['max_check'].shift(1) == 1) &
                (dataframe['max_check'] == 0)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'Max detected')
        
        dataframe.loc[
            (
                (dataframe['min_check'].shift(1) == 1) &
                (dataframe['min_check'] == 0)
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'Min detected')
        
        return dataframe