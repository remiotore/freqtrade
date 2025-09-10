# By Remiotore (Jorge F. F.)
# Espero poder darte una buena vida algún día...

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
    informative,
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


class ZaratustraV5(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.05
    inf_times = ["5m", "15m",]

    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.296

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.071
    trailing_only_offset_is_reached = True
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        return 10.0
    
    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {},
            'subplots' : {
                "RSI": {
                    'rsi_30m': {'color' : 'lightgrey'},
                    'rsi_15m': {'color' : 'grey'},
                    'rsi_5m' : {'color' : 'darkgrey'},
                },
                "PDI": {
                    'pdi_30m': {'color' : 'lightgrey'},
                    'pdi_15m': {'color' : 'grey'},
                    'pdi_5m' : {'color' : 'darkgrey'},
                },
                "MDI": {
                    'mdi_30m': {'color' : 'lightgrey'},
                    'mdi_15m': {'color' : 'grey'},
                    'mdi_5m' : {'color' : 'darkgrey'},
                },
            }
        }
        return plot_config
    
    @informative('5m')
    @informative('15m')
    @informative('30m')
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bbu'] = bollinger['upper']
        dataframe['bbm'] = bollinger['mid']
        dataframe['bbl'] = bollinger['lower']

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # RSI
                (dataframe['rsi_30m'] > 50) &
                (dataframe['rsi_15m'] > 50) &
                (dataframe['rsi_5m']  > 50) &
                # Directional Indicator
                (dataframe['pdi_30m'] > 25) &
                (dataframe['pdi_15m'] > 25) &
                (dataframe['pdi_5m']  > 25) &
                # Bollinger Bands
                (dataframe['close_30m'] > dataframe['bbm_30m']) &
                (dataframe['close_15m'] > dataframe['bbm_15m']) &
                (dataframe['close_5m']  > dataframe['bbm_5m'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish trend')

        dataframe.loc[
            (
                # RSI
                (dataframe['rsi_30m'] < 50) &
                (dataframe['rsi_15m'] < 50) &
                (dataframe['rsi_5m']  < 50) &
                # Directional Indicator
                (dataframe['mdi_30m'] > 25) &
                (dataframe['mdi_15m'] > 25) &
                (dataframe['mdi_5m']  > 25) &
                # Bollinger Bands
                (dataframe['close_30m'] < dataframe['bbm_30m']) &
                (dataframe['close_15m'] < dataframe['bbm_15m']) &
                (dataframe['close_5m']  < dataframe['bbm_5m'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish trend')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe