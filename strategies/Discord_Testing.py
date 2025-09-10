
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List, Optional, Union
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa

from datetime import datetime, timedelta
from freqtrade.persistence import Trade

# UltimateMomentumIndicator

import numpy as np  # noqa
import pandas as pd  # noqa

from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# ReinforcedQuickie
from pandas import DataFrame, DatetimeIndex, merge, Series

# FisherHull
from technical.indicators import hull_moving_average

# Supertrend
import logging
from numpy.lib import math


# import random

class Testing(IStrategy):

    INTERFACE_VERSION: int = 3

    

   

    minimal_roi = {
        "0": 100 # inactive
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.99

   
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

   

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }




    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

 

        return dataframe

    def test(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        (dataframe['ha_open'] < dataframe['ha_close'])
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []        
        
        enter_cond_1 = self.test(dataframe, metadata)
        conditions.append(enter_cond_1)
        dataframe.loc[enter_cond_1, 'exter_tag'] = 'Default entry 1'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """     
        
        conditions = []

        exit_cond_1 = (
                (dataframe['ha_low'] < dataframe['ha_open']) 
            )

        conditions.append(exit_cond_1)
        dataframe.loc[exit_cond_1, 'exit_tag'] = 'Default exit 1'    


        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1






       