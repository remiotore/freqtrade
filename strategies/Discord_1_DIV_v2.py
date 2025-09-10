# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from functools import reduce
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional, Union, Any, Callable, Dict, List

from collections import deque
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair,informative,informative_decorator, stoploss_from_absolute)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib

class DIV_v2(IStrategy):
    minimal_roi = {
        "0": 0.05
    }

    stoploss = -0.05

    timeframe = '5m'
    startup_candle_count = 200
    process_only_new_candles = True

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Divergence
        dataframe = find_divergence(dataframe, 'rsi', 60)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["regular_bear_div_rsi"] == True) &
                (dataframe['rsi'] < 30) &
                (dataframe["volume"] > 0)
            ), 'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


def find_divergence(dataframe: DataFrame, indicator, lookback) -> DataFrame:

        #hp - highest price
        #lp - lowest price

        #hi - highest indicator
        #li - lowest indicator

        #hp_iv - indicator value on highest price
        #lp_iv - indicator value on lowest price

        #hi_pv - price value on highest indicator
        #li_pv - price value on lowest indicator

        # 1.Find min/max

        #  price
        dataframe['hp'] = dataframe['high'].rolling(lookback).max()
        dataframe['lp'] = dataframe['low'].rolling(lookback).min()
        #  indicator
        dataframe['hi'] = dataframe[indicator].rolling(lookback).max()
        dataframe['li'] = dataframe[indicator].rolling(lookback).min()

        # 2. Find a index where extremum appeard

        #  price
        index_of_max_price = dataframe['high'].rolling(window=lookback).apply(lambda x: x.idxmax(), raw=False)
        index_of_min_price = dataframe['high'].rolling(window=lookback).apply(lambda x: x.idxmin(), raw=False)
        #  indicator
        index_of_max_indicator = dataframe[indicator].rolling(window=lookback).apply(lambda x: x.idxmax(), raw=False)
        index_of_min_indicator = dataframe[indicator].rolling(window=lookback).apply(lambda x: x.idxmin(), raw=False)

        # 3. Use index to create new raw with data on pivot 
        
        #  price
        dataframe['hp_iv'] = dataframe.loc[index_of_max_price, indicator]
        dataframe['lp_iv'] = dataframe.loc[index_of_min_price, indicator]
        #  indicator
        dataframe['hi_pv'] = dataframe.loc[index_of_max_indicator, 'high']
        dataframe['li_pv'] = dataframe.loc[index_of_min_indicator, 'low']



        # 4. compare and find divs 

        dataframe[f'regular_bear_div_{indicator}'] = (dataframe['high'] > dataframe['hp']) & (dataframe['hp_iv'] > dataframe[indicator])
        dataframe[f'regular_bull_div_{indicator}'] = (dataframe['low'] > dataframe['lp']) & (dataframe['hp_iv'] > dataframe[indicator])
        dataframe[f'hidden_bear_div_{indicator}'] =  (dataframe['high'] < dataframe['hp']) & (dataframe['hp_iv'] < dataframe[indicator])
        dataframe[f'hidden_bull_div_{indicator}'] = (dataframe['low'] < dataframe['lp']) & (dataframe['hp_iv'] < dataframe[indicator])


        return dataframe