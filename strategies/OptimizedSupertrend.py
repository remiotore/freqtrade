


import os
import sys

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib

from numba import njit

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path);
sys.path.append(script_path)


class OptimizedSupertrend(IStrategy):


    minimal_roi = {
        "0": 100
    }

    stoploss = -0.1

    trailing_stop = False
    trailing_stop_positive = 0.032
    trailing_stop_positive_offset = 0.084
    trailing_only_offset_is_reached = True

    timeframe = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        entries, exits = supertrend_strategy(dataframe)
        dataframe['buy'] = entries
        dataframe['sell'] = exits

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['buy'] == 1, 'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['sell'] == 1, 'exit_long'] = 1

        return dataframe


@njit
def get_final_bands_nb(close, upper, lower):
    trend = np.full(close.shape, np.nan)
    dir_ = np.full(close.shape, 1)
    long = np.full(close.shape, np.nan)
    short = np.full(close.shape, np.nan)

    for i in range(1, close.shape[0]):
        if close[i] > upper[i - 1]:
            dir_[i] = 1
        elif close[i] < lower[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if dir_[i] < 0 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]

        if dir_[i] > 0:
            trend[i] = long[i] = lower[i]
        else:
            trend[i] = short[i] = upper[i]

    return trend, dir_, long, short


def get_med_price(high, low):
    return (high + low) / 2


def get_basic_bands(med_price, atr, multiplier):
    matr = multiplier * atr
    upper = med_price + matr
    lower = med_price - matr
    return upper, lower


def faster_supertrend_talib(high, low, close, period=7, multiplier=3):







    avg_price = talib.MEDPRICE(high, low)
    atr = talib.ATR(high, low, close, period)
    upper, lower = get_basic_bands(avg_price, atr, multiplier)

    close = np.array(close)
    upper = np.array(upper)
    lower = np.array(lower)
    return get_final_bands_nb(close, upper, lower)

def supertrend_strategy(ohlcv, period=4, multiplier=3.5):
    high = ohlcv.high
    low = ohlcv.low
    close = ohlcv.close
    trend, dir_, long, short = faster_supertrend_talib(high, low, close, period, multiplier)

    long = np.where(long > 0, True, False)
    short = np.where(short > 0, True, False)

    return long, short
