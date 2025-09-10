
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from scipy.signal import argrelextrema
import numpy as np


class MinMaxF(IStrategy):

    minimal_roi = {
        "0":  10
    }

    stoploss = -0.05

    timeframe = '5m'

    trailing_stop = False

    process_only_new_candles = False

    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe_copy = dataframe.copy()
        frame_size = 250
        len_df = len(dataframe)
        dataframe['buy_signal'] = False
        dataframe['sell_signal'] = False
        lookback_size = 50

        for i in range(len_df):
            if i + frame_size < len_df:
                slice = dataframe_copy[i: i+frame_size]
                min_peaks = argrelextrema(
                    slice['close'].values, np.less, order=lookback_size)
                max_peaks = argrelextrema(
                    slice['close'].values, np.greater, order=lookback_size)


                if len(min_peaks[0]) and min_peaks[0][-1] == frame_size - 2:


                    dataframe.at[i + frame_size, 'buy_signal'] = True
                if len(max_peaks[0]) and max_peaks[0][-1] == frame_size - 2:


                    dataframe.at[i + frame_size, 'sell_signal'] = True

                if i + frame_size == len_df - 1:
                    pass























        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[
            (
                dataframe['buy_signal']
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                dataframe['sell_signal']
            ),
            'sell'] = 1
        return dataframe
