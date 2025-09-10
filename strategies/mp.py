
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

import numpy as np
from scipy.signal import argrelextrema


class mp(IStrategy):

    ticker_interval = '5m'
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    process_only_new_candles = True

    minimal_roi = {
        "0": 1.0
    }

    stoploss = -0.1
    trailing_stop = False
    trailing_stop_positive = 0.32234
    trailing_stop_positive_offset = 0.40815
    trailing_only_offset_is_reached = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['min'] = dataframe.iloc[argrelextrema(dataframe.close.values, np.less_equal, order=5)[0]]['close']
        dataframe['max'] = dataframe.iloc[argrelextrema(dataframe.close.values, np.greater_equal, order=5)[0]]['close']
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['min'].isnull() == False
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['max'].isnull() == False
            ),
            'sell'] = 1
        return dataframe
