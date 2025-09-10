
import numpy as np  # noqa
import pandas as pd  # noqa
from functools import reduce
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class RSI_hyper(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '1d'
    startup_candle_count: int = 25

    minimal_roi = {
        "60": 0.01,
        "30": 0.03,
        "20": 0.04,
        "0": 0.05
    }

    stoploss = -0.10
    trailing_stop = False
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = False


    buy_rsi = IntParameter(5, 30, default=20, space="buy")
    sell_rsi = IntParameter(75, 95, default=85, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['RSI'] = ta.RSI(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append((dataframe['RSI'] < self.buy_rsi.value))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append((dataframe['RSI'] > self.sell_rsi.value))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe