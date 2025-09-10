
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class SmaRsiStrategy_59(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '1d'
    startup_candle_count: int = 25

    minimal_roi = {"0": 0.99}
    stoploss = -0.10
    trailing_stop = False
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = False



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)


        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['rsi'] > 50) &
                (qtpylib.crossed_above(dataframe['close'], dataframe['sma21']))
            ),
            'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['rsi'] < 50) &
                (qtpylib.crossed_below(dataframe['close'], dataframe['sma21']))
            ),
            'sell'] = 1
        return dataframe
