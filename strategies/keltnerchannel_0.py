
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
import numpy as np  # noqa
import pandas as pd  # noqa


class keltnerchannel_0(IStrategy):
    timeframe = "6h"

    stoploss = -0.254
    minimal_roi = {"0": 100}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        keltner = qtpylib.keltner_channel(dataframe, window=16, atrs=1)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        dataframe['hline'] = 61
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['close'], dataframe['kc_upperband'])
            & (dataframe["rsi"] > dataframe['hline'])
            ),

            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['close'], dataframe['kc_middleband'])),

            "sell",
        ] = 1
        return dataframe
