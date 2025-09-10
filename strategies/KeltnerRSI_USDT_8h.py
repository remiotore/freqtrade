
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
import numpy as np  # noqa
import pandas as pd  # noqa

from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)

class KeltnerRSI_USDT_8h(IStrategy):

    timeframe = "8h"


    stoploss = -0.10

    minimal_roi = {"0": 100}

    plot_config = {
        "main_plot": {
            "kc_upperband" : {"color": "purple",'plotly': {'opacity': 0.4}},
            "kc_middleband" : {"color": "blue"},
            "kc_lowerband" : {"color": "purple",'plotly': {'opacity': 0.4}}
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "orange"},
                "hline": {"color": "grey","plotly": {"opacity": 0.4}}
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        keltner = qtpylib.keltner_channel(dataframe, window=20, atrs=1)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        hline = 55
        dataframe['hline'] = hline



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
