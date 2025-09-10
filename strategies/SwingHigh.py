

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class SwingHigh(IStrategy):


    minimal_roi = {"0": 0.16035, "23": 0.03218, "54": 0.01182, "173": 0}

    stoploss = -0.22274


    trailing_stop = True
    trailing_stop_positive = 0.08
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    ticker_interval = "30m"

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]



        dataframe["cci-buy"] = ta.CCI(dataframe, timeperiod=13)
        dataframe["cci-sell"] = ta.CCI(dataframe, timeperiod=76)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["macd"] > dataframe["macdsignal"])
                & (dataframe["cci-buy"] <= -188.0)
                & (dataframe["volume"] > 0)
            ),
            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["macd"] < dataframe["macdsignal"])
                & (dataframe["cci-sell"] >= 231.0)
                & (dataframe["volume"] > 0)
            ),
            "sell",
        ] = 1

        return dataframe
