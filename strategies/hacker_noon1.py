import numpy as np
import pandas as pd

from freqtrade.strategy.interface import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class hacker_noon1(IStrategy):
    INTERFACE_VERSION = 2
    minimal_roi = {
        "0": 0.33564,
        "48": 0.07452,
        "202": 0.03109,
        "418": 0
    }
    stoploss = -0.33706
    trailing_stop = True
    trailing_stop_positive = 0.31965
    trailing_stop_positive_offset = 0.32407
    trailing_only_offset_is_reached = True







    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe, metadata):
        stoch = ta.STOCH(dataframe)
        rsi = ta.RSI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe))

        dataframe["slowk"] = stoch["slowk"]
        dataframe["rsi"] = rsi

        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["sar"] = ta.SAR(dataframe)
        dataframe["CDLHAMMER"] = ta.CDLHAMMER(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe, metadata):
        dataframe.loc[
            (
                (dataframe["rsi"] < 29) &
                (dataframe["bb_lowerband"] > dataframe["close"])
            ),
            "buy"
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe, metadata):
        dataframe.loc[
            (

                (dataframe["fisher"] > 0.11938)
            ),
            "sell"
        ] = 1
        return dataframe
