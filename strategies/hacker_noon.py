import numpy as np
import pandas as pd

from freqtrade.strategy.interface import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class hacker_noon(IStrategy):
    INTERFACE_VERSION = 2
    minimal_roi = {
        "0": 0.05735,
        "10": 0.03845,
        "55": 0.01574,
        "157": 0
    }
    stoploss = -0.30566







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
                (dataframe["rsi"] < 20) &
                (dataframe["bb_lowerband"] > dataframe["close"])
            ),
            "buy"
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe, metadata):
        dataframe.loc[
            (

                (dataframe["fisher"] > -0.13955)
            ),
            "sell"
        ] = 1
        return dataframe
