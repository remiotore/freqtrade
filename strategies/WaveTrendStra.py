
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class WaveTrendStra(IStrategy):
    """

    author@: Gert Wohlgemuth

    just a skeleton

    """



    minimal_roi = {
        "0": 100   #disable roi
    }

    stoploss = -0.25

    timeframe = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ap = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3

        esa = ta.EMA(ap, 10)
        d = ta.EMA(abs(ap - esa), 10)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.EMA(ci, 21)

        dataframe["wt1"] = tci
        dataframe["wt2"] = ta.SMA(dataframe["wt1"], 4)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_above(dataframe["wt1"], dataframe["wt2"]))
            ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['wt1'], dataframe['wt2']))
            ,'sell'] = 1
        return dataframe
