
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class AverageStrategy_2(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:
        buys and sells on crossovers - doesn't really perfom that well and its just a proof of concept
    """


    minimal_roi = {
        "0": 0.5
    }


    stoploss = -0.2

    ticker_interval = '4h'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        macd = ta.MACD(dataframe)

        dataframe['maShort'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['maMedium'] = ta.EMA(dataframe, timeperiod=21)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maShort'], dataframe['maMedium'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maMedium'], dataframe['maShort'])
            ),
            'sell'] = 1
        return dataframe
