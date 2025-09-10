
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta


class MACD(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:

        uptrend definition:
            MACD above MACD signal
            and CCI < -50

        downtrend definition:
            MACD below MACD signal
            and CCI > 100

    """


    minimal_roi = {
        "0": 0.1858737982036954,
        "27": 0.02524445335072633,
        "80": 0.013912072159850514,
        "154": 0
    }


    stoploss = -0.27408960533563953

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['cci'] <= -205.0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['cci'] >= 360.0)
            ),
            'sell'] = 1

        return dataframe