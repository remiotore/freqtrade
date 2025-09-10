
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class Long(IStrategy):
    """

    author@: Gert Wohlgemuth

    """


    minimal_roi = {
        "60":  0.05,
        "30":  0.06,
        "20":  0.07,
        "0":  0.08
    }


    stoploss = -0.15

    ticker_interval = 60

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=50)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['macd'] > 0) &
                (dataframe['cci'] <= 0.0)
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


                (dataframe['sar'] > dataframe['close']) &
                (dataframe['fisher_rsi'] > 0.3)
            ),
            'sell'] = 1
        return dataframe
