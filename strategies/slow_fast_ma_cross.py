
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class slow_fast_ma_cross(IStrategy):


    minimal_roi = {
        "0": 0.5
    }


    stoploss = -0.2

    ticker_interval = '1h'

    plot_config = {
        'main_plot': {


            'maShort': {'color': 'red'},
            'maMedium': {'color': 'black'},

            'sar': {},
        }
   }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)

        dataframe['maShort'] = ta.MA(dataframe, timeperiod=50)
        dataframe['maMedium'] = ta.MA(dataframe, timeperiod=200)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
