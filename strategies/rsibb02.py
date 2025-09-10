

import numpy as np # noqa
import pandas as pd # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class rsibb02(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    minimal_roi = {
    "0": 0.24140975952086036,
    "13": 0.049595065708988986,
    "51": 0.01046521346331895,
    "135": 0
    }

    stoploss = -0.12515406445006344

    ticker_interval = '1h'

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['rsi'] = ta.RSI(dataframe)



        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['rsi'] > 19) &
                (dataframe["close"] < dataframe['bb_lowerband'] )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
                (dataframe['rsi'] > 83) &
                (dataframe["close"] > dataframe['bb_middleband'] )
            ),
            'sell'] = 1

        return dataframe
