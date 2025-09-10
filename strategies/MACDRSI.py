

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicator_helpers import fishers_inverse
from freqtrade.strategy.interface import IStrategy


class MACDRSI(IStrategy):

    minimal_roi = {
        "40": 0.0,
        "30": 0.01,
        "20": 0.02,
        "0": 0.04
    }

    stoploss = -0.10

    ticker_interval = '15m'

    """ order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    } """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd1 = ta.MACD(dataframe, fastperiod=5, slowperiod=15)
        dataframe['macd1'] = macd1['macd']
        dataframe['macdsignal1'] = macd1['macdsignal']
        dataframe['macdhist1'] = macd1['macdhist']

        macd2 = ta.MACD(dataframe, fastperiod=12, slowperiod=26)
        dataframe['macd2'] = macd2['macd']
        dataframe['macdsignal2'] = macd2['macdsignal']
        dataframe['macdhist2'] = macd2['macdhist']

        dataframe['rsi'] = ta.RSI(dataframe)
        

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
                (dataframe['rsi'] > 30) &
                (dataframe['rsi'] < 70) &
                (dataframe['macdhist1'] > 0) &
                (dataframe['macdhist1'].shift(-1) < 0) &
                (dataframe['macdhist2'] < 0) &
                (dataframe['macdhist2'].shift(-1) < dataframe['macdhist2']) &
                (dataframe['macdhist2'].shift(-2) < dataframe['macdhist2'].shift(-1))
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
                qtpylib.crossed_below(dataframe['macd1'], dataframe['macdsignal1'])
            ),
            'sell'] = 1
        return dataframe
