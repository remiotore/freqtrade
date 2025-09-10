


import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicator_helpers import fishers_inverse
from freqtrade.strategy.interface import IStrategy


class EMAMA(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    minimal_roi = {
        "40": 0.0,
        "30": 0.01,
        "20": 0.02,
        "0": 0.04
    }

    stoploss = -0.99

    ticker_interval = '15m'

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


        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)

        dataframe['ma4'] = ta.MA(dataframe, timeperiod=4)

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
                (dataframe['rsi'] < 35) &
                (dataframe['fastd'] < 35) &
                (dataframe['adx'] > 30) &
                (dataframe['plus_di'] > 0.5)
            ) |
            (
                (dataframe['adx'] > 65) &
                (dataframe['plus_di'] > 0.5)
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
                (
                    (qtpylib.crossed_above(dataframe['rsi'], 70)) |
                    (qtpylib.crossed_above(dataframe['fastd'], 70))
                ) &
                (dataframe['adx'] > 10) &
                (dataframe['minus_di'] > 0)
            ) |
            (
                (dataframe['adx'] > 70) &
                (dataframe['minus_di'] > 0.5)
            ),
            'sell'] = 1
        return dataframe
