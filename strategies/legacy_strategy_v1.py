

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


class legacy_strategy_v1(IStrategy):
    """
    This is a test strategy using the legacy function headers, which will be
    removed in a future update.
    Please do not use this as a template, but refer to user_data/strategy/sample_strategy.py
    for a uptodate version of this template.
    """


    minimal_roi = {
        "40": 0.0,
        "30": 0.01,
        "20": 0.02,
        "0": 0.04
    }


    stoploss = -0.10


    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """



        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['adx'] > 30) &
                (dataframe['tema'] > dataframe['tema'].shift(1)) &
                (dataframe['volume'] > 0)
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
                (dataframe['adx'] > 70) &
                (dataframe['tema'] < dataframe['tema'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
