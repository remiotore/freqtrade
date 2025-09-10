
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa

class CMCWinner_420(IStrategy):
    """
    This is a test strategy to inspire you.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """


    minimal_roi = {
        "40": 0.0,
        "30": 0.02,
        "20": 0.03,
        "0": 0.05
    }


    stoploss = -0.05

    timeframe = '15m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        dataframe['cci'] = ta.CCI(dataframe)

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['cmo'] = ta.CMO(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['cci'].shift(1) < -100) &
                (dataframe['mfi'].shift(1) < 20) &
                (dataframe['cmo'].shift(1) < -50)
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
                (dataframe['cci'].shift(1) > 100) &
                (dataframe['mfi'].shift(1) > 80) &
                (dataframe['cmo'].shift(1) > 50)
            ),
            'sell'] = 1
        return dataframe
