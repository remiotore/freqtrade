# https://github.com/mrjbq7/ta-lib/tree/master/docs/func_groups
# https://github.com/freqtrade/freqtrade-strategies/tree/master/user_data
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

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

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # ROI table:
    minimal_roi = {
        "0": 0.1641,
        "40": 0.05223,
        "87": 0.01553,
        "139": 0
    }

    # Stoploss:
    stoploss = -0.1529

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.13261
    trailing_stop_positive_offset = 0.1939
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe, fastperiod=24, slowperiod=56, signalperiod=11)
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
                    (dataframe['macd'] > dataframe['macdsignal'])
                    & (dataframe['cci'] <= -183)
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
                    (dataframe['macd'] < dataframe['macdsignal'])
                    & (dataframe['cci'] >= 325)
            ),
            'sell'] = 1

        return dataframe
