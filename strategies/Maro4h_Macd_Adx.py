
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class Maro4h_Macd_Adx(IStrategy):

    max_open_trades = 100
    stake_amount = 1000



    minimal_roi = {
        "0": 100
    }


    stoploss = -100

    timeframe = '4h'

    trailing_stop = False
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0.2

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):
        return [("BTC/USD", "4h"), ("ETH/USD", "4h")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']



        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe, timeperiod=14)





        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
            (qtpylib.crossed_above(dataframe['macdhist'],0)) &
            ((dataframe['adx'] >= 25) & (dataframe['di_plus'] > dataframe['di_minus']) &
             (dataframe['adx'] > dataframe['adx'].shift(1)))
            ),'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
            (qtpylib.crossed_below(dataframe['macdhist'],0)) &
            ((dataframe['adx'] >= 25) & (dataframe['di_minus'] > dataframe['di_plus']) &
             (dataframe['adx'] > dataframe['adx'].shift(1)) )
            ),'sell'] = 1

        return dataframe