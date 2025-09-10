
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class MACDStrategy_3(IStrategy):


    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }


    stoploss = -0.3

    macd_fast_period = IntParameter(low=10, high=20, default=12, space='buy', optimize=True)
    macd_slow_period= IntParameter(low=20, high=35, default=26, space='buy', optimize=True)
    macd_signal_period = IntParameter(low=5, high=15, default=9, space='sell', optimize=True)

    ema_slow_period = IntParameter(low=50, high=150, default=100, space='buy', optimize=True)
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe, fastperiod=self.macd_fast_period.value, slowperiod=self.macd_slow_period.value, signalperiod=self.macd_signal_period.value)
        ema_slow = ta.EMA(dataframe, timeperiod=self.ema_slow_period.value)



        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['ema_slow'] = ema_slow
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['macd'] > 0) &
                (dataframe['close'] > dataframe['ema_slow']) &
                (dataframe['macd'] > dataframe['macdsignal'])
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
                (dataframe['close'] < dataframe['ema_slow'])
            ),
            'sell'] = 1
        return dataframe
