
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)


class TheSimpleStrategy(IStrategy):



    minimal_roi = {
        "0": 0.01
    }


    stoploss = -0.25

    timeframe = '5m'

    macd_fast_period = IntParameter(low=10, high=20, default=12, space='buy', optimize=True)
    macd_slow_period= IntParameter(low=20, high=35, default=26, space='buy', optimize=True)
    macd_signal_period = IntParameter(low=5, high=15, default=9, space='sell', optimize=True)

    bbwindow = IntParameter(low=8, high=20, default=12, space='sell', optimize=True)
    bbdeviation = DecimalParameter(low=1, high=3, default=2, space='sell', optimize=True)

    sell_rsi = IntParameter(75, 95, default=85, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe, fastperiod=self.macd_fast_period.value, slowperiod=self.macd_slow_period.value, signalperiod=self.macd_signal_period.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=self.bbwindow.value, stds=self.bbdeviation.value)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe['macd'] > 0)  # over 0
                        & (dataframe['macd'] > dataframe['macdsignal'])  # over signal
                        & (dataframe['b'])
                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi.value)  # over sell_rsi
            ),
            'sell'] = 1
        return dataframe
