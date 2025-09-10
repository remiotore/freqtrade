
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Quickie_3(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:
        momentum based strategie. The main idea is that it closes trades very quickly, while avoiding excessive losses. Hence a rather moderate stop loss in this case
    """


    minimal_roi = {
        "60": 0.005,
        "10": 0.01,
    }


    stoploss = -0.25

    ticker_interval = 5

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (
                            (dataframe['adx'] > 30) &
                            (dataframe['tema'] < dataframe['bb_middleband']) &
                            (dataframe['tema'] > dataframe['tema'].shift(1)) &
                            (dataframe['sma_200'] > dataframe['close'])
                    )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['adx'] > 70) &
                    (dataframe['tema'] > dataframe['bb_middleband']) &
                    (dataframe['tema'] < dataframe['tema'].shift(1))
                )
            ),
            'sell'] = 1
        return dataframe
