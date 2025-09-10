













from freqtrade.strategy.parameters import IntParameter, DecimalParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame



import pandas as pd
import ta
from ta.utils import dropna
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import numpy as np


class Heracles_4(IStrategy):

    minimal_roi = {
        "0": 0.598,
        "644": 0.166,
        "3269": 0.115,
        "7289": 0
    }

    stoploss = -0.256

    timeframe = '4h'


    buy_div_min = DecimalParameter(0, 1, default=0.16, decimals=2, space='buy')
    buy_div_max = DecimalParameter(0, 1, default=0.75, decimals=2, space='buy')
    buy_indicator_shift = IntParameter(0, 20, default=16, space='buy')
    buy_crossed_indicator_shift = IntParameter(0, 20, default=9, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = dropna(dataframe)

        dataframe['volatility_kcw'] = ta.volatility.keltner_channel_wband(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            window=20,
            window_atr=10,
            fillna=False,
            original_version=True
        )

        dataframe['volatility_dcp'] = ta.volatility.donchian_channel_pband(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            window=10,
            offset=0,
            fillna=False
        )

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy strategy Hyperopt will build and use.
        """
        conditions = []

        IND = 'volatility_dcp'
        CRS = 'volatility_kcw'
        DFIND = dataframe[IND]
        DFCRS = dataframe[CRS]

        d = DFIND.shift(self.buy_indicator_shift.value).div(
            DFCRS.shift(self.buy_crossed_indicator_shift.value))

        conditions.append(
            d.between(self.buy_div_min.value, self.buy_div_max.value))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy']=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sell strategy Hyperopt will build and use.
        """
        dataframe.loc[:, 'sell'] = 0
        return dataframe
