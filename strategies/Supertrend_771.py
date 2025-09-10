
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import numpy as np


def supertrend(dataframe, multiplier=3, period=10):
    """
    Supertrend Indicator
    adapted for freqtrade
    from: https://github.com/freqtrade/freqtrade-strategies/issues/30
    """
    df = dataframe.copy()

    df['TR'] = ta.TRANGE(df)
    df['ATR'] = df['TR'].ewm(alpha=1 / period).mean()

    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)

    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < df['final_lb'].iat[i] else 0.00

    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)

    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return DataFrame(index=df.index, data={
        'ST': df[st],
        'STX': df[stx]
    })


class Supertrend_771(IStrategy):


    minimal_roi = {
        "0": 0.1,
        "2880": 0.01
    }


    stoploss = -0.25

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["supertrend_3_12"] = supertrend(dataframe, 3, 12)["STX"]
        dataframe["supertrend_1_10"] = supertrend(dataframe, 1, 10)["STX"]
        dataframe["supertrend_2_11"] = supertrend(dataframe, 2, 11)["STX"]

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe["supertrend_3_12"] == "up") &
                        (dataframe["supertrend_1_10"] == "up") &
                        (dataframe["supertrend_2_11"] == "up")
                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe["supertrend_3_12"] == "down") &
                        (dataframe["supertrend_1_10"] == "down") &
                        (dataframe["supertrend_2_11"] == "down")
                 )
            ),
            'sell'] = 1
        return dataframe
