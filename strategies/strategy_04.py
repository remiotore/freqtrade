import talib.abstract as ta
import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, RealParameter, informative, merge_informative_pair
import pandas_ta as pta


class strategy_04(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '30m'

    minimal_roi = {
        "0": 0.162,
        "69": 0.097,
        "229": 0.061,
        "566": 0
    }

    stoploss = -0.345

    trailing_stop = True

    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.058
    trailing_only_offset_is_reached = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema20'] = ta.EMA(dataframe, 20)
        dataframe['ema25'] = ta.EMA(dataframe, 25)
        dataframe['ema30'] = ta.EMA(dataframe, 30)
        dataframe['ema35'] = ta.EMA(dataframe, 35)
        dataframe['ema40'] = ta.EMA(dataframe, 40)
        dataframe['ema45'] = ta.EMA(dataframe, 45)
        dataframe['ema50'] = ta.EMA(dataframe, 50)
        dataframe['ema55'] = ta.EMA(dataframe, 55)

        dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []


        conditions.append(dataframe['volume'] > 0)

        conditions.append(dataframe['close'] < dataframe['ema20'])
        conditions.append(dataframe['close'] < dataframe['ema25'])
        conditions.append(dataframe['close'] < dataframe['ema30'])
        conditions.append(dataframe['close'] < dataframe['ema35'])
        conditions.append(dataframe['close'] < dataframe['ema40'])
        conditions.append(dataframe['close'] < dataframe['ema45'])
        conditions.append(dataframe['close'] < dataframe['ema50'])
        conditions.append(dataframe['close'] < dataframe['ema55'])

        conditions.append(qtpylib.crossed_above(dataframe['rsi'], 30))
        conditions.append(dataframe['rsi'] < 50)

        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []


        conditions.append(dataframe['volume'] > 0)

        conditions.append(dataframe['close'] > dataframe['ema20'])
        conditions.append(dataframe['close'] > dataframe['ema25'])
        conditions.append(dataframe['close'] > dataframe['ema30'])
        conditions.append(dataframe['close'] > dataframe['ema35'])
        conditions.append(dataframe['close'] > dataframe['ema40'])
        conditions.append(dataframe['close'] > dataframe['ema45'])
        conditions.append(dataframe['close'] > dataframe['ema50'])
        conditions.append(dataframe['close'] > dataframe['ema55'])

        conditions.append(qtpylib.crossed_below(dataframe['rsi'], 70))

        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
