from math import cos, exp, pi, sqrt

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, merge, DatetimeIndex

from freqtrade.strategy.interface import IStrategy



import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime

from freqtrade.persistence import Trade
from functools import reduce

from pandas.core.base import PandasObject

from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import timeframe_to_minutes



class vwap(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs

    """



    minimal_roi = {
        "0": 0.014,
        "15":0.018,
        "29":0.0295,
        "70":0.0145,
        "720":0
    }

    stoploss = -0.12

    timeframe = '1m'







    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        now = datetime.datetime.now()
        minutes=(now.hour * 60) + now.minute
        df_res = resample_to_interval(dataframe.resample(f'{minutes}min'))
        df_res['vwapday'] = Series.vwap(df_res)
        dataframe = resampled_merge(dataframe, df_res, fill_na=True)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=30)

        dataframe['vwap'] = Series.vwap(dataframe)
        devUD = [1.28, 2.01, 2.51, 3.09, 4.01]

        dataframe['Typical_Price'] = (dataframe['volume'] * (dataframe['high'] + dataframe['low']) / 2).cumsum()  

        dataframe['Typical_Volume'] = dataframe['volume'].cumsum()  
        dataframe['r_vwap'] = Series.rolling_vwap(dataframe,  window=200, min_periods=1)

        dataframe['vwap'] = dataframe['Typical_Price'] / dataframe['Typical_Volume']   

        dataframe['DEV'] = dataframe['vwap'].expanding().std()





        for dev in devUD:
            up = 'vwup{}'.format(dev)
            dow = 'vwdow{}'.format(dev)
            dataframe[up]= df_res['vwapday'] + dev * dataframe['DEV']
            dataframe[dow]= df_res['vwapday'] - dev * dataframe['DEV']










        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (   


























                (qtpylib.crossed_above(dataframe['rsi'], 30))

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (        


                (dataframe['rsi'] > 49)
            ),
            'sell'] = 1
        return dataframe
