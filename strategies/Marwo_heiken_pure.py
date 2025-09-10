
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy

class Marwo_heiken_pure(IStrategy):

    timeframe = '1h'

    minimal_roi = {
        "0": 0.04,
    }
    stoploss = -0.99
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:   
        dataframe['hclose']=(dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['hopen']= ((dataframe['open'].shift(2) + dataframe['close'].shift(2))/ 2) #it is not the same as real heikin ashi since I found that this is better.
        dataframe['hhigh']=dataframe[['open','close','high']].max(axis=1)
        dataframe['hlow']=dataframe[['open','close','low']].min(axis=1)

        dataframe['emac'] = ta.SMA(dataframe['hclose'], timeperiod=6) #to smooth out the data and thus less noise.
        dataframe['emao'] = ta.SMA(dataframe['hopen'], timeperiod=6)
        dataframe.loc[
            (
                (dataframe['emao'] > dataframe['emac'])
            ),
            'signal'] = 1
        dataframe['red_count'] = 0
        dataframe['green_count'] = 0
        dataframe['shall_enter'] = False
        dataframe['shall_exit'] = False

        for i in range(1, len(dataframe)):
            if ((dataframe.loc[i, 'hclose'] < dataframe.loc[i, 'hopen']) & (dataframe.loc[i - 1, 'signal'] == 1)):
                dataframe.loc[i, 'signal'] = 1
            if ((dataframe.loc[i, 'hopen'] < dataframe.loc[i, 'hclose']) & (dataframe.loc[i-1, 'signal'] == 1)):
                dataframe.loc[i, 'shall_enter'] = True
            elif ((dataframe.loc[i - 1, 'signal'] != 1) & (dataframe.loc[i, 'hclose'] < dataframe.loc[i, 'hopen'])):
                dataframe.loc[i, 'shall_exit'] = True







        return dataframe
        

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:







        dataframe.loc[(dataframe['shall_enter']), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:





        dataframe.loc[(dataframe["shall_exit"]), 'sell'] = 1
        return dataframe
