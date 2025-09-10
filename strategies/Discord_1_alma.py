from talipp.indicators import ALMA
from pandas import DataFrame
from functools import reduce
import numpy as np

import talib.abstract as ta

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

class alma(IStrategy):
    stoploss = -0.015
    timeframe = '5m'
    minimal_roi = {
          "0": 1000,

     }
    buy_aperiod=IntParameter(10,20,default=14,load=True,optimize=True)
    buy_aofset=DecimalParameter(0.33,0.65,default=0.65,decimals=2,load=True,optimize=True)
    buy_asigma=IntParameter(1,10, default=14,load=True,optimize=True)


    startup_candle_count = 60

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for val1 in self.buy_aperiod.range:
           
            for val2 in self.buy_aofset.range:
        
                for val3 in self.buy_asigma.range:       
                 x=ALMA(period=val1,offset=val2,sigma=val3,input_values=dataframe['close'])
                #  dataframe=dataframe.drop(range(y))
                #  dataframe.reset_index(inplace=True,drop=True)
                 x=np.append([np.nan]*(val1-1),x)
                 dataframe[f'alma_{val1}_{val2}_{val3}']=x

        
    
        
        return dataframe




    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(dataframe['close']>dataframe[f'alma_{self.buy_aperiod.value}_{self.buy_aofset.value}_{self.buy_asigma.value}'])


        if conditions:
             dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe['close']<dataframe[f'alma_{self.buy_aperiod.value}_{self.buy_aofset.value}_{self.buy_asigma.value}'])
                        
                        

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        return dataframe