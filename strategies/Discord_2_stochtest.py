from pandas import DataFrame
from functools import reduce

import talib.abstract as ta

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

class stochtest(IStrategy):
    stoploss = -0.05
    timeframe = '1h'
    minimal_roi = {
          "0": 1000,

     }
    startup_candle_count = 60
    # slowk_p=IntParameter(1,30, default=14)
    # slowd_p=IntParameter(1,30, default=14)
    # fastk_p=IntParameter(1,30, default=14)

    buy_fastk=IntParameter(1,20,default=14)
    buy_slowk=IntParameter(1,20,default=14)
    buy_slowd=IntParameter(1,20, default=14)

    sell_slowk=IntParameter(50,100,default=50)
    sell_slowd=IntParameter(50,100,default=50)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
        for val1 in self.buy_fastk.range:
            for val2 in self.buy_slowk.range:
                for val3 in self.buy_slowd.range:
                  dataframe[f'slowk_{val2}_{val1}'], dataframe[f'slowd_{val3}_{val1}'] = ta.STOCH(dataframe['high'], dataframe['low'], dataframe['close'], fastk_period=val1,
                  slowk_period=val2,slowk_matype=0,slowd_period=val3,slowd_matype=0)
        
    
        
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
    

        conditions.append(qtpylib.crossed_above(
                dataframe[f'slowk_{self.buy_slowk.value}_{self.buy_fastk.value}'], dataframe[f'slowd_{self.buy_slowd.value}_{self.buy_fastk.value}']))
        
        # conditions.append(dataframe[f'slowk_{self.slowk_p.value}'] > self.buy_slowk.value)
        # conditions.append(dataframe[f'slowd_{self.slowd_p.value}'] > self.buy_slowd.value)
        
           
        if conditions:
             dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
        return dataframe

    

        
   
   
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # conditions.append(qtpylib.crossed_above(
        #         dataframe[f'slowd_{self.buy_slowd.value}_{self.buy_fastk.value}'],   dataframe[f'slowk_{self.buy_slowk.value}_{self.buy_fastk.value}']
        #     ))
        conditions.append(dataframe[f'slowk_{self.buy_slowk.value}_{self.buy_fastk.value}'] < self.sell_slowk.value)
        conditions.append(dataframe[f'slowd_{self.buy_slowd.value}_{self.buy_fastk.value}'] < self.sell_slowd.value)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        return dataframe
        