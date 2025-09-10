from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd
import talib.abstract as ta
import talib
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from functools import reduce
from skopt.space import Categorical, Dimension, Integer, Real
from freqtrade.optimize.hyperopt_interface import IHyperOpt
import freqtrade.vendor.qtpylib.indicators as qtpylib



class asahyp3(IStrategy):
    hesma=IntParameter(1, 50, default=20,optimize=True,load=True ,space='buy' )
    
    # Buy enabler
    buy_hesma_enabled= CategoricalParameter([True, False], default=True, space="buy")
    buy_slowk_enabled=CategoricalParameter([True, False], default=True, space="buy")
    buy_slowd_enabled = CategoricalParameter([True, False], default=True, space="buy")
    # buy_price_enabled=DecimalParameter([True, False], default=True, space="buy")
    #Buy value
    buy_slowk=IntParameter(10, 50, default=40, space="buy")
    buy_slowd=IntParameter(10, 50, default=40, space="buy")
    # buy_price=DecimalParameter(0.1, 3.0, decimals=1, default=0.9, space="buy")
    # trigger
    buy_trigger = CategoricalParameter(["slowKd", "hasprv"], space="buy")

    # sell enabler
    sell_hesma_enabled = CategoricalParameter([True, False], default=True, space="sell")
    sell_slowk_enabled=CategoricalParameter([True, False], default=True, space="sell")
    sell_slowd_enabled=CategoricalParameter([True, False], default=True, space="sell")

    # sell value
    sell_slowk=IntParameter(50, 90, default=60, space="sell")
    sell_slowd=IntParameter(50, 90, default=60, space="sell")

    # trigger
    sell_trigger=CategoricalParameter("slowKd", space="sell")
  
  

    minimal_roi = {
          "0": 1000,

     }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.1

    # Optimal timeframe for the strategy
    timeframe = '1h'

    startup_candle_count = 60 

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        
        for val in self.hesma.range:
         dataframe[f'hacs_{val}']= ta.SMA(dataframe, timeperiod=val)
         dataframe['haos_{val}']=ta.SMA(dataframe['ha_open'], timeperiod=val)
        dataframe['prvs_close']=dataframe[f'hacs_{val}'].shift(1)
        # for i in self.hesma.range:
        
        dataframe['prvs_open']=dataframe[f'haos_'].shift(1)
        
        
        

        high=dataframe['high']
        low=dataframe['low']
        close=dataframe['close']
        open=dataframe['open']

        dataframe['slowk'], dataframe['slowd'] = talib.STOCH(high, low, close, fastk_period=14,
         slowk_period=3, slowk_matype=0,slowd_period=3, slowd_matype=0)
        # dataframe['slowk']=slowk
        # dataframe['slowd']=slowd
        return dataframe

   
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # gard
        if self.buy_hesma_enabled.value:
         conditions.append(dataframe[f'hacs_{self.hesma.value}']>dataframe[f'haos_'])      
        # if self.buy_price_enabled.value:
        #  conditions.append(dataframe['open']>self.buy_price.value)
        if self.buy_slowk_enabled.value:
          conditions.append(dataframe['slowk'] > self.buy_slowk.value)
        if self.buy_slowd_enabled.value:
          conditions.append(dataframe['slowd'] > self.buy_slowk.value)
        # trigger
        if self.buy_trigger.value == 'slowkd':
          conditions.append(qtpylib.crossed_above(
                dataframe['slowk'], dataframe['slowd']
            ))
        if self.buy_trigger.value == 'hasprv':
            conditions.append(dataframe['prvs_close'] > dataframe['prvs_open'])

        if conditions:
          dataframe.loc[
              reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
        return dataframe



    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
      conditions = []
      # gard
      if self.sell_slowk_enabled.value:
       conditions.append(dataframe['slowk'] > self.sell_slowk.value)
      if self.sell_slowd_enabled.value:
       conditions.append(dataframe['slowd'] > self.sell_slowd.value)
      if self.sell_hesma_enabled.value:
       conditions.append(dataframe[f'hacs_{self.hesma.value}']<dataframe[f'haos_'])
      # trigger
      if self.sell_trigger.value == 'slowkd':
          conditions.append(qtpylib.crossed_above(
                dataframe['slowd'], dataframe['slowk']
            ))

      conditions.append(dataframe['volume'] > 0)

      if conditions:
          dataframe.loc[
              reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
      return dataframe
