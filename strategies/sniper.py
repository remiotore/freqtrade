import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from functools import reduce
import technical.indicators as technicali
import technical.pivots_points as technicalp
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class sniper(IStrategy):


    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.172,
        "51": 0.146,
        "132": 0.043,
        "410": 0
    }

    stoploss = -0.5

    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    use_custom_stoploss = False

    timeframe = '15m'


    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False

    }
    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
    
   



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
      
      
      aroon = ta.AROON(dataframe)
      ichi = technicali.ichimoku(dataframe)
      pivot = technicalp.pivots_points(dataframe)
      vwmacd = technicali.vwmacd(dataframe)
      VIDYA = technicali.VIDYA(dataframe, length = 11)
      td = technicali.td_sequential(dataframe)

      dataframe['aroonup'] = aroon['aroonup']
      dataframe['aroondown'] = aroon['aroondown']
 
      dataframe['vwmacd'] = vwmacd['vwmacd']
      dataframe['vwmacds'] = vwmacd['signal']
      dataframe['td'] = td['TD_count']
      
      dataframe['p'] = pivot['pivot']

      dataframe['tenkan'] = ichi['tenkan_sen']
      dataframe['kijun'] = ichi['kijun_sen']

      dataframe['VIDYA'] = VIDYA


      


      return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []
        
        conditions.append(qtpylib.crossed_above(dataframe["close"], dataframe['VIDYA']))

        conditions.append(dataframe['td'] == 9)
        conditions.append(qtpylib.crossed_above(dataframe['aroonup'], 77))


        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
            return dataframe    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []



       
        conditions.append(qtpylib.crossed_below(dataframe['aroondown'], 68))
     
        conditions.append(qtpylib.crossed_below(dataframe['aroonup'], 24))
     
        conditions.append(qtpylib.crossed_below(dataframe['tenkan'], dataframe['kijun']))

        conditions.append(dataframe['volume'] > 0)
        
        
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1


        return dataframe