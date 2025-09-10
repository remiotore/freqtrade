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


class sniper_s(IStrategy):


    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.166,
        "109": 0.115,
        "180": 0.054,
        "538": 0
    }

    stoploss = -0.252


    buy_params = {
        "buy_aroondown": 11,
        "buy_aroonup": 69,
        "buy_arrondown_cat": False,
        "buy_arronup_cat": True,
        "buy_emas_cat": False,
        "buy_td": 5,
        "buy_td_cat": True,
        "buy_tke_cat": True,
        "buy_tke_val": 39,
        "buy_vfi_cat": True,
        "buy_vfi_cat2": False,
        "buy_vfi_cat3": True,
        "buy_vwmacd_cat": False,
        "buyema": 112,
    }

    sell_params = {
        "sell_aroondown": 96,
        "sell_aroonup": 16,
        "sell_arrondown_cat": False,
        "sell_arronup_cat": True,
        "sell_td_cat": False,
        "sell_tkcros_cat": True,
        "sell_tke_cat": True,
        "sell_tke_val": 64,
    }

    trailing_stop = True
    trailing_stop_positive = 0.164
    trailing_stop_positive_offset = 0.187
    trailing_only_offset_is_reached = True

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



    buy_aroonup = IntParameter(40, 100, default = 77, space = 'buy')
    buy_aroondown = IntParameter(0, 40, default = 24, space = 'buy' )
    buyema = IntParameter(45,120, default= 70, space ='buy')

    buy_tke_val = IntParameter(1, 40, default = 37, space = 'buy')


    buy_td = IntParameter(3, 9, default = 9, space = 'buy')



    sell_aroonup = IntParameter(0, 75, default = 24, space = 'sell' )
    sell_aroondown = IntParameter(40, 100, default = 68, space = 'sell')
    sell_tke_val = IntParameter(60, 90, default = 78, space = 'sell')


    buy_arrondown_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_arronup_cat = CategoricalParameter([True, False], default = True, space = 'buy' )

    
    buy_emas_cat = CategoricalParameter([True, False], default = True, space = 'buy' )

    buy_tke_cat = CategoricalParameter([True, False], default = True, space = 'buy' )

    buy_td_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_vwmacd_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_vfi_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_vfi_cat2 = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_vfi_cat3 = CategoricalParameter([True, False], default = True, space = 'buy' )



    sell_arronup_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_arrondown_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_tkcros_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_tke_cat = CategoricalParameter([True, False], default = True, space = 'sell' )

    sell_td_cat = CategoricalParameter([True, False], default = True, space = 'sell' )



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
      vfi = technicali.vfi(dataframe)
      
      aroon = ta.AROON(dataframe)


      tke = technicali.TKE(dataframe)

      td = technicali.td_sequential(dataframe)

      vwmacd = technicali.vwmacd(dataframe)


      dataframe['aroonup'] = aroon['aroonup']
      dataframe['aroondown'] = aroon['aroondown']





      dataframe['td'] = td['TD_count']










      dataframe['ema200'] = ta.EMA(dataframe, timeperiod = self.buyema.value)

      dataframe['vfi'] = vfi[0]
      dataframe['vfima'] = vfi[1]

      dataframe['vwmacd'] = vwmacd['vwmacd']
      dataframe['vwmacds'] = vwmacd['signal']





      dataframe['TKE'] = tke[0]

      return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        

        if self.buy_vfi_cat == True:
            conditions.append(qtpylib.crossed_above(dataframe['vfi'], dataframe['vfima']))

        if self.buy_vfi_cat2 == True:
            conditions.append(qtpylib.crossed_above(dataframe['vfi'], 0))
        if self.buy_vfi_cat3 == True:
            conditions.append(qtpylib.crossed_above(dataframe['vfima'], 0))

        if self.buy_vwmacd_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe['vwmacd'], dataframe['vwmacds']))



        if self.buy_tke_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe['TKE'], self.buy_tke_val.value))





        if self.buy_arronup_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe['aroonup'], self.buy_aroonup.value))

        if self.buy_arrondown_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe['aroondown'], self.buy_aroondown.value))


        if self.buy_emas_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe["close"], dataframe['ema200'])) 




        conditions.append(dataframe['volume'] > 0)
        if self.buy_td_cat == True:
            conditions.append(dataframe['td'] == self.buy_td.value)


        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
            return dataframe    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.sell_tke_cat.value == True:
            conditions.append(qtpylib.crossed_below(dataframe['TKE'], self.sell_tke_val.value))


        if self.sell_arrondown_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe['aroondown'], self.sell_aroondown.value))
        if self.sell_arronup_cat.value == True:
            conditions.append(qtpylib.crossed_below(dataframe['aroonup'], self.sell_aroonup.value))





        conditions.append(dataframe['volume'] > 0)
        
        
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1


        return dataframe