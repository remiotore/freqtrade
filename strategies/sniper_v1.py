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


class sniper_v1(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {
        "b_rsi_val": 16,
        "buy_emalongb": 178,
        "buy_emamediumb": 66,
        "buy_emas1_cat": False,
        "buy_emas2_cat": True,
        "buy_emashortb": 49,
        "buy_stoch": 50,
        "buy_stoch_cat": False,
        "buy_stochv_cat": True,
        "buy_tke_cat": False,
        "buy_tke_val": 29,
        "buy_vfi_cat": True,
        "buy_vfi_cat2": False,
        "buy_vfi_cat3": False,
        "buy_vwmacd_cat": False,
    }

    sell_params = {
        "s_rsi_val": 13,
        "sell_ema1_cat": True,
        "sell_ema2_cat": True,
        "sell_emalongs": 155,
        "sell_emamediums": 48,
        "sell_emashorts": 28,
        "sell_stoch": 68,
        "sell_stoch_cat": False,
        "sell_stochv_cat": True,
        "sell_tke_cat": True,
        "sell_tke_val": 69,
        "sell_vfi_cat": True,
        "sell_vfi_cat2": True,
        "sell_vfi_cat3": False,
        "sell_vwmacd_cat": True,
    }

    minimal_roi = {
        "0": 0.393,
        "95": 0.098,
        "214": 0.054,
        "395": 0
    }

    stoploss = -0.288

    trailing_stop = True
    trailing_stop_positive = 0.051
    trailing_stop_positive_offset = 0.073
    trailing_only_offset_is_reached = False

    use_custom_stoploss = False

    timeframe = '15m'
    informative_timeframe = '1h'

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

    ''''def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe)
                             for pair in pairs]
        if self.dp:
            for pair in pairs:
                informative_pairs += [(pair, "1d")]

        return informative_pairs

    '''''



    buy_emalongb = IntParameter( 130, 200, default = 140, space = 'buy')
    buy_emamediumb = IntParameter(40, 100, default = 50, space = 'buy')
    buy_emashortb = IntParameter(15, 50, default = 20, space = 'buy')
    
    
    buy_stoch= IntParameter(5, 50, default=40, space = 'buy')

    buy_tke_val = IntParameter(1, 40, default = 37, space = 'buy')
    b_rsi_val = IntParameter(8, 25, default = 14, space = 'buy')

    sell_stoch= IntParameter(50, 100, default=80, space = 'sell')
    sell_tke_val = IntParameter(60, 90, default = 78, space = 'sell')
    s_rsi_val = IntParameter(8, 25, default = 14, space = 'sell')


    sell_emalongs = IntParameter( 130, 200, default = 140, space = 'sell')
    sell_emamediums = IntParameter(40, 100, default = 50, space = 'sell')
    sell_emashorts = IntParameter(15, 50, default = 20, space = 'sell')


    buy_emas1_cat  = CategoricalParameter ( [True, False], default = True, space = 'buy')
    buy_emas2_cat  = CategoricalParameter ( [True, False], default = True, space = 'buy')
    buy_stochv_cat = CategoricalParameter ( [True, False], default = True, space = 'buy')
    buy_vwmacd_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_vfi_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_vfi_cat2 = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_vfi_cat3 = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_stoch_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_stochv_cat = CategoricalParameter([True, False], default = True, space = 'buy' )

    buy_tke_cat = CategoricalParameter([True, False], default = True, space = 'buy' )

    sell_vfi_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_vfi_cat2 = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_vfi_cat3 = CategoricalParameter([True, False], default = True, space = 'sell' )
    buy_stochv_cat = CategoricalParameter ( [True, False], default = True, space = 'buy')
    sell_ema1_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_ema2_cat = CategoricalParameter([True, False], default = True, space = 'sell' )


    sell_vwmacd_cat = CategoricalParameter([True, False], default = True, space = 'sell' )

    sell_stoch_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_stochv_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    
    sell_tke_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        vwmacd = technicali.vwmacd(dataframe)
        fastkb, fastdb = ta.STOCHRSI(dataframe["close"], timeperiod=self.b_rsi_val.value, fastk_period=5, fastd_period=3, fastd_matype=0)
        fastks, fastds = ta.STOCHRSI(dataframe["close"], timeperiod=self.s_rsi_val.value, fastk_period=5, fastd_period=3, fastd_matype=0)
        vfi = technicali.vfi(dataframe)
        tke = technicali.TKE(dataframe)

        dataframe['emalongb'] = ta.EMA(dataframe, timeperiod = self.buy_emalongb.value)
        dataframe['emamedb'] = ta.EMA(dataframe, timeperiod = self.buy_emamediumb.value)
        dataframe['emashortb'] = ta.EMA(dataframe, timeperiod = self.buy_emashortb.value)

        dataframe['emalong'] = ta.EMA(dataframe, timeperiod = self.sell_emalongs.value)
        dataframe['emameds'] = ta.EMA(dataframe, timeperiod = self.sell_emamediums.value)
        dataframe['emashorts'] = ta.EMA(dataframe, timeperiod = self.sell_emashorts.value)

        dataframe['vwmacd'] = vwmacd['vwmacd']
        dataframe['vwmacds'] = vwmacd['signal']

        dataframe['stochfb'] = fastdb
        dataframe['stochkb'] = fastkb

        dataframe['stochfs'] = fastds
        dataframe['stochks'] = fastks


        dataframe['vfi'] = vfi[0]
        dataframe['vfima'] = vfi[1]

        dataframe['TKE'] = tke[0]
         
         
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.buy_emas1_cat == True:
            conditions.append(qtpylib.crossed_above(dataframe['emamedb'], dataframe['emalongb']))
        
        if self.buy_emas2_cat == True:
            conditions.append(qtpylib.crossed_above(dataframe['emashortb'], dataframe['emalongb']))
                              

        if self.buy_vwmacd_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe['vwmacd'], dataframe['vwmacds']))

        conditions.append(qtpylib.crossed_above(dataframe['stochfb'], dataframe['stochkb']))
                                     
        if self.buy_stochv_cat.value == True:
             conditions.append(dataframe['stochkb'] > self.buy_stoch.value)
        
        if self.buy_vfi_cat == True:
            conditions.append(qtpylib.crossed_above(dataframe['vfi'], dataframe['vfima']))

        conditions.append(qtpylib.crossed_above(dataframe['vfi'], 0))
        if self.buy_vfi_cat3 == True:
            conditions.append(qtpylib.crossed_above(dataframe['vfima'], 0))
        
        if self.buy_tke_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe['TKE'], self.buy_tke_val.value))
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
        return dataframe 
    


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        if self.sell_ema1_cat == True:
            conditions.append(qtpylib.crossed_below(dataframe['emameds'], dataframe['emalongs']))
        
        if self.sell_ema2_cat == True:
            conditions.append(qtpylib.crossed_below(dataframe['emashorts'], dataframe['emalongs']))
        



        if self.sell_vwmacd_cat.value == True:
            conditions.append(qtpylib.crossed_below(dataframe['vwmacd'], dataframe['vwmacds']))


        if self.sell_stoch_cat.value == True:
            conditions.append(qtpylib.crossed_below(dataframe['stochfs'], dataframe['stochks']))

        conditions.append(dataframe['stochks'] > self.sell_stoch.value)
        
                              
        
        if self.sell_vfi_cat.value == True:
            conditions.append(qtpylib.crossed_below(dataframe['vfi'], dataframe['vfima']))

        conditions.append(qtpylib.crossed_below(dataframe['vfi'], 0))
        if self.sell_vfi_cat3 == True:
            conditions.append(qtpylib.crossed_below(dataframe['vfima'], 0))
        
        
        if self.buy_tke_cat.value == True:
            conditions.append(qtpylib.crossed_above(dataframe['TKE'], self.sell_tke_val.value))







        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1


        return dataframe