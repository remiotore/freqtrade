


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from pandas_ta.utils import data


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pd_ta 


class SuperBB(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0":  100
    }



    stoploss = -0.1

    trailing_stop = False




    timeframe = '4h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {

        'main_plot': {
            'ST_long':{'color': 'green'},
            'ST_short':{'color': 'red'},
            'bb_upperband':{'color': 'grey'},
            'bb_middleband':{'color': 'red'},
            'bb_lowerband':{'color': 'grey'}
           
        },
        'subplots': {

        
            "RSI": {
                'rsi': {'color': 'red'},
                'sobre_comprado': {'color':'blue'},
                'sobre_vendido': {'color':'green'}
            }
        }
    }
  
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


          dataframe['rsi'] = ta.RSI(dataframe)
          dataframe['sobre_comprado'] = 65
          dataframe['sobre_vendido'] = 30


          periodo = 7
          atr_multiplicador = 3.0
    
          dataframe['ST_long'] = pd_ta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=periodo,
                                                  multiplier=atr_multiplicador)[f'SUPERTl_{periodo}_{atr_multiplicador}']

          dataframe['ST_short'] = pd_ta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=periodo,
                                                  multiplier=atr_multiplicador)[f'SUPERTs_{periodo}_{atr_multiplicador}']


          bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
          dataframe['bb_lowerband'] = bollinger['lower']
          dataframe['bb_middleband'] = bollinger['mid']
          dataframe['bb_upperband'] = bollinger['upper']
          dataframe["bb_percent"] = (
          (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
           )
          dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
           )  

          return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        dataframe.loc[
            (    
               
                (dataframe['ST_long'] < dataframe['close']) & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
 
        dataframe.loc[
            (   (dataframe['close'] >=  dataframe['bb_middleband'] ) &
                (dataframe['ST_short'] > dataframe['close']) & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    