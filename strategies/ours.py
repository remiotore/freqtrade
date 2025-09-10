






import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ours(IStrategy):


    INTERFACE_VERSION = 3

    can_short: bool = False


    minimal_roi = {
        "0": 1
        
    }


    stoploss = -0.348






    timeframe = '1h'

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    emalow = IntParameter(low=1, high=55, default=21, space='buy', optimize=True, load=True)
    emahigh = IntParameter(low=10, high=200, default=55, space='buy', optimize=True, load=True)
    emalong = IntParameter(low=55, high=361, default=200, space='buy', optimize=True, load=True)
    emaverylow = IntParameter(low=9, high=90, default=15, space='sell', optimize=True, load=True)

    startup_candle_count: int = 12

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
                'emalow': {'color': 'red'},
                'emahigh': {'color': 'green'},
                'emalong': {'color': 'blue'},
                'emaverylow': {'color' : 'orange'},
        },
        'subplots': {
        }
    }

    def informative_pairs(self):
       
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe['trix9'] = ta.TRIX(dataframe['close'], timeperiod=15)
        dataframe['trix15'] = ta.TRIX(dataframe['close'], timeperiod=21)

        dataframe['emaverylow'] = ta.EMA(dataframe['close'], timeperiod=9)
        dataframe['emalow'] = ta.EMA(dataframe['close'], timeperiod=21)
        dataframe['emahigh'] = ta.EMA(dataframe['close'], timeperiod=89)
        dataframe['emalong'] = ta.EMA(dataframe['close'], timeperiod=200)

        

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (   
                (dataframe['emalow'] > dataframe['emahigh']) &
                (dataframe['emahigh'] > dataframe['emalong']) &
                (dataframe['low'] > dataframe['emahigh'] )
            ),
            ['enter_long','ours']] = 1
        
        dataframe.loc[
            (
                (dataframe['emalow'] < dataframe['emahigh']) &
                (dataframe['emahigh'] < dataframe['emalong']) &
                (dataframe['low'] < dataframe['emahigh'] )
            ),
            ['enter_short','ours']] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (   
                (dataframe['emaverylow'] < dataframe['emalow']) 
            ),

            ['exit_long','ours']] = 1

        dataframe.loc[
            (
                (dataframe['emaverylow'] > dataframe['emalow']) 
            ),
            ['exit_short','ours']] = 1

        return dataframe
