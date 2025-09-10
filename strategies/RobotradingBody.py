


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib






class RobotradingBody(IStrategy):
   
    INTERFACE_VERSION = 2

   
    minimal_roi = {
        "0": 0.9
    }

 
    stoploss = -0.99

    for_mult = IntParameter(1, 20, default=3, space='buy', optimize=True)
    for_sma_length = IntParameter(20, 200, default=100, space='buy', optimize=True)

    trailing_stop = False

    timeframe = '4h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 100

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
    

    def informative_pairs(self):
       
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['body'] = (dataframe['close'] - dataframe['open']).abs()
        dataframe['body_sma'] = (ta.SMA(dataframe['body'], timeperiod=int(self.for_sma_length.value))) * int(self.for_mult.value)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['open'] > dataframe['close'] ) &   
                (dataframe['body'] > dataframe['body_sma'] ) &   
                (dataframe['volume'] > 0)  
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['close'] > dataframe['open'] ) &   
                (dataframe['volume'] > 0) 
            ),
            'sell'] = 1
        return dataframe
    