
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade

import logging
logger = logging.getLogger(__name__)

class RSI_MFI_2(IStrategy):
    INTERFACE_VERSION = 2



     
    buy_rsi = IntParameter(1, 25, default=18, space="buy") 
    buy_mfi = IntParameter(1, 15, default=2, space="buy") 
    sell_rsi = IntParameter(75, 99, default=91, space="sell") 
    sell_mfi = IntParameter(85, 99, default=91, space="sell") 

    minimal_roi = {




        "0": 10
    }

    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.011
    trailing_stop_positive_offset = 0.085
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    process_only_new_candles = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=8)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=4)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=8)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
                
        dataframe.loc[
            (
                (dataframe["rsi"] <= self.buy_rsi.value) &
                (dataframe["mfi"] <= self.buy_mfi.value) &
                (dataframe['roc'] <= -1) & #guard
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[
            (    
                (dataframe["rsi"] >= self.sell_rsi.value) &
                (dataframe["mfi"] >= self.sell_mfi.value) &
                (dataframe['roc'] >= 1) & #guard
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'sell'] = 1
        return dataframe