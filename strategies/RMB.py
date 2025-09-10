
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

class RMB(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {




        "0": 10
    }

    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.343
    trailing_stop_positive_offset = 0.433
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    process_only_new_candles = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=14, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
                
        dataframe.loc[
            (
                (dataframe["rsi"] <= 30) & 
                (dataframe["mfi"] <= 20) & 
                (dataframe["low"] < dataframe["bb_lowerband"]) & #trigger
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (    
                (dataframe["rsi"] >= 70) &
                (dataframe["mfi"] >= 80) &
                (dataframe["high"] > dataframe["bb_upperband"]) & #Trigger
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'sell'] = 1
        return dataframe