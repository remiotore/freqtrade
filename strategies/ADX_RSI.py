
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy.hyper import DecimalParameter, IntParameter

import logging
logger = logging.getLogger(__name__)



class ADX_RSI(IStrategy):
    buy_rsi = IntParameter(10, 30, default=29, space="buy") 
    buy_adx = IntParameter(25, 40, default=39, space="buy")




    minimal_roi = {
        "0": 0.124,
        "18": 0.1,
        "60": 0.033,
        "90": 0
    }

    stoploss = -0.322

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
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    process_only_new_candles = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14) # overbought and oversold conditions
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14) #trend detection


    
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
              
        dataframe.loc[
            (
                ((dataframe["rsi"] <= self.buy_rsi.value)) & #oversold condition
                ((dataframe["adx"] >= self.buy_adx.value)) & #verify trend

                (dataframe['volume'] > 0) # volume above zero
            )
        ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (



                (dataframe['volume'] > 0) # volume above zero
            )
        ,'sell'] = 0
        return dataframe
