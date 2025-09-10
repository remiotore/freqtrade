
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade

import logging
logger = logging.getLogger(__name__)

class TRSI(IStrategy):



    
    INTERFACE_VERSION = 2
    buy_rsi1 = IntParameter(5, 30, default=7, space="buy")
    buy_rsi2 = IntParameter(15, 35, default=34, space="buy")
    buy_cmf = DecimalParameter(-0.5, -0.1, default=-0.162, space="buy")
    sell_rsi1 = IntParameter(70, 95, default=82, space="sell")
    sell_rsi2 = IntParameter(65, 85, default=68, space="sell") 
    sell_cmf = DecimalParameter(0.1, 0.5, default=0.173, space="sell")

    minimal_roi = {




        "0": 10
    }

    stoploss = -0.235

    trailing_stop = True
    trailing_stop_positive = 0.011
    trailing_stop_positive_offset = 0.033
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'limit',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    process_only_new_candles = True
    startup_candle_count = 20 #time to wait before valid signals
    
    def chaikin_mf(self, dataframe, periods=20):
        close = dataframe['close']
        low = dataframe['low']
        high = dataframe['high']
        volume = dataframe['volume']

        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= volume
        cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()

        return Series(cmf, name='cmf')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi1'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi2'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['cmf'] = self.chaikin_mf(dataframe) #guard
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
                
        dataframe.loc[
            (
                (dataframe["rsi1"] <= self.buy_rsi1.value) & #Trigger
                (dataframe["rsi2"] <= self.buy_rsi2.value) & #Trigger
                (dataframe["cmf"] <= self.buy_cmf.value) & #Guard
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (    
                (dataframe["rsi1"] >= self.sell_rsi1.value) & #Trigger
                (dataframe["rsi2"] >= self.sell_rsi2.value) & #Trigger
                (dataframe["cmf"] >= self.sell_cmf.value) & #Guard
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'sell'] = 1
        return dataframe
