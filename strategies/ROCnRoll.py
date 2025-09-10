
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy.hyper import IntParameter, DecimalParameter

import logging
logger = logging.getLogger(__name__)


class ROCnRoll(IStrategy):



    minimal_roi = {
        "0": 0.132,
        "32": 0.057,
        "80": 0.011,
        "195": 0
    }

    stoploss = -0.309


    buy_roc = DecimalParameter(-6.00, 6.00, default=-0.675, space="buy") 
    sell_mfi = IntParameter(75, 99, default=82, space="sell") 

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
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=20) #guard
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=20) #trigger
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=20) #trigger
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
              
        dataframe.loc[
            (
                ((dataframe["roc"] <= self.buy_roc.value)) &
                ((dataframe["roc"]) <= (dataframe["roc"].shift(1))) &
                ((dataframe["roc"]) <= (dataframe["roc"].shift(2))) &
                ((dataframe["adx"] >= 25)) & #guard
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (

                ((dataframe["mfi"] >= self.sell_mfi.value)) & 
                ((dataframe["adx"] >= 25)) &
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'sell'] = 1
        return dataframe

