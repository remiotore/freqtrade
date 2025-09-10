
# --- Do not remove these libs ---
from email.policy import default
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pdta
import talib.abstract as ta
# --------------------------------

class Kdj(IStrategy):

    # Optimal timeframe for the strategy
    timeframe = '5m'
    INTERFACE_VERSION: int = 3

    # run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    #####################################################

    # ROI table:
    minimal_roi = {
        "0": 0.318,
        "238": 0.233,
        "362": 0.09,
        "643": 0
    }

    # Stoploss:
    stoploss = -0.233

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.176
    trailing_stop_positive_offset = 0.213
    trailing_only_offset_is_reached = False
    #####################################################


    #####################################################
    #####################################################

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['t3'] = ta.T3(dataframe['close'], timeperiod=5, vfactor=0.9)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe['t3'] > dataframe['t3'].shift(1))

        # BUY
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe['t3'] < dataframe['t3'].shift(1))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        return dataframe
