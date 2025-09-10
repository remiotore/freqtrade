













import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade
from technical.util import resample_to_interval, resampled_merge

import logging
logger = logging.getLogger(__name__)

class SlowPotato_358(IStrategy):
    """
    This strategy uses the averages for the last 2 days high/low and sets up buy and sell orders acordingly
    Currently developing and testing this strategy
    """

    minimal_roi = {
        "0": 0.203,
        "36": 0.069,
        "78": 0.035,
        "156": 0
    }

    stoploss = -0.171

    timeframe = '5m'

    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.01

    use_sell_signal = False
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
        }

    process_only_new_candles = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        If close candle breaks lower or equal to average low for last 1 days buy it
        """
        
        dataframe.loc[
            (
                (dataframe['low'].rolling(288).mean() < dataframe['high'].rolling(288).mean()) & ## average is currently below high
                (dataframe['low'] <= dataframe['low'].rolling(288).mean()) & ## current dataframe is below average low
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        If open candle breaks higher or equal to average high for last 1 days sell it
        """

        dataframe.loc[
            (
                (dataframe['high'].rolling(288).mean() > dataframe['low'].rolling(288).mean()) & ## average is currently above high
                (dataframe['high'] >= dataframe['high'].rolling(288).mean()) & ## current dataframe is above average high
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'sell'] = 0
        return dataframe