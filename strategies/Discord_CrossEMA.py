# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce
from technical.indicators import RMI, zema

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
# import talib.abstract as ta
#import ta

class CrossEMA(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 100 # inactive
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.029 # inactive

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 202 

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # # EMA - Exponential Moving Average
        df = dataframe.copy()
        dataframe['ema1']=ta.EMA(df, timeperiod=28)
        dataframe['ema2']=ta.EMA(df, timeperiod=48)
    
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        
        dataframe.loc[
            (
                (dataframe['ema1'] > dataframe['ema2']) &
                (dataframe['ema1'] < 0.019)                                 
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        
        dataframe.loc[
            (
                (dataframe['ema1'] < dataframe['ema2']) 
            ),
            'sell'] = 1
        return dataframe