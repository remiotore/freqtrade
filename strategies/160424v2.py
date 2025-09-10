



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union
import copy
import logging
import pathlib
import rapidjson
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import DecimalParameter, CategoricalParameter
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
import time
from typing import Optional
import warnings
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta

class 160424v2(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """


    INTERFACE_VERSION = 3

    can_short: bool = True


    minimal_roi = {
        "30": 0.50,
        "0": 0.40
    }


    stoploss = -0.99

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.25  # Disabled / not configured

    timeframe = '5m'

    process_only_new_candles = True

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)

    startup_candle_count: int = 300

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['adx'] = ta.ADX(dataframe)






























        dataframe['rsi'] = ta.RSI(dataframe)










        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']







        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd'] # kısa 12 periyot ema
        dataframe['macdsignal'] = macd['macdsignal'] # uzun 26 periyot ema
        dataframe['macdhist'] = macd['macdhist']

        dataframe['mfi'] = ta.MFI(dataframe)





        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

















        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)





































































        """

        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 20.0
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                
                (dataframe['macdhist'] > 0) &
                (dataframe['ema5'] > dataframe['ema10']) & 
                (dataframe['ema10'] > dataframe['ema20']) &  

                 
                (dataframe['close'] > dataframe['ema20']) &
                (dataframe['volume'] > 0)  
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['macdhist'] < 0) &
                (dataframe['ema9'] < dataframe['ema21']) &  
                (dataframe['ema21'] > dataframe['ema50']) & 
                (dataframe['ema50'] > dataframe['ema100']) & 
                 
		(dataframe['close'] < dataframe['ema9']) & 
		(dataframe['close'] > dataframe['ema21']) &
                (dataframe['close'] > dataframe['ema50']) &
		(dataframe['close'] > dataframe['ema100']) &
                (dataframe['volume'] > 0)  
            ),
            'enter_short'] = 1   

        dataframe.loc[
            (
                (dataframe['ema9'] < dataframe['ema21']) &  
                (dataframe['ema21'] < dataframe['ema50']) &
                (dataframe['ema50'] < dataframe['ema100']) & 
                                 
		(dataframe['close'] < dataframe['ema9']) & 
		(dataframe['high'] >= dataframe['ema9']) & 
		(dataframe['close'] < dataframe['ema21']) &
                (dataframe['close'] < dataframe['ema50']) &
		(dataframe['close'] < dataframe['ema100']) &
                (dataframe['volume'] > 0)  
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
               (dataframe['close'] < dataframe['ema21']) & 
               (dataframe['macdhist'] < 0) &
               (dataframe['volume'] > 0)  
                 ),

            'exit_long'] = 1

        dataframe.loc[
            (
               (dataframe['close'] > dataframe['ema21'])   &
               (dataframe['macdhist'] > 0) & 
               (dataframe['volume'] > 0)  
            ),
            'exit_short'] = 1

        return dataframe
      
 
    