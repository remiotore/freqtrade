import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
                                DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter, stoploss_from_open)
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Tuple
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade_strategies.custom_indicators import zema, tv_hma, pmax
import talib.abstract as ta
import math
import pandas_ta as pta

import logging
from logging import FATAL
import time

logger = logging.getLogger(__name__)


class multi_tf(IStrategy):

    def version(self) -> str:
        return "v1"

    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.2
    }

    stoploss = -0.1

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 100




    @informative('30m')
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe




    @informative('1h', 'BTC/{stake}')
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe


    @informative('1h', 'ETH/BTC')
    def populate_indicators_eth_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe



    @informative('1h', 'BTC/{stake}', '{column}')
    def populate_indicators_btc_1h_2(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi_fast_upper'] = ta.RSI(dataframe, timeperiod=4)
        return dataframe



    @informative('1h', 'BTC/{stake}', '{base}_{column}_{timeframe}')
    def populate_indicators_btc_1h_3(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi_super_fast'] = ta.RSI(dataframe, timeperiod=2)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['rsi_less'] = dataframe['rsi'] < dataframe['rsi_1h']
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stake = self.config['stake_currency']
        dataframe.loc[
            (
                (dataframe[f'btc_{stake}_rsi_1h'] < 35)
                &
                (dataframe['eth_btc_rsi_1h'] < 50)
                &
                (dataframe['rsi_fast_upper'] < 40)
                &
                (dataframe['btc_rsi_super_fast_1h'] < 30)
                &
                (dataframe['rsi_30m'] < 40)
                &
                (dataframe['rsi_1h'] < 40)
                &
                (dataframe['rsi'] < 30)
                &
                (dataframe['rsi_less'] == True)
                &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_signal_rsi')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['rsi'] > 70)
                &
                (dataframe['rsi_less'] == False)
                &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'exit_signal_rsi')

        return dataframe
