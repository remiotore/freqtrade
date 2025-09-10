# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
from curses import meta
from itertools import pairwise
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame  # noqa
from datetime import datetime  # noqa
from typing import Optional, Union  # noqa

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, informative, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import sys
path_to_module = "/user_data/strategies/"
sys.path.append(path_to_module)
import logging
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade, Order
import copy
from freqtrade.strategy import stoploss_from_open
import drive_manager as dm
import os

logger = logging.getLogger(__name__)


class PriceAlertShort(IStrategy):

    """
    This is a strategy template to get you started.
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
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # ticker data activator
    process_only_new_candles = True

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # stoploss disabled
    #use_custom_stoploss = True
    #trailing_stop = False
    stoploss = -0.1

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # ROI disabled
    minimal_roi = {
        "0":  1.0
    }


    # Can this strategy go short?
    can_short: bool = True


    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # download file 
    operations, file_id = dm.download_file()

    # extract all the pair param
    pair_param = dm.extract_pair_param('short', operations)

    logger.info(f"pair param {pair_param}")

    informative_timeframe = '5m'

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        inf_tf = self.informative_timeframe
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe






    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']

        # even only for short operation enter_long has to be set
        dataframe['enter_long'] = None

        # logger.info(f"pair in self.pair_param {pair in self.pair_param}")
        # logger.info(f"self.pair_param[pair]['alert'] {self.pair_param[pair]['alert']}")
        # logger.info(f"qtpylib.crossed_below(dataframe['close_5m'], self.pair_param[pair]['entry_price']) {qtpylib.crossed_below(dataframe['close_5m'], self.pair_param[pair]['entry_price'])}")

        if pair in self.pair_param:
            if self.pair_param[pair]['alert']:
                dataframe.loc[
                    (
                        qtpylib.crossed_below(dataframe['close_5m'], self.pair_param[pair]['entry_price'])
                    ),
                    'enter_short'] = 1
            else:
                dataframe['enter_short'] = None

            last_candle = dataframe.iloc[-1]

            if last_candle['enter_short'] == 1:
                # self.pair_param[pair]['alert'] = False
                self.operations = dm.set_alert(pair, 'short', self.operations, self.file_id)
                self.pair_param = dm.extract_pair_param('short', self.operations)
                
        #dataframe.to_csv('user_data/output/strategy-df-{}.csv'.format(metadata['pair'].replace("/","-")))
        

        return dataframe





    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']
        if pair in self.pair_param:
            if not self.pair_param[pair]['alert']:
                dataframe.loc[
                    (
                        (qtpylib.crossed_above(dataframe['high'], self.pair_param[pair]['stop_loss'])) |
                        (qtpylib.crossed_below(dataframe['low'], self.pair_param[pair]['take_profit']))
                    ),
                    'exit_short'] = 1
            else:
                dataframe['exit_short'] = None

        dataframe.to_csv('user_data/output/strategy-df-{}-short.csv'.format(metadata['pair'].replace("/","-")))


        return dataframe
