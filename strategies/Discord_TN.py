# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

import utils

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class TN(IStrategy):
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

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.005
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.005

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'natr': {'color': 'green'},
                'natr_5m': {'color': 'red'},
            },
            'subplots': {
            }
        }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '5m') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.dp:
            inf_tf = '5m'
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

            dataframe['natr'] = utils.natr(dataframe['high'], dataframe['low'], dataframe['close'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['natr'] > 0.05) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

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
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        return dataframe
    