



import logging

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)


import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib

logger = logging.getLogger(__name__)






class AwesomeStrategy20240418(IStrategy):
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


    INTERFACE_VERSION = 3
    BUY_ROC = 2.444
    SELL_ROC = 3.683
    LEVERAGE = 4.242

    timeframe = '1m'

    can_short: bool = True

    stoploss = -0.294

    minimal_roi: {
        "0": 0.009,
        "4": 0.006,
        "16": 0.003,
        "34": 0
    }

    buy_leverage = DecimalParameter(1.0, 5.0, default=1.0, space='buy', optimize=True, load=True)
    sell_leverage = DecimalParameter(1.0, 5.0, default=1.0, space='sell', optimize=True, load=True)

    buy_roc = DecimalParameter(0.1, 5.0, default=1.0, space='buy', optimize=True, load=True)
    sell_roc = DecimalParameter(0.1, 5.0, default=1.0, space='sell', optimize=True, load=True)

    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.0  # Disabled / not configured

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 2

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
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

        dataframe['roc'] = ta.ROC(dataframe, timeperiod=1)

        """

        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """


        dataframe.loc[
            (
                (dataframe['roc'] >= self.SELL_ROC)
            ),
            'enter_short'] = 1

        dataframe.loc[
            (
                (dataframe['roc'] <= -self.BUY_ROC)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

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
        return self.LEVERAGE


