# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame  # noqa
from datetime import datetime  # noqa
from typing import Optional, Union  # noqa

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class DcaBitcoin(IStrategy):
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
    timeframe = '1w'

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 1000000000000
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -1.0

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

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # Example specific variables
    max_entry_position_adjustment = 365

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'bb_lowerband': {},
                'bb_middleband': {},
                'bb_upperband': {}
            },
            'subplots': {
                'BB Percent': {
                    'bb_percent': {}
                },
                'BB Width': {
                    'bb_width': {}
                }
            }
        }

    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: 'Optional[str]', side: str, **kwargs) -> float:
        """
        Customize stake size for each new trade.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_stake: A stake amount proposed by the bot.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A stake size, which is between min_stake and max_stake.
        """
        return proposed_stake

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

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=30, stds=2)
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
                (dataframe['volume'] > 0)
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
                (dataframe['volume'] < 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        if exit_reason == 'force_exit' and (trade.calc_profit_ratio(rate) < 0 or trade.calc_profit_ratio(rate) > 0):
            # Reject force-sells with negative profit
            # This is just a sample, please adjust to your needs
            # (this does not necessarily make sense, assuming you know when you're force-selling)
            return False
        return True

    def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: Optional[float],
                              max_stake: float, **kwargs):
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """
        return max_stake
