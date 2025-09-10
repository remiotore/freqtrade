# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from datetime import datetime, timedelta, timezone

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.pivots_points import pivots_points


class FuturesStrategy1(IStrategy):
    custom_info = {}
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
    timeframe = '15m'

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.261,
        "455": 0.184,
        "1053": 0.088,
        "1757": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.15

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.052

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False



    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30


    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                "MACD": {
                    'fastd': {'color': 'blue'},
                    'fastk': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                },
                "Pivot": {
                    'pivot': {'color': 'black'},
                },
                'SMA': {
                    'sma15': {'color': 'white'},
                    'sma50': {'color': 'yellow'},
                },
            },
            'subplots': {

            }
        }

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str, **kwargs) -> float:


        return self.wallets.get_total_stake_amount() / 10



    def informative_pairs(self):

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Pivot point
        pp = pivots_points(dataframe)
        dataframe['pivot'] = pp["r1"]

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']


        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 10.0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        
        
        dataframe.loc[
            (
                # LONG
                (dataframe['close'] >= dataframe['open']) & # Check if candle is winning
                (dataframe['open'] >= dataframe['ema9']) &
                (dataframe['open'] >= dataframe['ema21']) &
                (dataframe['macdsignal'] >= 0) & # MACD positive
                (dataframe['rsi'] <= 70) & # RSI below overbought level
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # SHORT
                (dataframe['close'] <= dataframe['open']) & # Check if candle is winning
                (dataframe['open'] <= dataframe['ema9']) &
                (dataframe['open'] <= dataframe['ema21']) &
                (dataframe['macdsignal'] < 0) & # MACD positive
                (dataframe['rsi'] >= 70) & # RSI below oversell level
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
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
                #(dataframe["close"] > dataframe["open"]) & # Exit if price reverse
                (dataframe['volume'] == 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 0
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)

        dataframe.loc[
            (
                #(dataframe["close"] < dataframe["open"]) & # Exit if price reverse
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1

        return dataframe
