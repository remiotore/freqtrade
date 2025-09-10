# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


# This class is a sample. Feel free to customize it.
class REMORABB(IStrategy):
    """
    This is a custom strategy based on the given Tongda Xin code.
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

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = "1d"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "sma1": {},
            "sma2": {},
            "bbiboll": {},
            "upper": {},
            "lower": {},
            "long_stoploss": {"color": "red"},
            "short_stoploss": {"color": "blue"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
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
        """
        # Define moving averages
        MA1 = 5
        MA2 = 20
        dataframe["sma1"] = ta.SMA(dataframe, timeperiod=MA1)
        dataframe["sma2"] = ta.SMA(dataframe, timeperiod=MA2)

        # Calculate BBIBOLL
        dataframe["bbiboll"] = ta.SMA((dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3, timeperiod=5)
        dataframe["upper"] = dataframe["bbiboll"] + 2 * ta.STDDEV((dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3, timeperiod=5)
        dataframe["lower"] = dataframe["bbiboll"] - 2 * ta.STDDEV((dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3, timeperiod=5)

        # Calculate ATR
        # 使用 numpy 的 abs 函数来计算绝对值
        dataframe["tr1"] = dataframe["high"] - dataframe["low"]
        dataframe["tr2"] = (dataframe["high"] - dataframe["close"].shift(1)).abs()
        dataframe["tr3"] = (dataframe["low"] - dataframe["close"].shift(1)).abs()
        dataframe["tr"] = dataframe[["tr1", "tr2", "tr3"]].max(axis=1)
        dataframe["atr14"] = ta.SMA(dataframe["tr"], timeperiod=14)

        # Calculate stoploss lines
        dataframe["long_stoploss"] = dataframe["close"] - 2 * dataframe["atr14"]
        dataframe["short_stoploss"] = dataframe["close"] + 2 * dataframe["atr14"]

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
                (qtpylib.crossed_above(dataframe["sma1"], dataframe["sma2"]))  # Signal: SMA1 crosses above SMA2
                & (dataframe["close"] > dataframe["bbiboll"])  # Guard: Close above BBIBOLL
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["sma2"], dataframe["sma1"]))  # Signal: SMA2 crosses above SMA1
                & (dataframe["close"] < dataframe["bbiboll"])  # Guard: Close below BBIBOLL
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "enter_short",
        ] = 1

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
                (qtpylib.crossed_above(dataframe["sma2"], dataframe["sma1"]))  # Signal: SMA2 crosses above SMA1
                & (dataframe["close"] < dataframe["bbiboll"])  # Guard: Close below BBIBOLL
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["sma1"], dataframe["sma2"]))  # Signal: SMA1 crosses above SMA2
                & (dataframe["close"] > dataframe["bbiboll"])  # Guard: Close above BBIBOLL
                & (dataframe["volume"] > 0)  # Make sure Volume
                ),
            "exit_short",
        ] = 1

        return dataframe