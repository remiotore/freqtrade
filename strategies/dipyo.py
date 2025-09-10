# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
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


class chatgpt3(IStrategy):
    """
    Futures Profit Strategy for leveraging technical analysis indicators to
    trade futures contracts profitably by using key strategies like RSI, MACD, and Bollinger Bands.

    """
    INTERFACE_VERSION = 3
    can_short: bool = True  # Futures allows short selling.
    minimal_roi = {
        "60": 0.01,  # ROI after 60 minutes at 1% profit
        "30": 0.03,  # ROI after 30 minutes at 3% profit
        "0": 0.05,  # ROI after 0 minutes at 5% profit (final exit)
    }

    stoploss = -0.10  # Stoploss at 10% to limit losses
    trailing_stop = True  # Enable trailing stop
    trailing_stop_positive = 0.02  # Trailing stop at 2% profit
    trailing_stop_positive_offset = 0.03  # Offset trailing stop by 3%

    timeframe = "15m"  # Optimal timeframe for futures trading
    process_only_new_candles = True  # Only process new candle data for efficiency

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    startup_candle_count: int = 200

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add technical indicators to the DataFrame
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["macd"], dataframe["macdsignal"], dataframe["macdhist"] = ta.MACD(dataframe)
        dataframe["sar"] = ta.SAR(dataframe)
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
                dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )

        # Stochastic Fast Indicators
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Entry logic based on RSI and Bollinger Bands
        dataframe.loc[
            (qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value))
            & (dataframe["tema"] <= dataframe["bb_middleband"])
            & (dataframe["tema"] > dataframe["tema"].shift(1))
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1

        dataframe.loc[
            (qtpylib.crossed_above(dataframe["rsi"], self.short_rsi.value))
            & (dataframe["tema"] > dataframe["bb_middleband"])
            & (dataframe["tema"] < dataframe["tema"].shift(1))
            & (dataframe["volume"] > 0),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit logic based on RSI and trend reversal
        dataframe.loc[
            (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value))
            & (dataframe["tema"] > dataframe["bb_middleband"])
            & (dataframe["tema"] < dataframe["tema"].shift(1))
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1

        dataframe.loc[
            (qtpylib.crossed_above(dataframe["rsi"], self.exit_short_rsi.value))
            & (dataframe["tema"] <= dataframe["bb_middleband"])
            & (dataframe["tema"] > dataframe["tema"].shift(1))
            & (dataframe["volume"] > 0),
            "exit_short",
        ] = 1

        return dataframe
