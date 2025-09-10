# safe_max_strategy.py

# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np

class SafeMaxStrategy(IStrategy):
    """
    SafeMaxStrategy
    Goal: Provide a balanced approach with trailing stop, partial ROI, and
    multiple indicators to catch good entries while minimizing risk.
    """

    INTERFACE_VERSION = 3

    can_short = False  # Set True if you want to enable shorting (and your exchange supports it)

    # --- Strategy Parameters ---
    # 1) ROI
    minimal_roi = {
        "120": 0.02,  # Accept 2% profit if trade open >= 120 min
        "60": 0.04,   # Accept 4% profit if trade open >= 60 min
        "0": 0.10     # Aim for 10% profit otherwise
    }

    # 2) Stoploss
    stoploss = -0.10  # Hard stop if the trade goes -10%

    # 3) Timeframe
    timeframe = "5m"

    # 4) Trailing Stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = False

    # Only process new candles to reduce CPU usage
    process_only_new_candles = True

    # We’ll use exit signal for confirmation
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # We'll define some hyperopt parameters for RSI, etc. if you want to optimize
    # from freqtrade.strategy import IntParameter, CategoricalParameter, etc.

    # Strategy order types
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    startup_candle_count = 200  # Enough candles for indicators

    def informative_pairs(self):
        # Return an empty list if no informative pairs are needed
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add technical indicators.
        """
        # RSI (standard)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # (Optional) Additional indicators like Parabolic SAR, Volume filters, etc. 
        # dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry logic (long).
        """
        # Overbought/oversold + Bollinger + MACD cross
        dataframe.loc[
            (
                (dataframe['rsi'] < 35)  # RSI oversold threshold
                & (dataframe['close'] < dataframe['bb_lowerband'])  # Price below lower Boll band
                & (dataframe['macd'] > dataframe['macdsignal'])  # MACD bullish cross
                & (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic (long).
        """
        # If RSI recovers above 65 or price recovers to/beyond Bollinger mid, exit
        dataframe.loc[
            (
                (dataframe['rsi'] > 65)
                | (dataframe['close'] >= dataframe['bb_middleband'])
            ),
            'exit_long'
        ] = 1

        return dataframe