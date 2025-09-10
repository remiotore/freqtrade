# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pdta
from functools import reduce


class WenRarri(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:

    # Stoploss:
    stoploss = -0.1

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.308
    trailing_stop_positive_offset = 0.314
    trailing_only_offset_is_reached = True

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 100

    # Optional order type mapping.
    order_types = {
        "buy": "limit",
        "sell": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        bbands = pdta.bbands(dataframe["close"], length=20, std=2)
        stoch = pdta.stoch(dataframe["high"], dataframe["low"], dataframe["close"])

        dataframe["BB_LOWER_20"] = bbands["BBL_20_2.0"]
        dataframe["BB_MIDDLE_20"] = bbands["BBM_20_2.0"]
        dataframe["BB_UPPER_20"] = bbands["BBU_20_2.0"]

        dataframe["STOCH_k_14_3_3"] = stoch["STOCHk_14_3_3"]
        dataframe["STOCH_d_14_3_3"] = stoch["STOCHd_14_3_3"]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                    (dataframe["close"].shift(1) > dataframe["BB_LOWER_20"]) &
                    (dataframe["STOCH_k_14_3_3"] < 20) &
                    (dataframe["STOCH_d_14_3_3"] < 20) &
                    (dataframe["volume"] > 0)
            ),
            "buy"] = 1
        
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                    (dataframe["close"].shift(1) > dataframe["BB_UPPER_20"]) &
                    (dataframe["STOCH_k_14_3_3"] > 80) &
                    (dataframe["STOCH_d_14_3_3"] > 80) &
                    (dataframe["volume"] > 0)
            ),
            "sell"] = 1

        return dataframe