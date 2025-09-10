import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (
    merge_informative_pair,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
    CategoricalParameter,
    stoploss_from_open,
)
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Tuple
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date
from technical.indicators import zema, ichimoku
import talib.abstract as ta
import math
import pandas_ta as pta
from finta import TA as fta
import logging
from logging import FATAL
import time

logger = logging.getLogger(__name__)











def tv_wma(df, length=9) -> DataFrame:
    """
    Source: Tradingview "Moving Average Weighted"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : WMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_wma'
    """

    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + df.shift(i) * weight

    tv_wma = (sum / norm) if norm > 0 else 0
    return tv_wma


def tv_hma(dataframe, length=9) -> DataFrame:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : HMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    h = 2 * tv_wma(dataframe["close"], math.floor(length / 2)) - tv_wma(dataframe["close"], length)

    tv_hma = tv_wma(h, math.floor(math.sqrt(length)))


    return tv_hma


class Cenderawasih_1b(IStrategy):
    def version(self) -> str:
        return "v1b"

    INTERFACE_VERSION = 2

    minimal_roi = {"0": 100.0}

    buy_params = {
        "base_nb_candles_buy_hma": 93,
        "low_offset_hma": 0.939,
        "base_nb_candles_buy_ema": 38,
        "low_offset_ema": 1.08,
        "buy_length_volatility": 10,
        "buy_max_volatility": 1.43,
        "base_nb_candles_buy_ema2": 36,
        "low_offset_ema2": 0.999,
        "buy_rsi_1": 65,
        "buy_rsi_fast_1": 39,
        "rsi_buy_ema": 56,
    }

    sell_params = {
        "base_nb_candles_sell_hma": 87,
        "high_offset_hma": 0.933,
        "base_nb_candles_sell_ema": 16,
        "high_offset_ema": 0.97,
        "base_nb_candles_sell_ema2": 70,
        "high_offset_ema2": 0.989,
        "base_nb_candles_sell_ema3": 93,
        "high_offset_ema3": 1.004,
    }

    protection_params = {
        "cooldown_lookback": 2,  # value loaded from strategy
    }

    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=False)

    @property
    def protections(self):
        prot = []

        prot.append(
            {"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}
        )
        return prot

    rsi_buy_ema = IntParameter(20, 70, default=61, space="buy", optimize=False)
    buy_rsi_1 = IntParameter(0, 70, default=50, space="buy", optimize=False)
    buy_rsi_fast_1 = IntParameter(0, 70, default=50, space="buy", optimize=False)

    optimize_buy_hma = False
    base_nb_candles_buy_hma = IntParameter(
        5, 100, default=6, space="buy", optimize=optimize_buy_hma
    )
    low_offset_hma = DecimalParameter(
        0.9, 0.99, default=0.95, space="buy", optimize=optimize_buy_hma
    )

    optimize_buy_ema = False
    base_nb_candles_buy_ema = IntParameter(
        5, 100, default=6, space="buy", optimize=optimize_buy_ema
    )
    low_offset_ema = DecimalParameter(0.9, 1.1, default=1, space="buy", optimize=optimize_buy_ema)

    optimize_buy_ema2 = False
    base_nb_candles_buy_ema2 = IntParameter(
        5, 80, default=6, space="buy", optimize=optimize_buy_ema2
    )
    low_offset_ema2 = DecimalParameter(0.9, 1.1, default=1, space="buy", optimize=optimize_buy_ema2)

    optimize_buy_volatility = False
    buy_length_volatility = IntParameter(
        10, 200, default=72, space="buy", optimize=optimize_buy_volatility
    )
    buy_min_volatility = DecimalParameter(
        0, 0.5, default=0, decimals=2, space="buy", optimize=False
    )
    buy_max_volatility = DecimalParameter(
        0.5, 2, default=1, decimals=2, space="buy", optimize=optimize_buy_volatility
    )

    optimize_sell_hma = False
    base_nb_candles_sell_hma = IntParameter(
        5, 100, default=6, space="sell", optimize=optimize_sell_hma
    )
    high_offset_hma = DecimalParameter(
        0.9, 1.1, default=0.95, space="sell", optimize=optimize_sell_hma
    )

    optimize_sell_ema = False
    base_nb_candles_sell_ema = IntParameter(
        5, 100, default=6, space="sell", optimize=optimize_sell_ema
    )
    high_offset_ema = DecimalParameter(
        0.9, 1.1, default=0.95, space="sell", optimize=optimize_sell_ema
    )

    optimize_sell_ema2 = False
    base_nb_candles_sell_ema2 = IntParameter(
        5, 100, default=6, space="sell", optimize=optimize_sell_ema2
    )
    high_offset_ema2 = DecimalParameter(
        0.9, 1.1, default=0.95, space="sell", optimize=optimize_sell_ema2
    )

    optimize_sell_ema3 = False
    base_nb_candles_sell_ema3 = IntParameter(
        5, 100, default=6, space="sell", optimize=optimize_sell_ema3
    )
    high_offset_ema3 = DecimalParameter(
        0.9, 1.1, default=0.95, space="sell", optimize=optimize_sell_ema3
    )

    stoploss = -0.098

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    timeframe = "5m"

    process_only_new_candles = True
    startup_candle_count = 200
    age_filter = 30

    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["age_filter_ok"] = (
            dataframe["volume"].rolling(window=self.age_filter, min_periods=self.age_filter).min()
            > 0
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)

        dataframe["sqzmi"] = fta.SQZMI(dataframe)

        dataframe["live_data_ok"] = dataframe["volume"].rolling(window=72, min_periods=72).min() > 0

        if not self.optimize_buy_hma:
            dataframe["hma_offset_buy"] = (
                tv_hma(dataframe, int(self.base_nb_candles_buy_hma.value))
                * self.low_offset_hma.value
            )

        if not self.optimize_buy_ema:
            dataframe["ema_offset_buy"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value))
                * self.low_offset_ema.value
            )

        if not self.optimize_buy_ema2:
            dataframe["ema_offset_buy2"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_buy_ema2.value))
                * self.low_offset_ema2.value
            )

        if not self.optimize_buy_volatility:
            df_std = dataframe["close"].rolling(int(self.buy_length_volatility.value)).std()
            dataframe["volatility"] = (df_std > self.buy_min_volatility.value) & (
                df_std < self.buy_max_volatility.value
            )

        if not self.optimize_sell_hma:
            dataframe["hma_offset_sell"] = (
                tv_hma(dataframe, int(self.base_nb_candles_sell_hma.value))
                * self.high_offset_hma.value
            )

        if not self.optimize_sell_ema:
            dataframe["ema_offset_sell"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_sell_ema.value))
                * self.high_offset_ema.value
            )

        if not self.optimize_sell_ema2:
            dataframe["ema_offset_sell2"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_sell_ema2.value))
                * self.high_offset_ema2.value
            )

        if not self.optimize_sell_ema3:
            dataframe["ema_offset_sell3"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_sell_ema3.value))
                * self.high_offset_ema3.value
            )

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        if self.optimize_buy_hma:
            dataframe["hma_offset_buy"] = (
                tv_hma(dataframe, int(self.base_nb_candles_buy_hma.value))
                * self.low_offset_hma.value
            )

        if self.optimize_buy_ema:
            dataframe["ema_offset_buy"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value))
                * self.low_offset_ema.value
            )

        if self.optimize_buy_ema2:
            dataframe["ema_offset_buy2"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_buy_ema2.value))
                * self.low_offset_ema2.value
            )

        if self.optimize_buy_volatility:
            df_std = dataframe["close"].rolling(int(self.buy_length_volatility.value)).std()
            dataframe["volatility"] = (df_std > self.buy_min_volatility.value) & (
                df_std < self.buy_max_volatility.value
            )

        dataframe.loc[:, "buy_tag"] = ""
        dataframe.loc[:, "buy"] = 0

        add_check = (
            dataframe["live_data_ok"]
            & dataframe["age_filter_ok_1d"]
            & dataframe["volatility"]
            & (dataframe["close"] < dataframe["ema_offset_buy"])
            & (dataframe["volume"] > 0)
            & (dataframe["sqzmi"] == False)
            & (dataframe["rsi_fast"] < self.buy_rsi_fast_1.value)
            & (dataframe["rsi"] < self.buy_rsi_1.value)
        )

        buy_offset_hma = dataframe["close"] < dataframe["hma_offset_buy"]
        dataframe.loc[buy_offset_hma, "buy_tag"] += "hma "
        conditions.append(buy_offset_hma)

        buy_offset_ema2 = dataframe["close"] < dataframe["ema_offset_buy2"]
        dataframe.loc[buy_offset_ema2, "buy_tag"] += "ema2 "
        conditions.append(buy_offset_ema2)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions) & add_check,
                "buy",
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.optimize_sell_hma:
            dataframe["hma_offset_sell"] = (
                tv_hma(dataframe, int(self.base_nb_candles_sell_hma.value))
                * self.high_offset_hma.value
            )

        if self.optimize_sell_ema:
            dataframe["ema_offset_sell"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_sell_ema.value))
                * self.high_offset_ema.value
            )

        if self.optimize_sell_ema2:
            dataframe["ema_offset_sell2"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_sell_ema2.value))
                * self.high_offset_ema2.value
            )

        if self.optimize_sell_ema3:
            dataframe["ema_offset_sell3"] = (
                ta.EMA(dataframe, int(self.base_nb_candles_sell_ema3.value))
                * self.high_offset_ema3.value
            )

        dataframe.loc[:, "exit_tag"] = ""
        conditions = []

        sell_cond_1 = (dataframe["close"] > dataframe["hma_offset_sell"]) & (
            dataframe["volume"] > 0
        )
        conditions.append(sell_cond_1)
        dataframe.loc[sell_cond_1, "exit_tag"] += "HMA_1 "

        sell_cond_3 = (
            (dataframe["close"] < dataframe["ema_offset_sell3"]).rolling(2).sum() == 2
        ) & (dataframe["volume"] > 0)
        conditions.append(sell_cond_3)
        dataframe.loc[sell_cond_3, "exit_tag"] += "EMA_3 "

        sell_cond_2 = (dataframe["close"] > dataframe["ema_offset_sell"]) & (
            dataframe["volume"] > 0
        )
        conditions.append(sell_cond_2)
        dataframe.loc[sell_cond_2, "exit_tag"] += "EMA_1 "

        sell_cond_4 = (dataframe["close"] < dataframe["ema_offset_sell2"]) & (
            dataframe["volume"] > 0
        )
        conditions.append(sell_cond_4)
        dataframe.loc[sell_cond_4, "exit_tag"] += "EMA_2 "

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        return dataframe
