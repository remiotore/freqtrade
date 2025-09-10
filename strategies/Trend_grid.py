



import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
from typing import Optional, Union, Tuple
import pandas_ta as pd_ta

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)


from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade, Order


class Trend_grid(IStrategy):
    """
    author@: Bryant Suen
    github@: https://github.com/BryantSuen

    """

    INTERFACE_VERSION = 3

    timeframe = "5m"

    can_short: bool = True

    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    minimal_roi = {"0": 100}

    stoploss = -100






    process_only_new_candles = True
    use_exit_signal = False

    startup_candle_count: int = 100

    position_adjustment_enable = True
    max_entry_position_adjustment = 4
    max_dca_multiplier = 5
    grid_ratio = 0.015
    drawdown_ratio = 0.01



    buy_params = {
        "long_cti_32": -0.56,
        "long_rsi_32": 25,
        "long_rsi_fast_32": 59,
        "long_sma15_32": 0.95,
        "short_cti_32": 0.37,
        "short_rsi_32": 69,
        "short_rsi_fast_32": 40,
        "short_sma15_32": 1.01,
    }

    sell_params = {
        "exit_long_fastx": 96,
        "exit_short_fastx": 0,
    }

    is_optimize_32 = True

    long_rsi_fast_32 = IntParameter(20, 70, default=buy_params["long_rsi_fast_32"], space="buy", optimize=is_optimize_32)
    long_rsi_32 = IntParameter(15, 50, default=buy_params["long_rsi_32"], space="buy", optimize=is_optimize_32)
    long_sma15_32 = DecimalParameter(0.900, 1, default=buy_params["long_sma15_32"], decimals=2, space="buy", optimize=is_optimize_32)
    long_cti_32 = DecimalParameter(-1, 0, default=buy_params["long_cti_32"], decimals=2, space="buy", optimize=is_optimize_32)

    short_rsi_fast_32 = IntParameter(30, 80, default=buy_params["short_rsi_fast_32"], space="buy", optimize=is_optimize_32)
    short_rsi_32 = IntParameter(50, 85, default=buy_params["short_rsi_32"], space="buy", optimize=is_optimize_32)
    short_sma15_32 = DecimalParameter(1, 1.1, default=buy_params["short_sma15_32"], decimals=2, space="buy", optimize=is_optimize_32)
    short_cti_32 = DecimalParameter(0, 1, default=buy_params["short_cti_32"], decimals=2, space="buy", optimize=is_optimize_32)

    exit_long_fastx = IntParameter(50, 100, default=sell_params["exit_long_fastx"], space="sell", optimize=True)
    exit_short_fastx = IntParameter(0, 50, default=sell_params["exit_short_fastx"], space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["sma_15"] = ta.SMA(dataframe, timeperiod=15)
        dataframe["cti"] = pd_ta.cti(dataframe["close"], length=20)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe["fastk"] = stoch_fast["fastk"]

        dataframe["atr"] = ta.ATR(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        triggers_long = {}
        triggers_short = {}

        guards_long = []
        guards_short = []


        triggers_long["e0v1e_long"] = (
            (dataframe["rsi_slow"] < dataframe["rsi_slow"].shift(1))
            & (dataframe["rsi_fast"] < self.long_rsi_fast_32.value)
            & (dataframe["rsi"] > self.long_rsi_32.value)
            & (dataframe["close"] < dataframe["sma_15"] * self.long_sma15_32.value)
            & (dataframe["cti"] < self.long_cti_32.value)
        )

        triggers_short["e0v1e_short"] = (
            (dataframe["rsi_slow"] > dataframe["rsi_slow"].shift(1))
            & (dataframe["rsi_fast"] > self.short_rsi_fast_32.value)
            & (dataframe["rsi"] < self.short_rsi_32.value)
            & (dataframe["close"] > dataframe["sma_15"] * self.short_sma15_32.value)
            & (dataframe["cti"] > self.short_cti_32.value)
        )

        guards_long.append(dataframe["volume"] > 0)
        guards_short.append(dataframe["volume"] > 0)

        if triggers_long:
            for trigger_name, trigger in triggers_long.items():
                dataframe.loc[(trigger & reduce(lambda x, y: x & y, guards_long)), ["enter_long", "enter_tag"]] = [
                    1,
                    trigger_name,
                ]

        if triggers_short:
            for trigger_name, trigger in triggers_short.items():
                dataframe.loc[(trigger & reduce(lambda x, y: x & y, guards_short)), ["enter_short", "enter_tag"]] = [
                    1,
                    trigger_name,
                ]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        triggers_long = {}
        triggers_short = {}

        guards_long = []
        guards_short = []


        guards_long.append(dataframe["volume"] > 0)
        guards_short.append(dataframe["volume"] > 0)

        if triggers_long:
            for trigger_name, trigger in triggers_long.items():
                dataframe.loc[(trigger & reduce(lambda x, y: x & y, guards_long)), ["exit_long", "exit_tag"]] = [
                    1,
                    trigger_name,
                ]

        if triggers_short:
            for trigger_name, trigger in triggers_short.items():
                dataframe.loc[(trigger & reduce(lambda x, y: x & y, guards_short)), ["exit_short", "exit_tag"]] = [
                    1,
                    trigger_name,
                ]

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:

        return 1

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:

        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:

        filled_entries = trade.select_filled_orders(trade.entry_side)

        last_grid_price = filled_entries[-1].price

        if trade.is_short:
            if current_rate < last_grid_price * (1 - self.grid_ratio):
                try:
                    stake_amount = filled_entries[0].stake_amount
                    return stake_amount, "grid_increase"
                except Exception as exception:
                    return None
            
            if current_rate > last_grid_price * (1 + self.grid_ratio):
                return -trade.stake_amount, "grid_exit"
            
        else:
            if current_rate > last_grid_price * (1 + self.grid_ratio):
                try:
                    stake_amount = filled_entries[0].stake_amount
                    return stake_amount, "dca_increase"
                except Exception as exception:
                    print(f"Exception: {exception}")
                    return None
            
            if current_rate < last_grid_price * (1 - self.grid_ratio):
                return -trade.stake_amount, "grid_exit"

        return None

    @property
    def plot_config(self):
        return {"main_plot": {}, "subplots": {}}
