# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
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

# --------------------------------
# Add your lib to import here
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade, Order


class HedgeStrategy(IStrategy):
    """
    author@: Bryant Suen
    github@: https://github.com/BryantSuen

    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    inf_timeframe = "1h"

    can_short: bool = True

    order_types = {"entry": "limit", "exit": "market",
                   "stoploss": "market", "stoploss_on_exchange": False}
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    pair_trade_list = {
        "BTC/USDT:USDT": "ETH/USDT:USDT"
    }

    # Minimal ROI designed for the strategy.
    minimal_roi = {"0": 10}

    stoploss = -0.2

    process_only_new_candles = True
    use_exit_signal = True

    startup_candle_count: int = 100

    # DCA Settings
    position_adjustment_enable = True
    max_entry_position_adjustment = 1
    max_dca_multiplier = 2

    # HyperParameters
    buy_params = {
        "corr_threshold": 0.7,
        "hedge_long": -3,
        "hedge_short": 3,
        "stoploss_long": -6,
        "stoploss_short": 6,
    }
    hedge_long = DecimalParameter(-5, -1, default=buy_params["hedge_long"], decimals=1, space="buy", optimize=True)
    hedge_short = DecimalParameter(1, 5, default=buy_params["hedge_short"], decimals=1, space="buy", optimize=True)

    stoploss_long = DecimalParameter(-10, -3, default=buy_params["stoploss_long"], decimals=1, space="buy", optimize=True)
    stoploss_short = DecimalParameter(3, 10, default=buy_params["stoploss_short"], decimals=1, space="buy", optimize=True)

    corr_threshold = DecimalParameter(0.5, 1, default=buy_params["corr_threshold"], decimals=1, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        is_btc = metadata["pair"] == "BTC/USDT:USDT"
        current_pair = metadata["pair"]
        counter_pair = "ETH/USDT:USDT" if is_btc else "BTC/USDT:USDT"
        ols_window = 64

        informative = self.dp.get_pair_dataframe(
            pair=current_pair, timeframe=self.inf_timeframe)
        informative_counter = self.dp.get_pair_dataframe(
            pair=counter_pair, timeframe=self.inf_timeframe)
        counter_df = self.dp.get_pair_dataframe(
            pair=counter_pair, timeframe=self.timeframe)

        # counter = hedge_ratio * pair + C
        informative_counter_c = informative_counter[["close", "date"]]
        informative_counter_c = informative_counter_c.rename(
            columns={"close": "close_counter"})
        informative = pd.merge_ordered(
            informative, informative_counter_c, fill_method="ffill", left_on="date", right_on="date", how="left")
        if is_btc:
            informative["hedge_ratio"] = 1
            informative["counter_ratio"] = informative["close"].rolling(ols_window).apply(
                lambda x: np.polyfit((informative["close_counter"].iloc[x.index]).astype('float'), x, 1)[0], raw=False)
        else:
            informative["hedge_ratio"] = informative["close"].rolling(ols_window).apply(
                lambda x: np.polyfit(x, (informative["close_counter"].iloc[x.index]).astype('float'), 1)[0], raw=False)
            informative["counter_ratio"] = 1

        informative["corr"] = informative["close"].rolling(
            ols_window).corr(informative["close_counter"])

        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        dataframe["hedged_value"] = dataframe[f"hedge_ratio_{self.inf_timeframe}"] * \
            dataframe["close"] - \
            dataframe[f"counter_ratio_{self.inf_timeframe}"] * \
            counter_df["close"]
        # normalize hedge value in a rolling window
        dataframe["hedged_norm"] = (dataframe["hedged_value"] - dataframe["hedged_value"].rolling(
            ols_window).mean()) / dataframe["hedged_value"].rolling(ols_window).std()

        # Convert to stake ratio
        # stake_ratio = hedge_ratio * close / (hedge_ratio * close + counter_ratio * counter_close)
        dataframe["stake_ratio"] = dataframe[f"hedge_ratio_{self.inf_timeframe}"] * dataframe["close"] / (
            dataframe[f"hedge_ratio_{self.inf_timeframe}"] * dataframe["close"] + dataframe[f"counter_ratio_{self.inf_timeframe}"] * counter_df["close"])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        triggers_long = {}
        triggers_short = {}

        guards_long = []
        guards_short = []

        pair = metadata["pair"]

        if pair in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
            # 1. Add entry conditions
            triggers_long[f"hedge_long"] = dataframe["hedged_norm"] < self.hedge_long.value
            triggers_short["hedge_short"] = dataframe["hedged_norm"] > self.hedge_short.value

            # 2. Add entry guards
            guards_long.append(dataframe["volume"] > 0)
            guards_short.append(dataframe["volume"] > 0)
            
            # abs correlation
            guards_long.append(abs(dataframe[f"corr_{self.inf_timeframe}"]) > self.corr_threshold.value)
            guards_short.append(abs(dataframe[f"corr_{self.inf_timeframe}"]) > self.corr_threshold.value)

        # 3. Set the entry
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

        # 1. Add exit conditions
        triggers_long["profit_long"] = qtpylib.crossed_above(
            dataframe["hedged_norm"], 0)
        triggers_short["profit_short"] = qtpylib.crossed_below(
            dataframe["hedged_norm"], 0)

        triggers_long["stoploss_long"] = qtpylib.crossed_below(
            dataframe["hedged_norm"], self.stoploss_long.value)
        triggers_short["stoploss_short"] = qtpylib.crossed_above(
            dataframe["hedged_norm"], self.stoploss_short.value)

        # 2. Add exit guards
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

        return 3

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

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        stake_ratio = dataframe["stake_ratio"].iloc[-1]
        return proposed_stake / self.max_dca_multiplier * stake_ratio

    @property
    def plot_config(self):
        return {"main_plot": {}, "subplots": {
            "Hedge": {
                "hedged_norm": {"color": "blue"},
            }
        }}
