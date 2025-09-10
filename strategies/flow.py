# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# --- Core Imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union, Tuple
from scipy.stats import zscore
import talib.abstract as ta
from functools import reduce

# --- Freqtrade Imports ---
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from freqtrade.persistence import Trade, Order


class Flow(IStrategy):
    """
    1. Volume Profile Value Areas
    2. Auction Market Theory
    3. Order Flow Imbalance
    """

    INTERFACE_VERSION = 3

    timeframe = "15m"

    can_short: bool = True

    minimal_roi = {"0": 10}
    stoploss = -0.8

    process_only_new_candles = True
    use_exit_signal = True

    startup_candle_count: int = 128

    order_types = {"entry": "limit", "exit": "limit", "stoploss": "market", "stoploss_on_exchange": True}
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # DCA Settings
    position_adjustment_enable = True
    max_entry_position_adjustment = 4
    max_dca_multiplier = 5

    buy_params = {
        "absorption_threshold": 0.15,
        "position_size_pct": 2.1,
        "stoploss_atr_multiplier": 2.2,
        "take_profit_atr_multiplier": 2.9,
        "volatility_threshold": 0.02,
    }

    # Risk Parameters
    position_size_pct = DecimalParameter(0.5, 3.0, decimals=1, default=buy_params["position_size_pct"], space="buy")
    volatility_threshold = DecimalParameter(0.02, 0.1, decimals=2, default=buy_params["volatility_threshold"], space="buy")

    # Strategy Parameters
    # vwap_window = IntParameter(20, 100, default=50, space="buy")
    absorption_threshold = DecimalParameter(0.02, 0.4, decimals=2, default=buy_params["absorption_threshold"], space="buy")

    # DCA Parameters
    take_profit_atr_multiplier = DecimalParameter(
        1, 4, default=buy_params["take_profit_atr_multiplier"], decimals=1, space="buy", optimize=True, load=True
    )
    stoploss_atr_multiplier = DecimalParameter(
        1, 3, default=buy_params["stoploss_atr_multiplier"], decimals=1, space="buy", optimize=True, load=True
    )

    leverage_value = 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core Market Structure Indicators
        dataframe = self.calculate_volume_profile(dataframe, profile_period=96)
        dataframe = self.calculate_auction_indicators(dataframe, vwap_period=32)
        dataframe = self.calculate_orderflow_imbalance(dataframe, ofi_period=32)

        # Volatility Measures
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["realized_vol"] = dataframe["close"].pct_change().rolling(24).std()

        return dataframe

    def calculate_volume_profile(self, dataframe: DataFrame, profile_period=96):
        # Volume Profile Value Area Calculation
        dataframe["vp_poc"] = (
            dataframe["volume"]
            .rolling(profile_period)
            .apply(lambda x: x.idxmax(), raw=False)
            .apply(lambda x: dataframe["close"][int(x)] if not np.isnan(x) else np.nan)
        )

        return dataframe

    def calculate_auction_indicators(self, dataframe: DataFrame, vwap_period=20):
        # Auction Market Theory Indicators
        rolling_close_volume = (
            ((dataframe["close"] + dataframe["high"] + dataframe["low"]) / 3 * dataframe["volume"]).rolling(vwap_period).sum()
        )
        rolling_volume = dataframe["volume"].rolling(vwap_period).sum()

        dataframe["vwap"] = rolling_close_volume / rolling_volume

        dataframe["market_balance"] = zscore(dataframe["close"] - dataframe["vwap"], nan_policy="omit")

        return dataframe

    def calculate_orderflow_imbalance(self, dataframe: DataFrame, ofi_period=20):
        # Order Flow Imbalance (simplified)
        bid_volume = (dataframe["volume"] * (dataframe["close"] > dataframe["open"])).rolling(ofi_period).sum()
        ask_volume = (dataframe["volume"] * (dataframe["close"] < dataframe["open"])).rolling(ofi_period).sum()

        dataframe["ofi"] = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        dataframe["volume_mean"] = dataframe["volume"].rolling(ofi_period).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        triggers_long = {}
        triggers_short = {}

        guards_long = []
        guards_short = []

        triggers_long["init_long"] = (
            (dataframe["ofi"].abs() > self.absorption_threshold.value)
            & (dataframe["volume"] > dataframe["volume_mean"])
            & (dataframe["close"] > dataframe["vp_poc"])
            & (dataframe["market_balance"] < -1)
            & (dataframe["realized_vol"] < self.volatility_threshold.value)
        )

        triggers_short["init_short"] = (
            (dataframe["ofi"].abs() > self.absorption_threshold.value)
            & (dataframe["volume"] > dataframe["volume_mean"])
            & (dataframe["close"] < dataframe["vp_poc"])
            & (dataframe["market_balance"] > 1)
            & (dataframe["realized_vol"] < self.volatility_threshold.value)
        )

        # 2. Add entry guards
        guards_long.append(dataframe["volume"] > 0)
        guards_short.append(dataframe["volume"] > 0)

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
        last_candle = dataframe.iloc[-1].squeeze()
        volatility_adj = last_candle["close"] / last_candle["atr"]
        return (self.position_size_pct.value / 100) * max_stake * volatility_adj

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

        # Update the stoploss ratio
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        atr = last_candle["atr"]  # type: ignore
        profit_ratio = atr / trade.open_rate * self.take_profit_atr_multiplier.value
        stoploss_ratio = -atr / trade.open_rate * self.stoploss_atr_multiplier.value

        # print(f"** Current Profit: {current_profit}, Profit Ratio: {profit_ratio}, Stoploss Ratio: {stoploss_ratio}")

        if trade.nr_of_successful_exits == 0:
            if current_profit > profit_ratio * self.leverage_value:
                # If the trade first reached the profit target, then exit half of the position
                return -(trade.stake_amount / 2), "exit_roi_first"

            if current_profit < stoploss_ratio * self.leverage_value:
                # If the trade is in loss, then exit the position
                return -trade.stake_amount, "exit_stoploss"

        if trade.nr_of_successful_exits == 1:
            # Process the second exit, this is equivalent to trailing stoploss

            updated_stoploss_ratio = max(stoploss_ratio, 0)
            if current_profit < updated_stoploss_ratio * self.leverage_value:
                return -trade.stake_amount, "exit_trailing"

        if trade.nr_of_successful_entries < self.max_entry_position_adjustment + 1:

            add_position_condition = (trade.is_short and last_candle["enter_short"]) or (
                (not trade.is_short) and last_candle["enter_long"]
            )

            if add_position_condition and current_profit > profit_ratio * self.leverage_value / self.max_dca_multiplier:
                filled_entries = trade.select_filled_orders(trade.entry_side)
                try:
                    stake_amount = filled_entries[0].stake_amount
                    return stake_amount, "dca_increase"
                except Exception as exception:
                    return None

        return None

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

        return self.leverage_value
