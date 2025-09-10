
import os
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import talib.abstract as ta

import numpy as np

from freqtrade.utils.tradingview import generate_tv_url
from freqtrade.utils.binance_rest_api import get_ongoing_candle

from typing import List

import logging

logger = logging.getLogger(__name__)


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def calculate_percentage_change(start_value: float, final_value: float) -> float:
    return (final_value - start_value) / start_value * 100


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


class DNSTrader(IStrategy):
    minimal_roi = {
        "0": 0.99
    }

    stoploss = -0.99

    timeframe = '1m'



    max_bars_back = 500
    max_simultaneous_engulf_patterns = 10
    BTC_ETH = ["BTC", "ETH"]

    last_buy_red_line = None
    position_is_open = False

    show_buy_message = False
    show_sell_message = False

    def __init__(self, config: dict) -> None:
        self.btc_eth_alert_percentage = float(config['btc_eth_alert_percentage'])
        self.altcoins_alert_percentage = float(config['altcoins_alert_percentage'])
        self.btc_eth_restart_alert_percentage = float(config['btc_eth_restart_alert_percentage'])
        self.altcoins_restart_alert_percentage = float(config['altcoins_restart_alert_percentage'])
        super().__init__(config)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        if self.show_buy_message:
            msg = f"{pair} opened new position at {self.timeframe}"
            os.system(f"notify-send \"{msg}\" -t 10000 -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            self.show_buy_message = False
        if self.show_sell_message:
            msg = f"{pair} stop loss at {self.timeframe}"
            os.system(f"notify-send \"{msg}\" -t 10000 -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            self.show_sell_message = False

        short_df = dataframe.tail(self.max_bars_back)
        short_df = short_df.append(get_ongoing_candle(pair=pair, timeframe=self.timeframe), ignore_index=True)

        previous_range = short_df["open"].shift(1) - short_df["close"].shift(1)
        green, red = self.get_closest_bull_zone(previous_range=previous_range, dataframe=short_df)

        buy_criteria = False
        sell_criteria = False

        ongoing_close = short_df["close"].iloc[-1]

        print(f"g {green} r {red} pos_r {self.last_buy_red_line}")
        if not self.position_is_open:
            if green and red and ongoing_close < green and calculate_distance_percentage(ongoing_close, red) <= 0.4:
                self.position_is_open = True
                self.last_buy_red_line = red
                self.show_buy_message = True
                buy_criteria = True
        else:
            if ongoing_close < self.last_buy_red_line:
                self.position_is_open = False
                self.last_buy_red_line = None
                self.show_sell_message = True
                sell_criteria = True

        dataframe["buy_criteria"] = buy_criteria
        dataframe["sell_criteria"] = sell_criteria

        return dataframe

    def get_closest_bull_zone(self, previous_range: Series, dataframe, limit: float):
        open = dataframe["open"]
        low = dataframe["low"]
        close = dataframe["close"]

        is_bull_engulf = (
                (previous_range > 0) &
                (close > open.shift(1))
        )

        bull_engulf_low = np.where(low < low.shift(1), low, low.shift(1))

        low_list = low.tolist()
        min_low_to_end = []
        for i in range(0, len(low_list)):
            min_low_to_end.append(min(low_list[i:]))
        dataframe["min_low_to_end"] = min_low_to_end

        rsi = ta.RSI(dataframe, timeperiod=14).tolist()
        next_4_candles_rsi_change = [0.0] * len(rsi)
        for i in range(0, len(rsi) - 4):
            next_4_candles_rsi_change[i] = calculate_percentage_change(
                start_value=rsi[i], final_value=rsi[i + 4]
            )
        dataframe["next_4_candles_rsi_change"] = next_4_candles_rsi_change

        dataframe["green_line"] = np.where(
            is_bull_engulf &
            (dataframe["min_low_to_end"] >= bull_engulf_low) &
            (dataframe["next_4_candles_rsi_change"].shift(1).abs() > limit),
            open.shift(1),
            np.nan
        )

        dataframe["red_line"] = np.where(
            dataframe["green_line"].isnull(),
            np.nan,
            bull_engulf_low
        )
        try:
            return dataframe["green_line"].dropna().iloc[-1], dataframe["red_line"].dropna().iloc[-1]
        except Exception as e:
            return None, None

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["buy_criteria"])
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["sell_criteria"]
            ),
            'sell'] = 1
        return dataframe
