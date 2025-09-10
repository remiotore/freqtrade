
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

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from colorama import Fore, Style

logger = logging.getLogger(__name__)


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def calculate_percentage_change(start_value: float, final_value: float) -> float:
    if final_value == 0:
        return 0
    return (final_value - start_value) / start_value * 100


def calculate_increment(n: float, pct_increment: float) -> float:
    return n + (n * pct_increment / 100)


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


def green_text(text):
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"


def yellow_text(text):
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"


def get_cmd_pair(pair):
    s = pair.split("/")
    return s[0] + "\\/" + s[1]


def in_range(ongoing_close, green, red, green_distance, red_distance):
    if ongoing_close > green:
        return calculate_distance_percentage(ongoing_close, green) <= green_distance and \
               calculate_distance_percentage(ongoing_close, red) <= red_distance
    return calculate_distance_percentage(ongoing_close, red) <= red_distance


class RSIDropDNS(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '5m'



    alarm_emitted = dict()
    max_bars_back = 500




    process_only_new_candles = True





    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        if pair not in self.alarm_emitted:
            self.alarm_emitted[pair] = False
        short_df = dataframe.tail(self.max_bars_back)

        if self.dp and \
                self.dp.runmode.value in ('live', 'dry_run'):
            short_df = short_df.append(get_ongoing_candle(pair=pair, timeframe=self.timeframe), ignore_index=True)

        previous_range = short_df["open"].shift(1) - short_df["close"].shift(1)

        rsi_limit = 11.0
        green, red = self.get_closest_bull_zone(previous_range=previous_range, dataframe=short_df, limit=rsi_limit)

        ongoing_close = short_df["close"].iloc[-1]

        if self.timeframe == "1h":
            green_distance = 0.3
            red_distance = 1.3
            increment_pct = 0.5
            tv_interval = 60
            drop_rsi_threshold = 30
        elif self.timeframe == "30m":
            green_distance = 0.3
            red_distance = 1.3
            increment_pct = 0.5
            tv_interval = 30
            drop_rsi_threshold = 30
        elif self.timeframe == "5m":
            green_distance = 0.3
            red_distance = 2
            increment_pct = 0.3
            tv_interval = 5
            drop_rsi_threshold = 25
        elif self.timeframe == "1m":
            green_distance = 0.3
            red_distance = 2
            increment_pct = 0.3
            tv_interval = 1
            drop_rsi_threshold = 30


        if green and red and in_range(ongoing_close, green, red, green_distance, red_distance) and \
                self.rsi_in_range(dataframe=dataframe, rsi_threshold=drop_rsi_threshold):
            if not self.alarm_emitted[pair]:
                self.alarm_emitted[pair] = True
                is_dry_run = "false"
                stake_amount = 25
                print(yellow_text(f"https://www.tradingview.com/chart/?symbol=binance:{pair.replace('/', '')}&interval={tv_interval}"))
                if calculate_distance_percentage(green, red) < increment_pct:
                    cmd = f"export buy_zone_price_top={green} buy_zone_price_bottom={red} " \
                          f"pair=\"{get_cmd_pair(pair)}\" is_dry_run={is_dry_run} stake_amount={stake_amount} && ./buynstoploss.sh"
                else:
                    cmd = f"export buy_zone_price_top={calculate_increment(red, increment_pct)} buy_zone_price_bottom={red} " \
                          f"pair=\"{get_cmd_pair(pair)}\" is_dry_run={is_dry_run} stake_amount={stake_amount} && ./buynstoploss.sh"
                print(green_text(cmd))
                desktop_notif_text = f"{pair} DNS found"
                os.system(
                    f"notify-send \"{desktop_notif_text.upper()}\" -t 10000 -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
        else:
            self.alarm_emitted[pair] = False

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

    def rsi_in_range(self, dataframe, rsi_threshold):
        rsi = ta.RSI(dataframe, timeperiod=14).tolist()
        lookback_candles = 12
        last_rsi = rsi[-1]
        result = False
        for i in range(2, lookback_candles + 1):
            if calculate_percentage_change(last_rsi, rsi[-i]) > rsi_threshold:
                result = True
        return result

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
