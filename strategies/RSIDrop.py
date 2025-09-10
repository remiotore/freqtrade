
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


def calculate_distance_percentage_no_abs(current_price: float, green_line_price: float) -> float:
    distance = current_price - green_line_price
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


class RSIDrop(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '5m'



    alarm_emitted = dict()




    process_only_new_candles = True





    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        if pair not in self.alarm_emitted:
            self.alarm_emitted[pair] = False

        rsi_threshold = 30
        tv_interval = 5
        if self.rsi_in_range(dataframe=dataframe, rsi_threshold=rsi_threshold):
            if not self.alarm_emitted[pair]:
                self.alarm_emitted[pair] = True
                print(yellow_text(f"https://www.tradingview.com/chart/?symbol=binance:{pair.replace('/', '')}&interval={tv_interval}"))
                desktop_notif_text = f"{pair} RSI drop found"
                os.system(
                    f"notify-send \"{desktop_notif_text.upper()}\" -t 10000 -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
        else:
            self.alarm_emitted[pair] = False

        return dataframe

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
