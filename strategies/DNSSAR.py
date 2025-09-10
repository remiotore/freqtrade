
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import os

import numpy as np

import talib.abstract as ta

from freqtrade.rpc import RPCMessageType
from beepy import beep
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval

from colorama import Fore, Style

from freqtrade.utils.binance_rest_api import get_candles
from freqtrade.utils.notifications import notify_critical
from freqtrade.utils.tradingview import generate_tv_url


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def calculate_percentage_change(start_value: float, final_value: float) -> float:
    if final_value == 0:
        return 0
    return (final_value - start_value) / start_value * 100


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


def yellow_text(text):
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"


class DNSSAR(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sar"] = ta.SAR(dataframe).tolist()
        last_sar = dataframe["sar"].iloc[-1]
        last_close = dataframe["close"].iloc[-1]
        if self.timeframe == "15m":

        elif self.timeframe == "1h":


        print(dataframe["sar"])
        return dataframe

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
