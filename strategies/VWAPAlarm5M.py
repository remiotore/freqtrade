
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import os

from colorama import Fore, Style

import numpy as np

from freqtrade.rpc import RPCMessageType
from beepy import beep
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval

def green(text):
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


class VWAPAlarm5M(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '5m'
    process_only_new_candles = True

    alarm_emitted = dict()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        if pair not in self.alarm_emitted:
            self.alarm_emitted[pair] = False




        dataframe['vwap'] = qtpylib.rolling_vwap(dataframe, window=14)








        pct = 0.2
        vwap = dataframe["vwap"].iloc[-1]
        current_close = dataframe["close"].iloc[-1]
        previous_vwap = dataframe["vwap"].iloc[-2]
        previous_close = dataframe["close"].iloc[-2]
        if previous_close < previous_vwap and current_close > vwap:
            if not self.alarm_emitted[pair]:
                binance_pair = pair.replace("/", "_")
                ticker = self.dp.ticker(pair)
                last_price = ticker['last']
                distance = calculate_distance_percentage(last_price, vwap)
                if distance <= pct:
                    beep(3)
                    binance_link = f'https://www.binance.com/en/trade/{binance_pair}?layout=pro&type=spot'

                    print(green(f'{pair} {round(calculate_distance_percentage(last_price, vwap), 2)} {binance_link}'))
            self.alarm_emitted[pair] = True
        else:
            self.alarm_emitted[pair] = False











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
