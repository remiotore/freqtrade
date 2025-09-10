
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import os

import numpy as np

from freqtrade.rpc import RPCMessageType
from beepy import beep
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


class VWAPAlarm(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '3m'
    process_only_new_candles = True

    alarm_emitted = dict()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        if pair not in self.alarm_emitted:
            self.alarm_emitted[pair] = False




        df_2h = resample_to_interval(dataframe, 120)
        df_2h['vwap'] = qtpylib.rolling_vwap(df_2h, window=14)






        def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
            distance = abs(current_price - green_line_price)
            return distance * 100 / current_price



        pct = 1.0
        vwap = df_2h["vwap"].iloc[-1]
        price = df_2h["close"].iloc[-1]
        previous_vwap = df_2h["vwap"].iloc[-2]

        previous_low = df_2h["low"].iloc[-2]

        if previous_low > previous_vwap and (vwap + (vwap * pct / 100)) >= price >= vwap:
            if not self.alarm_emitted[pair]:
                binance_pair = pair.replace("/", "_")
                beep(1)
                os.system(f'xdg-open https://www.binance.com/en/trade/{binance_pair}?layout=pro&type=spot')
                print(f'{pair} {calculate_distance_percentage(price, vwap)}')
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
