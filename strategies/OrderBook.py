
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import os

import numpy as np

from freqtrade.rpc import RPCMessageType
from beepy import beep
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval
import math
import collections


def truncate_ceil(x, base):
    return base * math.ceil(x / base)


def truncate_floor(x, base):
    return base * math.floor(x / base)


def aggregate_ob_asks(ob, aggregated_ob, base):
    key = "asks"
    for n in ob[key]:
        truncated = truncate_ceil(n[0], base)
        if truncated not in aggregated_ob[key]:
            aggregated_ob[key][truncated] = 0
        aggregated_ob[key][truncated] += n[1]


def aggregate_ob_bids(ob, aggregated_ob, base):
    key = "bids"
    for n in ob[key]:
        truncated = truncate_floor(n[0], base)
        if truncated not in aggregated_ob[key]:
            aggregated_ob[key][truncated] = 0
        aggregated_ob[key][truncated] += n[1]


def is_in_alert_bid_range(current_price, bid_price):
    return current_price <= (bid_price + (bid_price * 0.65 / 100))


def is_in_alert_ask_range(current_price, ask_price):
    return current_price >= (ask_price - (ask_price * 0.65 / 100))


def calculate_ob_average(aggregated_ob, key):
    acum = 0
    for _, v in aggregated_ob[key].items():
        acum += v
    return acum / len(aggregated_ob[key])


class OrderBook(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '1h'

    alarm_multiplier = 3

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        ob = self.dp.orderbook(metadata['pair'], 1000)
        ticker = self.dp.ticker(pair)
        current_price = ticker["last"]

        aggregated_ob = dict()
        aggregated_ob["bids"] = collections.OrderedDict()
        aggregated_ob["asks"] = collections.OrderedDict()
        aggregate_ob_bids(ob, aggregated_ob, 50)
        aggregate_ob_asks(ob, aggregated_ob, 50)

        asks_average = calculate_ob_average(aggregated_ob, 'asks')

        asks_threshold = 80.0
        asks_alarm_text = ""
        for k, v in aggregated_ob["asks"].items():
            if v > asks_threshold and is_in_alert_ask_range(current_price, k):

                asks_alarm_text += f"SELL PRESSURE {int(v)} at {k}"
                break
        if asks_alarm_text != "":
            os.system(
                f"notify-send \"{asks_alarm_text.upper()}\" -t 10000 -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            print(asks_alarm_text.upper())

        bids_average = calculate_ob_average(aggregated_ob, 'bids')

        bids_threshold = 80.0
        bids_alarm_text = ""
        for k, v in aggregated_ob["bids"].items():
            if v > bids_threshold and is_in_alert_bid_range(current_price, k):

                bids_alarm_text += f"BUY PRESSURE {int(v)} at {k}"
                break

        if bids_alarm_text != "":
            os.system(
                f"notify-send \"{bids_alarm_text.upper()}\" -t 10000 -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            print(bids_alarm_text.upper())
        print("")
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
