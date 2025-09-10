
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import os

import numpy as np

from freqtrade.rpc import RPCMessageType
from beepy import beep
from colorama import Fore, Style


def green(text):
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"


class CryptosRejectBitcoin(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '5m'

    process_only_new_candles = True

    alarm_emitted = False

    def informative_pairs(self):
        return [("BTC/USDT", "5m")
                ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        btc_15m = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe="5m")
        last_btc_open = btc_15m["open"].iloc[-1]
        last_btc_close = btc_15m["close"].iloc[-1]

        last_open = dataframe["open"].iloc[-1]
        last_close = dataframe["close"].iloc[-1]

        if (last_close > last_open) and (last_btc_close < last_btc_open):
            tv_pair = pair.replace("/", "")
            if not self.alarm_emitted:
                beep(6)
                self.alarm_emitted = True
            tv_url = f"https://www.tradingview.com/chart/?symbol=binance:{tv_pair}&interval=5"
            alarm_text = f"{tv_pair} IS REJECTING BITCOIN IN {self.timeframe}"
            print(green(tv_url))
            os.system(f"notify-send \"{alarm_text.upper()}\" -t 10000 -i /usr/share/icons/gnome/48x48/actions/stock_about.png")

        if metadata["pair"] == "AAVE/USDT":
            self.alarm_emitted = False

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
