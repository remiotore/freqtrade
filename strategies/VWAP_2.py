
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


class VWAP_2(IStrategy):
    minimal_roi = {
        "0": 0.05
    }

    stoploss = -0.01

    timeframe = '1h'



    process_only_new_candles = True




    bought_at = dict()
    btc_rsi = None
    btc_macd_hist = None

    def informative_pairs(self):
        return [("BTC/USDT", "1h"),
                ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        if self.dp and \
                self.dp.runmode.value in ('live', 'dry_run'):
            if pair not in self.bought_at:
                self.bought_at[pair] = None
            if pair == "ETH/USDT":
                btc_df = get_candles(pair="BTC/USDT", timeframe=self.timeframe)
                self.btc_macd_hist = ta.MACD(btc_df)["macdhist"].tolist()


            vwap = qtpylib.rolling_vwap(dataframe, window=14).tolist()
            last_closed_candle_open = dataframe["open"].iloc[-1]
            last_closed_candle_close = dataframe["close"].iloc[-1]
            last_closed_candle_vwap = vwap[-1]

            buy_criteria, sell_criteria = False, False

            if not self.bought_at[pair] and \
                    last_closed_candle_open < last_closed_candle_vwap < last_closed_candle_close and \
                    self.btc_macd_hist[-1] > self.btc_macd_hist[-2]:


                buy_criteria = True
                self.bought_at[pair] = self.dp.ticker(pair)["last"]
            if self.bought_at[pair] and \
                    last_closed_candle_close < last_closed_candle_vwap < last_closed_candle_open:
                sell_criteria = True
                sold_at = self.dp.ticker(pair)["last"]
                profit = calculate_percentage_change(self.bought_at[pair], sold_at)
                msg = f"Sold {pair} at {self.timeframe} profit: {profit}"
                notify_critical(msg)
                print(yellow_text(generate_tv_url(pair, self.timeframe)))
                self.bought_at[pair] = None

            dataframe["buy_criteria"] = buy_criteria
            dataframe["sell_criteria"] = sell_criteria
        else:
            btc_df = self.dp.get_pair_dataframe(pair="BTC/USDT",
                                                     timeframe="1h")
            dataframe["btc_macd_hist"] = ta.MACD(btc_df)["macdhist"]
            dataframe["btc_rsi"] = ta.RSI(btc_df, timeperiod=14)
            dataframe["vwap"] = qtpylib.rolling_vwap(dataframe, window=14)

            dataframe["buy_criteria"] = (
                    (dataframe["open"] < dataframe["vwap"]) &
                    (dataframe["vwap"] < dataframe["close"]) &
                    (dataframe["btc_macd_hist"] > dataframe["btc_macd_hist"].shift(1)) &
                    (dataframe["btc_rsi"] > dataframe["btc_rsi"].shift(1))
            )




        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["buy_criteria"]
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["sell_criteria"]
            ),
            'sell'] = 1
        return dataframe
