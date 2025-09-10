



import talib.abstract as ta
import pandas_ta as pta
import numpy as np  # noqa
import pandas as pd  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)

class powerxhopt_2(IStrategy):


    stoploss = -1
    timeframe = "1d"
    minimal_roi = {"0": 100.}

    order_types = {
        "buy": "limit",
        "sell": "limit",
        "emergencysell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    plot_config = {

        "main_plot": {
            "sma": {},
        },
        "subplots": {

            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "STOCH": {
                "slowd": {"color": "red"},
            },
        },
    }

    rsi_hline = IntParameter(30, 70, default=50, space="buy")
    stoch_hline = IntParameter(30, 70, default=50, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["SMA"] = ta.SMA(dataframe, timeperiod=20)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=7)


        stoch = ta.STOCH(
            dataframe,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )
        dataframe["slowd"] = stoch["slowd"]
        dataframe["slowk"] = stoch["slowk"]


        macd = ta.MACD(
            dataframe,
            fastperiod=12,
            fastmatype=0,
            slowperiod=26,
            slowmatype=0,
            signalperiod=9,
            signalmatype=0,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        conditions.append(
            (dataframe['rsi'] > self.rsi_hline.value )
            & (dataframe['slowd'] > self.stoch_hline.value )
            & (dataframe["macdhist"] > 0)
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        conditions = []
        conditions.append(




            (dataframe['slowd'] < self.stoch_hline.value )

            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
