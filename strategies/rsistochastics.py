
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class rsistochastics(IStrategy):


    stoploss = -100.0
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

        dataframe.loc[
            (
                (dataframe["rsi"] > 50)
                & (dataframe["slowd"] > 50)
                & (dataframe["macdhist"] > 0)
            ),
            "buy",
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["rsi"] < 50)
                | (dataframe["slowd"] < 50)
                | (dataframe["macdhist"] < 0)
            ),
            "sell",
        ] = 1


        return dataframe