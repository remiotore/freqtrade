

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce


class MacdStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 0.03024,
        "296": 0.02924,
        "596": 0.02545,
        "840": 0.02444,
        "966": 0.02096,
        "1258": 0.01709,
        "1411": 0.01598,
        "1702": 0.0122,
        "1893": 0.00732,
        "2053": 0.00493,
        "2113": 0,
    }


    stoploss = -0.04032

    trailing_stop = True




    timeframe = "5m"

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    order_types = {
        "buy": "limit",
        "sell": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"buy": "gtc", "sell": "gtc"}

    plot_config = {

        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
        },
        "subplots": {

            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["sell-rsi"] = dataframe["rsi"]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"]))
                & (dataframe["rsi"].rolling(8).min() < 41)
                & (dataframe["close"] > dataframe["ema200"])
                & (dataframe["volume"] > 0)
            ),
            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"]))
                & (dataframe["rsi"].rolling(8).max() > 93)
                & (dataframe["macd"] > 0)
                & (dataframe["volume"] > 0)
            ),
            "sell",
        ] = 1
        return dataframe
