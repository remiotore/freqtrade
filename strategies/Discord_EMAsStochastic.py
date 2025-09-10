from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy import IntParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
import numpy as np


###########################################################################################################
##                WORK IN PROGRESS! USE AT YOUR OWN RISK!!!                                                                                                 ##
###########################################################################################################
##                EMAsStochastic by birrafondaio                                                         ##
##                                                                                                       ##
##    https://github.com/brokenseal                                                                      ##
##                                                                                                       ##                                                                                                       ##
###########################################################################################################
##                 GENERAL RECOMMENDATIONS                                                               ##
##                                                                                                       ##
##   For optimal performance, suggested to use 2 open trades, with unlimited stake.                      ##
##   With my pairlist which can be found in this repo.                                                   ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS 2 @brokenseal                                                                 ##
##                                                                                                       ##
##   Absolutely not required. However, will be accepted as a token of appreciation.                      ##
##                                                                                                       ##
##   ETH: 0x531CfE1fb299709726FE38259dD7dc666a2fA95D                                                     ##
##                                                                                                       ##
###########################################################################################################


class EMAsStochastic(IStrategy):
    timeframe = "1h"

    first_ema_length = IntParameter(6, 10, default=8, space="buy")
    second_ema_length = IntParameter(12, 16, default=14, space="buy")
    third_ema_length = IntParameter(40, 60, default=50, space="buy")
    max_open_hours = IntParameter(2, 48, default=12, space="buy")

    buy_params = {
        "first_ema_length": 10,
        "second_ema_length": 14,
        "third_ema_length": 42,
        "max_open_hours": 4,
    }

    # Sell hyperspace params:
    sell_params = {}

    # ROI table:
    minimal_roi = {"0": 0.567, "173": 0.218, "403": 0.059, "1295": 0}

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False
    order_types = {
        "buy": "limit",
        "sell": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    plot_config = {
        "main_plot": {
            "first_ema": {"color": "red"},
            "second_ema": {"color": "orange"},
            "third_ema": {"color": "blue"},
        },
        "subplots": {
            "stochastic": {
                "fastk": {"color": "red"},
                "slowd": {"color": "blue"},
            }
        },
    }

    use_custom_stoploss = True
    stoploss = -0.9

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        # Manage losing trades and open room for better ones.
        if (current_profit < 0) & (
            current_time - timedelta(hours=int(self.max_open_hours.value))
            > trade.open_date_utc
        ):
            return 0.01
        return 0.99

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["has_volume"] = dataframe["volume"] > 0

        dataframe["first_ema"] = ta.EMA(dataframe["close"], self.first_ema_length.value)
        dataframe["second_ema"] = ta.EMA(
            dataframe["close"], self.second_ema_length.value
        )
        dataframe["third_ema"] = ta.EMA(dataframe["close"], self.third_ema_length.value)

        fastk, slowd = ta.STOCH(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
        )
        dataframe["fastk"] = fastk
        dataframe["slowd"] = slowd

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = (
            dataframe["has_volume"]
            & (dataframe["first_ema"] > dataframe["second_ema"])
            & (dataframe["second_ema"] > dataframe["third_ema"])
        )
        trigger = qtpylib.crossed_above(dataframe["fastk"], dataframe["slowd"])

        dataframe.loc[
            conditions & trigger,
            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe["fastk"], dataframe["slowd"]), "sell"
        ] = 0
        return dataframe
