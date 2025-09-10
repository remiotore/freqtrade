import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from functools import reduce
import pandas_ta as pdta


class Test4(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "buy_sig_dip_1": 0.043,
    }

    # Sell hyperspace params:
    sell_params = {
    }

    # ROI table:
    minimal_roi = {
        "0": 0.084,
        "37": 0.056,
        "53": 0.022,
        "122": 0
    }

    # Stoploss:
    stoploss = -0.344

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.268
    trailing_stop_positive_offset = 0.348
    trailing_only_offset_is_reached = True

    # Custom stoploss
    use_custom_stoploss = False

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 210

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Buy Hyperopt
    buy_sig_dip_1 = DecimalParameter(0.01, 0.1, default=0.03, space="buy", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["ema_9"] = pdta.ema(dataframe["close"], length=9)
        dataframe["ema_21"] = pdta.ema(dataframe["close"], length=21)
        dataframe["ema_200"] = pdta.ema(dataframe["close"], length=200)

        dataframe["SIG_dip_1"] = (dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                    (dataframe["SIG_dip_1"] > self.buy_sig_dip_1.value) &
                    (dataframe["volume"] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x: x, conditions),
                "buy"
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                    # IDEJA: Ne cross između ema 9 i ema 21, nego neka najveća delta...
                    (qtpylib.crossed_below(dataframe["ema_9"], dataframe["ema_21"])) &
                    (dataframe["close"] > dataframe["ema_200"]) &
                    (dataframe["volume"] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x: x, conditions),
                "sell"
            ] = 1

        return dataframe