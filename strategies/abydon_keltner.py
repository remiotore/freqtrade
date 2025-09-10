
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class abydon_keltner(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '5m'
    startup_candle_count: int = 30

    minimal_roi = {"0": 0.99}
    stoploss = -0.10
    trailing_stop = False
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = False


    plot_config = {
        'main_plot': {
            'SMA': {'color': 'blue'},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]
        dataframe["kc_percent"] = (
                (dataframe["close"] - dataframe["kc_lowerband"]) /
                (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        )
        dataframe["kc_width"] = (
                (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        )
        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (


            ),
            'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (


            ),
            'sell'] = 1
        return 