
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class abydon_macd(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '1d'
    startup_candle_count: int = 25

    minimal_roi = {"0": 0.99}
    stoploss = -0.10
    trailing_stop = False
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = False


    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
        }
    }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']


        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['macd'] < 0) &
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
            ),
            'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['macd'] > 0) &
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))
            ),
            'sell'] = 1
        return dataframe