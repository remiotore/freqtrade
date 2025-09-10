# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Monday2Friday(IStrategy):
  

    INTERFACE_VERSION = 2

    minimal_roi = {
      "0": 0.416,
      "472": 0.16699999999999998,
      "885": 0.091,
      "1860": 0.001
    }

    stoploss = -0.273

    trailing_stop = True,
    trailing_stop_positive = 0.232,
    trailing_stop_positive_offset = 0.266,
    trailing_only_offset_is_reached = True

    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 14

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {},
        'subplots': {}
    }


    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['day'] = dataframe['date'].dt.day_name()
        dataframe['hour'] = dataframe['date'].dt.hour
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['day'] == 'Monday') |
                    (dataframe['day'] == 'Tuesday')
                ) &
                (dataframe['macd'] < 0) &
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['day'] == 'Friday') &
                (dataframe['macd'] > 0) &
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))
            ),
            'sell'] = 1
        return dataframe
