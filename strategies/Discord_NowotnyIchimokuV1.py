# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import math
from typing import Callable

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IntParameter
from pandas import Series
from numpy.typing import ArrayLike
from datetime import datetime, timedelta
import technical.indicators as indicators


class NowotnyIchimokuV1(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '1h'

    startup_candle_count = 180

    process_only_new_candles = False

    # stoploss = -0.04

    plot_config = {
        # # Main plot indicators (Moving averages, ...)
        'main_plot': {
            # 'lead_1': {
            #     'color': 'green',
            #     'fill_to': 'senkou_b',
            #     'fill_label': 'Ichimoku Cloud',
            #     'fill_color': 'rgba(0,0,0,0.2)',
            # },
            # # plot senkou_b, too. Not only the area to it.
            # 'lead_2': {
            #     'color': 'red',
            # },
            # 'conversion_line': {'color': 'blue'},
            # 'base_line': {'color': 'orange'},
            # 'lagging_conversion_line': {'color': 'purple'},
            # 'lagging_span': {'color': 'green'}
        },
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        ichi = indicators.ichimoku(df)

        df['conversion_line'] = ichi['tenkan_sen']
        df['base_line'] = ichi['kijun_sen']

        df['lead_1'] = ichi['senkou_span_a']
        df['lead_2'] = ichi['senkou_span_b']

        df['cloud_green'] = ichi['cloud_green']

        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conversion_line_above_cloud = (df['conversion_line'] > df['lead_1']) & (
                df['conversion_line'] > df['lead_2'])

        should_buy = (df['cloud_green'].shift(-25)) & (df['conversion_line'] > df['base_line']) & conversion_line_above_cloud & (df['close'].shift(25) > df['close'])

        df.loc[
            should_buy & ~(should_buy.shift())
            , 'buy'] = 1
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            qtpylib.crossed_below(df['close'].shift(25), df['close'])
            , 'sell'] = 1
        return df
