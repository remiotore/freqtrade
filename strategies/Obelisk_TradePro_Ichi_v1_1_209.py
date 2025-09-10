
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'

import technical.indicators as ftt
from technical.util import resample_to_interval, resampled_merge

from functools import reduce
from datetime import datetime, timedelta










class Obelisk_TradePro_Ichi_v1_1_209(IStrategy):

    timeframe = '1h'

    startup_candle_count = 120
    process_only_new_candles = True

    minimal_roi = {
        "0": 10,
    }

    stoploss = -0.015

    plot_config = {

        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(0,0,0,0.2)',
            },

            'senkou_b': {
                'color': 'red',
            },
            'tenkan_sen': { 'color': 'orange' },
            'kijun_sen': { 'color': 'blue' },

            'chikou_span': { 'color': 'lightgreen' },
        },
        'subplots': {
            "Signals": {
                'go_long': {'color': 'blue'},
                'future_green': {'color': 'green'},
                'chikou_high': {'color': 'lightgreen'},
            },
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:









        displacement = 30
        ichimoku = ftt.ichimoku(dataframe, 
            conversion_line_period=20, 
            base_line_periods=60,
            laggin_span=120, 
            displacement=displacement
            )

        dataframe['chikou_span'] = ichimoku['chikou_span']

        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1


        dataframe['future_green'] = (dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b']).astype('int') * 2



        dataframe['chikou_high'] = (
                (dataframe['chikou_span'] > dataframe['senkou_a']) &
                (dataframe['chikou_span'] > dataframe['senkou_b'])
            ).shift(displacement).fillna(0).astype('int')


        dataframe['go_long'] = (
                (dataframe['tenkan_sen'] > dataframe['kijun_sen']) &
                (dataframe['close'] > dataframe['senkou_a']) &
                (dataframe['close'] > dataframe['senkou_b']) &
                (dataframe['future_green'] > 0) &
                (dataframe['chikou_high'] > 0)
                ).astype('int') * 3


        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[

            qtpylib.crossed_above(dataframe['go_long'], 0),

        'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[

            qtpylib.crossed_below(dataframe['tenkan_sen'], dataframe['kijun_sen']) 
            | 
            qtpylib.crossed_below(dataframe['close'], dataframe['kijun_sen']),

        'sell'] = 1

        return dataframe
































































