
from freqtrade.strategy.interface import IStrategy
from functools import reduce
from pandas import DataFrame

import numpy as np
import pandas as pd  # noqa
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import technical.indicators as ftt
from freqtrade.strategy.parameters import DecimalParameter, IntParameter, CategoricalParameter


class ichiV1_n1(IStrategy):
    buy_trend_above_senkou_level = IntParameter(1, 8, default=1, space='buy', optimize=False)
    buy_trend_bullish_level = IntParameter(1, 8, default=6, space='buy', optimize=True)
    buy_fan_magnitude_shift_value = IntParameter(1, 10, default=3, space='buy', optimize=True)
    buy_min_fan_magnitude_gain = DecimalParameter(1.000, 1.010, default=1.008, space='buy', optimize=True)

    sell_trend_indicator = CategoricalParameter(
        [
            'trend_close_5m',
            'trend_close_15m',
            'trend_close_30m',
            'trend_close_1h',
            'trend_close_2h',
            'trend_close_4h',
            'trend_close_6h',
            'trend_close_8h',
        ],
        default='trend_close_2h',
        space='sell',
        optimize=True,
    )

    buy_params = {
      "buy_trend_above_senkou_level": 1,
      "buy_fan_magnitude_shift_value": 1,
      "buy_min_fan_magnitude_gain": 1.01,
      "buy_trend_bullish_level": 1
    }

    sell_params = {
      "sell_trend_indicator": "trend_close_6h"
    }

    minimal_roi = {
      "0": 0.057,
      "11": 0.025,
      "21": 0.01,
      "118": 0
    }

    stoploss = -0.275

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
    }

    timeframe = '5m'

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = False




    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    plot_config = {
        'main_plot': {

            'senkou_a': {
                'color': 'green', #optional
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud', #optional
                'fill_color': 'rgba(255,76,46,0.2)', #optional
            },

            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'}
        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']

        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=96)

        dataframe['trend_open_5m'] = dataframe['open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=96)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['chikou_span'] = ichimoku['chikou_span']
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        dataframe['atr'] = ta.ATR(dataframe)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        if self.buy_trend_above_senkou_level.value >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_b'])

        if self.buy_trend_bullish_level.value >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])

        if self.buy_trend_bullish_level.value >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])

        if self.buy_trend_bullish_level.value >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])

        if self.buy_trend_bullish_level.value >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['trend_open_1h'])

        if self.buy_trend_bullish_level.value >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['trend_open_2h'])

        if self.buy_trend_bullish_level.value >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])

        if self.buy_trend_bullish_level.value >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['trend_open_6h'])

        if self.buy_trend_bullish_level.value >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['trend_open_8h'])

        conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_min_fan_magnitude_gain.value)
        conditions.append(dataframe['fan_magnitude'] > 1)

        for x in self.buy_fan_magnitude_shift_value.range:
            conditions.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_trend_indicator.value]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
