from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
pd.options.mode.chained_assignment = None
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair
import numpy as np
from freqtrade.strategy import stoploss_from_open

class LSV1(IStrategy):
    can_short = True

    enter_long_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.002
    }

    exit_long_params = {
        "exit_long_indicator": "trend_close_2h",
    }

    enter_short_params = {
        "short_trend_below_senkou_level": 1,
        "short_trend_bearish_level": 6,
        "short_fan_magnitude_shift_value": 3,
        "short_min_fan_magnitude_gain": 0.998
    }

    exit_short_params = {
        "exit_short_indicator": "trend_close_2h",
    }

    minimal_roi = {
        "0": 0.14,
        "20": 0.08,
        "50": 0.04,
        "90": 0.015,
        "140": 0
    }

    stoploss = -0.275

    timeframe = '15m'

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    plot_config = {
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(255,76,46,0.2)',
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

        dataframe['fan_magnitude'] = (dataframe['trend_close_2h'] / dataframe['trend_close_8h'])
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

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_long = []
        conditions_short = []

        if self.enter_long_params['buy_trend_above_senkou_level'] >= 1:
            conditions_long.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions_long.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])

        if self.enter_long_params['buy_trend_above_senkou_level'] >= 2:
            conditions_long.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            conditions_long.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])

        if self.enter_long_params['buy_trend_above_senkou_level'] >= 3:
            conditions_long.append(dataframe['trend_close_30m'] > dataframe['senkou_a'])
            conditions_long.append(dataframe['trend_close_30m'] > dataframe['senkou_b'])

        if self.enter_long_params['buy_trend_above_senkou_level'] >= 4:
            conditions_long.append(dataframe['trend_close_1h'] > dataframe['senkou_a'])
            conditions_long.append(dataframe['trend_close_1h'] > dataframe['senkou_b'])

        if self.enter_long_params['buy_trend_above_senkou_level'] >= 5:
            conditions_long.append(dataframe['trend_close_2h'] > dataframe['senkou_a'])
            conditions_long.append(dataframe['trend_close_2h'] > dataframe['senkou_b'])

        if self.enter_long_params['buy_trend_above_senkou_level'] >= 6:
            conditions_long.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            conditions_long.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])

        if self.enter_long_params['buy_trend_above_senkou_level'] >= 7:
            conditions_long.append(dataframe['trend_close_6h'] > dataframe['senkou_a'])
            conditions_long.append(dataframe['trend_close_6h'] > dataframe['senkou_b'])

        if self.enter_long_params['buy_trend_above_senkou_level'] >= 8:
            conditions_long.append(dataframe['trend_close_8h'] > dataframe['senkou_a'])
            conditions_long.append(dataframe['trend_close_8h'] > dataframe['senkou_b'])

        if self.enter_long_params['buy_trend_bullish_level'] >= 1:
            conditions_long.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])

        if self.enter_long_params['buy_trend_bullish_level'] >= 2:
            conditions_long.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])

        if self.enter_long_params['buy_trend_bullish_level'] >= 3:
            conditions_long.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])

        if self.enter_long_params['buy_trend_bullish_level'] >= 4:
            conditions_long.append(dataframe['trend_close_1h'] > dataframe['trend_open_1h'])

        if self.enter_long_params['buy_trend_bullish_level'] >= 5:
            conditions_long.append(dataframe['trend_close_2h'] > dataframe['trend_open_2h'])

        if self.enter_long_params['buy_trend_bullish_level'] >= 6:
            conditions_long.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])

        if self.enter_long_params['buy_trend_bullish_level'] >= 7:
            conditions_long.append(dataframe['trend_close_6h'] > dataframe['trend_open_6h'])

        if self.enter_long_params['buy_trend_bullish_level'] >= 8:
            conditions_long.append(dataframe['trend_close_8h'] > dataframe['trend_open_8h'])

        conditions_long.append(dataframe['fan_magnitude'] > 1)
        for x in range(self.enter_long_params['buy_fan_magnitude_shift_value']):
            conditions_long.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])

        if self.enter_short_params['short_trend_below_senkou_level'] >= 1:
            conditions_short.append(dataframe['trend_close_5m'] < dataframe['senkou_a'])
            conditions_short.append(dataframe['trend_close_5m'] < dataframe['senkou_b'])

        if self.enter_short_params['short_trend_below_senkou_level'] >= 2:
            conditions_short.append(dataframe['trend_close_15m'] < dataframe['senkou_a'])
            conditions_short.append(dataframe['trend_close_15m'] < dataframe['senkou_b'])

        if self.enter_short_params['short_trend_below_senkou_level'] >= 3:
            conditions_short.append(dataframe['trend_close_30m'] < dataframe['senkou_a'])
            conditions_short.append(dataframe['trend_close_30m'] < dataframe['senkou_b'])

        if self.enter_short_params['short_trend_below_senkou_level'] >= 4:
            conditions_short.append(dataframe['trend_close_1h'] < dataframe['senkou_a'])
            conditions_short.append(dataframe['trend_close_1h'] < dataframe['senkou_b'])

        if self.enter_short_params['short_trend_below_senkou_level'] >= 5:
            conditions_short.append(dataframe['trend_close_2h'] < dataframe['senkou_a'])
            conditions_short.append(dataframe['trend_close_2h'] < dataframe['senkou_b'])

        if self.enter_short_params['short_trend_below_senkou_level'] >= 6:
            conditions_short.append(dataframe['trend_close_4h'] < dataframe['senkou_a'])
            conditions_short.append(dataframe['trend_close_4h'] < dataframe['senkou_b'])

        if self.enter_short_params['short_trend_below_senkou_level'] >= 7:
            conditions_short.append(dataframe['trend_close_6h'] < dataframe['senkou_a'])
            conditions_short.append(dataframe['trend_close_6h'] < dataframe['senkou_b'])

        if self.enter_short_params['short_trend_below_senkou_level'] >= 8:
            conditions_short.append(dataframe['trend_close_8h'] < dataframe['senkou_a'])
            conditions_short.append(dataframe['trend_close_8h'] < dataframe['senkou_b'])

        if self.enter_short_params['short_trend_bearish_level'] >= 1:
            conditions_short.append(dataframe['trend_close_5m'] < dataframe['trend_open_5m'])

        if self.enter_short_params['short_trend_bearish_level'] >= 2:
            conditions_short.append(dataframe['trend_close_15m'] < dataframe['trend_open_15m'])

        if self.enter_short_params['short_trend_bearish_level'] >= 3:
            conditions_short.append(dataframe['trend_close_30m'] < dataframe['trend_open_30m'])

        if self.enter_short_params['short_trend_bearish_level'] >= 4:
            conditions_short.append(dataframe['trend_close_1h'] < dataframe['trend_open_1h'])

        if self.enter_short_params['short_trend_bearish_level'] >= 5:
            conditions_short.append(dataframe['trend_close_2h'] < dataframe['trend_open_2h'])

        if self.enter_short_params['short_trend_bearish_level'] >= 6:
            conditions_short.append(dataframe['trend_close_4h'] < dataframe['trend_open_4h'])

        if self.enter_short_params['short_trend_bearish_level'] >= 7:
            conditions_short.append(dataframe['trend_close_6h'] < dataframe['trend_open_6h'])

        if self.enter_short_params['short_trend_bearish_level'] >= 8:
            conditions_short.append(dataframe['trend_close_8h'] < dataframe['trend_open_8h'])

        conditions_short.append(dataframe['fan_magnitude'] < 1)
        for x in range(self.enter_short_params['short_fan_magnitude_shift_value']):
            conditions_short.append(dataframe['fan_magnitude'].shift(x+1) > dataframe['fan_magnitude'])









        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                'enter_long'] = 1

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_short),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_long = []
        conditions_short = []

        conditions_long.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.exit_long_params['exit_long_indicator']]))

        conditions_short.append(qtpylib.crossed_above(dataframe['trend_close_5m'], dataframe[self.exit_short_params['exit_short_indicator']]))

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                'exit_long'] = 1

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_short),
                'exit_short'] = 1

        return dataframe







