
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair
import numpy as np
from freqtrade.strategy import stoploss_from_open


class ichiV1_4(IStrategy):


    buy_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.002 # NOTE: Good value (Win% ~70%), alot of trades

    }


    sell_params = {
        "sell_trend_indicator": "trend_close_2h",
    }

    minimal_roi = {
        "0": 0.059,
        "10": 0.037,
        "41": 0.012,
        "114": 0
    }

    stoploss = -0.05

    timeframe = '1m'

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = False




    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

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
            'trend_open_5m': {'color': '#33FF7D'},



            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_open_4h': {'color': '#61FF33'},


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

        dataframe['trend_close_1m'] = dataframe['close']
        dataframe['trend_close_5m'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=15)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=30)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=60)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=120)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=240)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=360)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=480)

        dataframe['trend_open_1m'] = dataframe['open']
        dataframe['trend_open_5m'] = ta.EMA(dataframe['open'], timeperiod=5)
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=15)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=30)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=60)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=120)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=240)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=360)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=480)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=100, base_line_periods=300, laggin_span=600, displacement=150)
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

        if self.buy_params['buy_trend_above_senkou_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_bullish_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])

        if self.buy_params['buy_trend_bullish_level'] >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])

        if self.buy_params['buy_trend_bullish_level'] >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])

        if self.buy_params['buy_trend_bullish_level'] >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['trend_open_1h'])

        if self.buy_params['buy_trend_bullish_level'] >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['trend_open_2h'])

        if self.buy_params['buy_trend_bullish_level'] >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])

        if self.buy_params['buy_trend_bullish_level'] >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['trend_open_6h'])

        if self.buy_params['buy_trend_bullish_level'] >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['trend_open_8h'])

        conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_params['buy_min_fan_magnitude_gain'])
        conditions.append(dataframe['fan_magnitude'] > 1)

        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            conditions.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
