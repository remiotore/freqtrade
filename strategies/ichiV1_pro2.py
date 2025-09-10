
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

class ichiV1_pro2(IStrategy):


    buy_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.002
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

    stoploss = -0.275

    timeframe = '5m'

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

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

        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe, window=50)

        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_9'] = ta.RSI(dataframe, timeperiod=9)

        dataframe['ewo'] = self.calculate_ewo(dataframe, param1, param2)  # Sostituisci con i parametri appropriati


        return dataframe

    def calculate_ewo(self, dataframe: DataFrame, param1, param2):


        ewo = ...
        return ewo

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(dataframe['rsi_9'] < 35)
        conditions.append(dataframe['close'] < dataframe['ema_9'] * offset_basso)
        conditions.append(dataframe['ewo'] > soglia_alta)
        conditions.append(dataframe['rsi_14'] < soglia_acquisto)
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['close'] < dataframe['ema_50'] * offset_elevato)

        conditions.append(dataframe['rsi_9'] < 25)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []


        conditions.append(dataframe['close'] > dataframe['sma_9'])
        conditions.append(dataframe['close'] > dataframe['ema_50'] * offset_elevato)
        conditions.append(dataframe['rsi_14'] > 50)
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['rsi_9'] > dataframe['rsi_14'])

        conditions.append(dataframe['close'] < dataframe['hma_50'])
        conditions.append(dataframe['close'] > dataframe['ema_50'] * offset_elevato)
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['rsi_9'] > dataframe['rsi_14'])

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe