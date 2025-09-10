

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class hlhb_423(IStrategy):
    """
    The HLHB ("Huck loves her bucks!") System simply aims to catch short-term forex trends.
    More information in https://www.babypips.com/trading/forex-hlhb-system-explained
    """

    INTERFACE_VERSION = 2

    position_stacking = "True"


    minimal_roi = {
        "0": 0.6225,
        "703": 0.2187,
        "2849": 0.0363,
        "5520": 0
    }


    stoploss = -0.3211

    trailing_stop = True
    trailing_stop_positive = 0.0117
    trailing_stop_positive_offset = 0.0186
    trailing_only_offset_is_reached = True

    timeframe = '4h'

    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {

        'main_plot': {
            'ema5': {},
            'ema10': {},
        },
        'subplots': {

            "RSI": {
                'rsi': {'color': 'red'},
            },
            "ADX": {
                'adx': {},
            }
        }
    }
    
    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['hl2'] = (dataframe["close"] + dataframe["open"]) / 2



        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10, price='hl2')

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)

        dataframe['adx'] = ta.ADX(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 50)) &
                (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema10'])) &
                (dataframe['adx'] > 25) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsi'], 50)) &
                (qtpylib.crossed_below(dataframe['ema5'], dataframe['ema10'])) &
                (dataframe['adx'] > 25) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    
