

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

_total_length = 500

class STRATEGY_SMOOTHED(IStrategy):
    """
    Strategy SMOOTHED
    author@: Fractate_Dev
    github@: https://github.com/Fractate/freqbot
    How to use it?
    > python3 ./freqtrade/main.py -s SMOOTHED
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }


    stoploss = -0.10

    trailing_stop = False

    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 20

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
            'ema10': {},
            'ema10smoothed': {},
            'ema_smoothed_goingup': {},
            'wt_goingup':{},
        },
        'subplots': {

            "WaveTrend": {
                'wt1': {'color': 'blue'},
                'wt2': {'color': 'orange'},
                '0' : {},
                '-70' : {},
                '-80' : {},
                '70' : {},
                '80' : {},
            },
        }
    }
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=_total_length)

        dataframe['ema10smoothed'] = 0
        for i in range(_total_length):
            dataframe['ema10smoothed'] += dataframe['ema10'].shift(i)/_total_length









        dataframe['ap'] = dataframe["close"] + dataframe["high"] + dataframe["low"]
        dataframe['esa'] = ta.EMA(dataframe['ap'], _total_length)

        d = ta.EMA(abs(dataframe['ap'] - dataframe['esa']), _total_length * 2)
        dataframe['ci'] = (dataframe['ap'] - dataframe['esa']) / (0.015 * d)

        dataframe['tci'] = ta.EMA(dataframe['ci'], _total_length)

        dataframe['wt1'] = dataframe['tci']
        dataframe['wt2'] = ta.EMA(dataframe['wt1'], _total_length) #4)

        dataframe['0'] = 0
        dataframe['-70'] = -70
        dataframe['-80'] = -80
        dataframe['70'] = 70
        dataframe['80'] = 80

        dataframe['ema_smoothed_goingup'] = dataframe['ema10smoothed'] > dataframe['ema10smoothed'].shift(1)
        dataframe['wt_goingup'] = dataframe['wt2'] > dataframe['wt2'].shift(1)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (






                (dataframe['ema_smoothed_goingup'].shift(1) == False) &
                (dataframe['ema_smoothed_goingup'] == False) &
                (dataframe['wt_goingup'].shift(1) == False) &
                (dataframe['wt_goingup'] == True) &
                (dataframe['wt2'] < -75)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['ema_smoothed_goingup'].shift(1) == True) &
                (dataframe['ema_smoothed_goingup'] == True) &
                (dataframe['wt_goingup'].shift(1) == True) &
                (dataframe['wt_goingup'] == False) &
                (dataframe['wt2'] > 75)
            ),
            'sell'] = 1
        return dataframe
