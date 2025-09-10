

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


_trend_length = 14

class STRATEGY_RSI_BB_CROSS(IStrategy):
    """
    Strategy RSI_BB_CROSS
    author@: Fractate_Dev
    github@: https://github.com/Fractate/freqbot
    How to use it?
    > python3 ./freqtrade/main.py -s RSI_BB_CROSS
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
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {





            "BB": {
                'bb_percent': {'color': 'red'},
                '1': {},
                '0': {},
            },





            "RSI_Percent": {
                'rsi_percent': {'color': 'red'},
                '1': {},
                '0': {},
            },






            "bb_minus_rsi_percent": {
                'bb_minus_rsi_percent': {},
                '0': {},
            },
            "bb_rsi_count": {
                'bb_above_rsi_count': {},
                'bb_below_rsi_count': {},
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

        for i in range(1):
            print("")




        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        dataframe['bb_percent'] = (dataframe['close'] - bollinger['lower']) / (bollinger['upper'] - bollinger['lower'])

        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['70'] = 70
        dataframe['30'] = 30
        dataframe['1'] = 1
        dataframe['0'] = 0

        rsi_limit = 30
        dataframe['rsi_percent'] = (dataframe['rsi'] - rsi_limit) / (100 - rsi_limit * 2)

        dataframe['bb_minus_rsi_percent'] = dataframe['bb_percent'] - dataframe['rsi_percent']

        dataframe['bb_above_rsi_count'] = True
        dataframe['bb_below_rsi_count'] = True
        for i in range(_trend_length):
            
            dataframe['bb_above_rsi_count'] = (dataframe['bb_minus_rsi_percent'].shift(i) > 0) & dataframe['bb_above_rsi_count']

            dataframe['bb_below_rsi_count'] = (dataframe['bb_minus_rsi_percent'].shift(i) < 0) & dataframe['bb_below_rsi_count']

















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




                qtpylib.crossed_above(dataframe['bb_percent'], dataframe['rsi_percent']) &
                (dataframe['bb_percent'] < 0.5) & 
                (dataframe['rsi_percent'] < 0.5) &
                (dataframe['bb_below_rsi_count'].shift(1))
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




                qtpylib.crossed_below(dataframe['bb_percent'], dataframe['rsi_percent']) &
                (dataframe['bb_percent'] > 0.5) & 
                (dataframe['rsi_percent'] > 0.5) &
                (dataframe['bb_above_rsi_count'].shift(1))
            ),
            'sell'] = 1
        return dataframe
    