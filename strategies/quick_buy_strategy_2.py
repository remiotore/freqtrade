


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class quick_buy_strategy_2(IStrategy):


    INTERFACE_VERSION = 2


    minimal_roi = {
        "40": 0.0,
        "30": 0.01,
        "20": 0.02,
        "0": 0.04
    }


    stoploss = -0.1

    trailing_stop = False




    timeframe = '1hr'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

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

        dataframe['rsi'] = ta.RSI(dataframe)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_midband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

              (dataframe['close'] > dataframe['bb_midband'])
            ),
            'buy'] = 1

        print(dataframe)
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
                (dataframe['close'] > dataframe['bb_midband']) # Signal: price is greater than mid bb
            ),
            'sell'] = 1

        return dataframe