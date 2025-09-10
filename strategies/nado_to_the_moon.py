


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class nado_to_the_moon(IStrategy):


    INTERFACE_VERSION = 2


    minimal_roi = {
      "0": 0.15437,
      "67": 0.10256,
      "106": 0.03905,
      "375": 0
    }


    stoploss = -0.28227

    trailing_stop = False




    timeframe = '15m'

    process_only_new_candles = False

    startup_candle_count: int = 30

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'bb_upperband': {'color': 'green'},
            'bb_midband': {'color': 'orange'},
            'bb_lowerband': {'color': 'red'},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'yellow'},
            }
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

        dataframe['rsi'] = ta.RSI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_midband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['rsi'] < 28) &  # Signal: RSI is lower 38
                (dataframe['close'] < dataframe['bb_lowerband']) # Signal: price is less than lower bb 2sd
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['rsi'] > 75) &  # Signal: RSI is greater 88
                (dataframe['close'] > dataframe['bb_midband']) # Signal: price is greater than mid bb
            ),
            'sell'] = 1

        return dataframe