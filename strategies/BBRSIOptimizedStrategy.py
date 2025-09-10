


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class BBRSIOptimizedStrategy(IStrategy):


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 0.186,
        "37": 0.074,
        "89": 0.033,
        "195": 0
    }


    stoploss = -0.295

    trailing_stop = False




    timeframe = '5m'

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

        bollinger_1sd = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_midband_1sd'] = bollinger_1sd['mid']

        bollinger_3sd = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband_3sd'] = bollinger_3sd['lower']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['close'] < dataframe['bb_lowerband_3sd']) # Signal: price is less than lower bb 2sd
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 64) &  # Signal: RSI is greater 88
                    (dataframe['close'] > dataframe['bb_midband_1sd']) # Signal: price is greater than mid bb
            ),
            'sell'] = 1

        return dataframe
