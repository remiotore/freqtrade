


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class bbrsi_optimized_strategy_679(IStrategy):


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 0.323,
        "107": 0.097,
        "150": 0.019,
        "238": 0
    }


    stoploss = -0.344

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
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
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
        dataframe['bb_lowerband_1sd'] = bollinger_1sd['lower']

        bollinger_2sd = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband_2sd'] = bollinger_2sd['lower']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 12) & 
                (dataframe['close'] < dataframe['bb_lowerband_2sd'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 96) & 
                (dataframe['close'] > dataframe['bb_lowerband_1sd'])
            ),
            'sell'] = 1
        return dataframe
