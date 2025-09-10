



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class bbrsi_optimized_strategy_905(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """


    INTERFACE_VERSION = 2







    minimal_roi = {
        "0": 0.199,
        "115": 0.083,
        "219": 0.042,
        "497": 0
    }


    stoploss = -0.28

    trailing_stop = False




    buy_rsi = IntParameter(low=1, high=50, default=30,
                           space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70,
                            space='sell', optimize=True, load=True)

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
        dataframe['bb_upperband_1sd'] = bollinger_1sd['upper']

        bollinger_3sd = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband_3sd'] = bollinger_3sd['lower']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lowerband_3sd'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 72) &
                (dataframe['close'] > dataframe['bb_upperband_1sd'])
            ),
            'sell'] = 1

        return dataframe
