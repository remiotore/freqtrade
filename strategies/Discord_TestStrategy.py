# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TestStrategy(IStrategy):
    """
    Strategy 001_custom_sell
    author@: Gerald Lonlas, froggleston
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy001_custom_sell
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        #"60":  0.01,
        #"30":  0.03,
        #"20":  0.04,
        #"0":  0.05
        "0": 10
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    #stoploss = -0.10
    stoploss = -0.99

    # Optimal timeframe for the strategy
    timeframe = '1m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
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
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        for i in [65,177,370,558,639,677,684,699,741,905,924,1023,1176,1200,1269,1320]:
            dataframe.iloc[i, dataframe.columns.get_loc('buy')] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        for i in [67,179,372,560,641,679,686,701,743,907,926,1025,1178,1202,1271,1322]:
            dataframe.iloc[i, dataframe.columns.get_loc('sell')] = 1

        return dataframe