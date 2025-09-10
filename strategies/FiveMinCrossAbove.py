
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class FiveMinCrossAbove(IStrategy):
    """
    Strategy 005
    author@: Gerald Lonlas
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy005
    """


    minimal_roi = {
        "0": 0.015,
		"25": 0.01,
		"100": 0.005
    }


    stoploss = -0.99

    timeframe = '5m'






    process_only_new_candles = False

    use_sell_signal = False
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'market',
        'sell': 'market',
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




        dataframe['rsi8'] = ta.RSI(dataframe, timeperiod=8)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param metadata:
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[

            (
                    (qtpylib.crossed_above(dataframe['rsi8'], 30)) &
					(dataframe['rsi8'] < 41)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param metadata:
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[

            (
                    (dataframe['close'] > 9999999999)
            ),

            'sell'] = 1
        return dataframe
