

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class STRATEGY_005_FASTSTOCH_RSI(IStrategy):
    """
    Strategy 005
    author@: Gerald Lonlas
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy005
    """











    stoploss = -0.25

    ticker_interval = '5m'

    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

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

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']



        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[

            (
                (dataframe['close'] > 0.00000200) &
                (dataframe['volume'] > dataframe['volume'].rolling(200).mean() * 4) &
                (dataframe['close'] < dataframe['sma']) &
                (dataframe['fastd'] > dataframe['fastk']) &
                (dataframe['rsi'] > 0) &
                (dataframe['fastd'] > 0) &

                (dataframe['fisher_rsi_norma'] < 38.900000000000006)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[

            (
                (qtpylib.crossed_above(dataframe['rsi'], 50)) &
                (dataframe['macd'] < 0) &
                (dataframe['minus_di'] > 0)
            ) |
            (
                (dataframe['sar'] > dataframe['close']) &
                (dataframe['fisher_rsi'] > 0.3)
            ),

            'sell'] = 1
        return dataframe
