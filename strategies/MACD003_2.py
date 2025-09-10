
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class MACD003_2(IStrategy):
    """
    Strategy 003 sell + MACD buy signal

    How to use it?
    > python3 ./freqtrade/main.py -s MACD003
    """
    buy_mfi = DecimalParameter(10, 50, decimals=0, default=20, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.94, space="buy")

    buy_bb_gain = DecimalParameter(0, 0.20, decimals=1, default=0.04, space="buy")

    sell_mfi = DecimalParameter(1, 99, decimals=0, default=80, space="sell")
    sell_fisher = DecimalParameter(-1, 1, decimals=2, default=0.3, space="sell")

    minimal_roi = {
        "0": 0.171,
        "15": 0.08,
        "40": 0.011,
        "131": 0
    }

    stoploss = -0.332

    trailing_stop = True
    trailing_stop_positive = 0.153
    trailing_stop_positive_offset = 0.219
    trailing_only_offset_is_reached = True

    timeframe = '5m'

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

        dataframe['mfi'] = ta.MFI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (
                            (dataframe['mfi'] < self.buy_mfi.value) |

                            (dataframe['fisher_rsi'] < self.buy_fisher.value) |


                            (
                                    (dataframe['fastd'] > dataframe['fastk']) &
                                    (dataframe['fastk'] < 20)
                            )
                    ) &
                    (
                        (dataframe['macd'] < 0.0) &
                        (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
                        (dataframe['bb_gain'] >= self.buy_bb_gain.value)
                    )
            ),
            'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (dataframe['mfi'] > self.sell_mfi.value) &
                    (dataframe['sar'] > dataframe['close']) &
                    (dataframe['fisher_rsi'] > self.sell_fisher.value)
            ),
            'sell'] = 1
        return dataframe