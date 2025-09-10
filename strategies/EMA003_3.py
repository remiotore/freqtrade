
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

class EMA003_3(IStrategy):
    """
    Combo of Strategy 003 and EMA Cross

    How to use it?
    > python3 ./freqtrade/main.py -s EMA003
    """
    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

    minimal_roi = {
        "0": 0.278,
        "39": 0.087,
        "124": 0.038,
        "135": 0
    }

    trailing_stop = True
    trailing_stop_positive = 0.172
    trailing_stop_positive_offset = 0.212
    trailing_only_offset_is_reached = False

    stoploss = -0.333

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
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

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']

        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)
        dataframe['ema25'] = ta.EMA(dataframe, timeperiod=25)

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
                    (dataframe['ema7'] > dataframe['ema25']) &
                    (dataframe['ema7'].shift(1) <= dataframe['ema25'].shift(1))
            ),
            'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0
            return dataframe

        dataframe.loc[
            (
                (dataframe['sar'] > dataframe['close']) &
                (dataframe['fisher_rsi'] > 0.3)
            ),
            'sell'] = 1
        return dataframe