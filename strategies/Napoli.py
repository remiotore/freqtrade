

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame

from freqtrade.strategy import (merge_informative_pair, CategoricalParameter,
                                DecimalParameter, IntParameter)

class Napoli(IStrategy):

    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.05
    }

    black_box_period = IntParameter(200, 400, default=200)
    atr_period = IntParameter(5, 30, default=14)
    atr_factor = DecimalParameter(0.1, 2.0, default=1.0)
    stoploss = -0.099

    trailing_stop = False




    timeframe = '15m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30




    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        p = self.black_box_period.value
        atr_p = self.atr_period.value
        atr_f = self.atr_factor.value

        h = dataframe['high'].copy()
        l = dataframe['low'].copy()
        highest_low = ta.MAX(h, timeperiod=p)
        lowest_high = ta.MIN(l, timeperiod=p)
        atr = ta.ATR(dataframe, timeperiod=atr_p)

        e = atr * atr_f
        src = dataframe['close'].shift(1)

        h[src > h.shift(1)] = highest_low + e
        l[src < l.shift(1)] = lowest_high - e

        dataframe['bl'] = (h + l) / 2


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        bottom_condition = (
            (dataframe['rsi'].shift(2) > dataframe['rsi'].shift(3)) &
            (dataframe['rsi'].shift(2) > dataframe['rsi'].shift(4)) &
            (dataframe['rsi'].shift(2) < 40) &
            (dataframe['rsi'].shift(2) > dataframe['rsi'].shift(1)) &
            (dataframe['bl'] > dataframe['bl'].shift(1)) &
            (dataframe['bl'].diff() > 0)
        )
        dataframe.loc[bottom_condition, 'buy'] = 1


        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """


        dataframe.loc[
        (
            (
                (dataframe['rsi'].shift(2) < dataframe['rsi'].shift(3)) &
                (dataframe['rsi'].shift(2) < dataframe['rsi'].shift(4)) &
                (dataframe['rsi'].shift(2) > 60) &
                (dataframe['rsi'].shift(2) > dataframe['rsi'].shift(1))
            )
            &
            (
                (dataframe['bl'] < dataframe['bl'].shift(1)) &
                (dataframe['bl'].diff() < 0)
            )
        ), 'sell'] = 1
        return dataframe
