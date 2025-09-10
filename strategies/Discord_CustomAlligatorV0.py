# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
import time
from pandas import DataFrame

from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


def diff_lips_teeth_increase(dataframe, param):
    condition = True
    for i in range(0, param):
            condition = condition & ((dataframe['diff_lips_teeth'].shift(i) > dataframe['diff_lips_teeth'].shift(i+1)))
    return condition

def diff_lips_teeth_decrease(dataframe, param):
    condition = True
    for i in range(0, param):
            condition = condition & ((dataframe['diff_lips_teeth'].shift(i) < dataframe['diff_lips_teeth'].shift(i+1)))
    return condition


class CustomAlligatorV0(IStrategy):
    """
    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # # ROI table:
    # minimal_roi = {
    #     "0": 0.2,
    #     "720": 0.1,
    #     "1440": 0.05,
    #     "2160": 0.01,
    #     "2880": 0
    # }

    # # Stoploss:
    # stoploss = -0.15892


    # Stoploss:
    stoploss = -0.34269

    # # Trailing stop:
    # trailing_stop = True
    # trailing_stop_positive = 0.13776
    # trailing_stop_positive_offset = 0.23586
    # trailing_only_offset_is_reached = True

    # Trailing stop: (Sortino Optimized)
    # trailing_stop = True
    # trailing_stop_positive = 0.2088
    # trailing_stop_positive_offset = 0.2472
    # trailing_only_offset_is_reached = True


    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

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
        dataframe['lips'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['smma_lips'] = dataframe['lips'].rolling(3).mean()
        dataframe['teeth'] = ta.SMA(dataframe, timeperiod=8)
        dataframe['smma_teeth'] = dataframe['teeth'].rolling(5).mean()
        dataframe['jaw'] = ta.SMA(dataframe, timeperiod=13)
        dataframe['smma_jaw'] = dataframe['jaw'].rolling(8).mean()
        dataframe['diff_lips_teeth'] = (dataframe['smma_lips'] - dataframe['smma_teeth']).abs()
         # EMA 200
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # ADX
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=8)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=8)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['smma_lips'] < dataframe['smma_teeth']) &
                (dataframe['smma_lips'] < dataframe['smma_jaw'])  &
                (dataframe['smma_teeth'] > dataframe['smma_jaw']) & 
                (dataframe['close'].crossed_above(dataframe['smma_teeth'])) &
                (diff_lips_teeth_increase(dataframe, 1)) &
                (dataframe['plus_di'] > dataframe['minus_di']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            )
            ,'buy'] = 1

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
                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['smma_lips'] > dataframe['smma_teeth']) &
                (dataframe['smma_lips'] > dataframe['smma_jaw']) &
                (dataframe['open'].crossed_below(dataframe['smma_lips'])) &
                (diff_lips_teeth_decrease(dataframe, 1)) &
                (dataframe['plus_di'] >= dataframe['minus_di']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ) 
            , 'sell'] = 1
        return dataframe


