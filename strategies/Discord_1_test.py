# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
# --------------------------------


class test(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 1,
    }

    stoploss = -0.05
    # --- Configuration ---
    timeframe = "1d"
    use_sell_signal = True
    sell_profit_only = False
    process_only_new_candles = True
    startup_candle_count = 10
    use_custom_stoploss = False
    # --- Configuration ---

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema10']))
            ),
            ['buy', 'buy_tag']] = (1, 'Golden Cross')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["rsi"], 70))
            ),
            'sell'] = 1

        return dataframe
