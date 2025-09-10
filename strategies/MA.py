"""
3. sma ema with complicated support
4. sma ema with simple support
4-0.2. sma ema with simple support with 0.2 SL
5. sma wma with simple support
6. sma wma with VWAP simple support
"""

from datetime import datetime
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import numpy as np # noqa


class MA(IStrategy):








    minimal_roi = {
        "0": 0.1,
        "83": 0.05,
        "142": 0.02,
        "161": 0
    }


    stoploss = -0.02

    ticker_interval = '15m'
    timeframe = '15min'

    trailing_stop = False

    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    startup_candle_count = 55

    def informative_pairs(self):
        return []

    def populate_indicators(dataframe: DataFrame, metadata=None) -> DataFrame:



        dataframe["SLOWMA"] = ta.EMA(dataframe, 6, )
        dataframe["FASTMA"] = ta.TEMA(dataframe, 6, )
        dataframe["SupportMA"] = ta.SMA(dataframe, 50, )

        return dataframe

    def populate_buy_trend(dataframe: DataFrame, metadata=None) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['FASTMA'], dataframe['SLOWMA'])) &
            (dataframe['close'].astype(float) >= (dataframe['SupportMA'] * 0.95))
            ,'buy'] = 1

        return dataframe

    def populate_sell_trend(dataframe: DataFrame, metadata=None) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['FASTMA'], dataframe['SLOWMA'])) &
            (dataframe['close'].astype(float) <= (dataframe['SupportMA'] * 0.95))
            ,'sell'] = 1
        return dataframe