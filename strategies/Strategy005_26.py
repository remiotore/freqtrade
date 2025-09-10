"""
3. sma ema with complicated support
4. sma ema with simple support
4-0.2. sma ema with simple support with 0.2 SL
5. sma wma with simple support
6. sma wma with VWAP simple support
"""

from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


from finta import TA as F
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np # noqa


class Strategy005_26(IStrategy):
    minimal_roi = {
        "0": 0.5,
        "30": 0.3,
        "60": 0.125,
        "120": 0.06,
        "180": 0.01
    }


    stoploss = -0.5

    ticker_interval = '15m'

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

    startup_candle_count = 34

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["SLOWMA"] = F.EMA(dataframe, 13)
        dataframe["FASTMA"] = F.EMA(dataframe, 34)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[qtpylib.crossed_above(dataframe['FASTMA'], dataframe['SLOWMA']) ,'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[qtpylib.crossed_below(dataframe['FASTMA'], dataframe['SLOWMA']) ,'sell'] = 1
        
        return dataframe