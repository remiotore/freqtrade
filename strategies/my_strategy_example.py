
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
import numpy as np


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)

    return rolling_mean, lower_band


class my_strategy_example(IStrategy):
    minimal_roi = {
      "0": 0.15437,
      "67": 0.10256,
      "106": 0.03905,
      "375": 0
    }

    stoploss = -0.34338
    timeframe = '15m'
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive_offset = 0.00375  # Trigger positive stoploss once crosses above this percentage
    trailing_stop_positive = 0.00175 # Sell asset if it dips down this much

    @property
    def protections(self):
        return [




            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=40, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_midband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['mid'] = np.nan_to_num(mid)
        dataframe['lower'] = np.nan_to_num(lower)
        dataframe['bbdelta'] = (dataframe['mid'] - dataframe['lower']).abs()
        dataframe['pricedelta'] = (dataframe['open'] - dataframe['close']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * 0.008) &
                dataframe['closedelta'].gt(dataframe['close'] * 0.0175) &
                dataframe['tail'].lt(dataframe['bbdelta'] * 0.25) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift())
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        no sell signal
        """
        dataframe.loc[:, 'sell'] = 0
        return dataframe
