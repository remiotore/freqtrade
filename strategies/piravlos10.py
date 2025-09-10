"""
Strategy Name: My_Custom_Strategy
Author: YourName
Date Created: 2025-01-09
Description: Optimized strategy using the best Hyperopt results.
"""

import numpy as np
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
import talib.abstract as ta
from functools import reduce

class My_Custom_Strategy(IStrategy):
    # Strategy Parameters
    minimal_roi = {"0": 1}
    stoploss = -0.25
    timeframe = "5m"

    buy_params = {
        "buy_cti_32": -0.8,
        "buy_rsi_32": 25,
        "buy_rsi_fast_32": 55,
        "buy_sma15_32": 0.979,
    }

    sell_params = {
        "sell_fastx": 56,
        "sell_loss_cci": 134,
        "sell_loss_cci_profit": 0.0,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['sma15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = ta.CTI(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe['cti'] < self.buy_params['buy_cti_32'],
            dataframe['rsi'] < self.buy_params['buy_rsi_32'],
            dataframe['rsi'] > self.buy_params['buy_rsi_fast_32'],
            dataframe['sma15'] < self.buy_params['buy_sma15_32'],
        ]
        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe['rsi'] > self.sell_params['sell_fastx'],
            dataframe['cti'] > self.sell_params['sell_loss_cci'],
        ]
        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
        return dataframe
