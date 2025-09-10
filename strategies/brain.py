





import logging

from numpy.lib import math
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame



import pandas as pd

from ta import add_all_ta_features
from ta.utils import dropna
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import numpy as np


class brain(IStrategy):




    nodes = 4

    decimals = 2




    buy_params = {
        "buy-node-input-0": "trend_mass_index",
        "buy-node-enabled-0": 0,
        "buy-node-reversed-0": 1,
        "buy-node-wight-0": 81,
        "buy-node-input-1": "momentum_ao",
        "buy-node-enabled-1": 0,
        "buy-node-reversed-1": -1,
        "buy-node-wight-1": 36,
        "buy-node-input-2": "volatility_ui",
        "buy-node-enabled-2": 1,
        "buy-node-reversed-2": -1,
        "buy-node-wight-2": 59,
        "buy-node-input-3": "volatility_kcp",
        "buy-node-enabled-3": 1,
        "buy-node-reversed-3": 1,
        "buy-node-wight-3": 76,
    }

    sell_params = {
        "sell-node-input-0": "volatility_bbp",
        "sell-node-enabled-0": 0,
        "sell-node-reversed-0": -1,
        "sell-node-wight-0": 25,
        "sell-node-input-1": "trend_vortex_ind_diff",
        "sell-node-enabled-1": 0,
        "sell-node-reversed-1": 1,
        "sell-node-wight-1": 1,
        "sell-node-input-2": "trend_macd",
        "sell-node-enabled-2": 0,
        "sell-node-reversed-2": -1,
        "sell-node-wight-2": 4,
        "sell-node-input-3": "momentum_ppo_hist",
        "sell-node-enabled-3": 0,
        "sell-node-reversed-3": -1,
        "sell-node-wight-3": 72,
    }

    minimal_roi = {
        "0": 0.347,
        "392": 0.126,
        "727": 0.023,
        "1411": 0
    }

    stoploss = -0.256


    timeframe = '1h'

    decimals = 10 ** decimals

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe = add_all_ta_features(
            dataframe, open="open", high="high", low="low", close="close", volume="volume", fillna=False)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        RESULT = 0

        for i in range(self.nodes):
            DFINP = dataframe[self.buy_params[f'buy-node-input-{i}']]
            ENABLED = self.buy_params[f'buy-node-enabled-{i}']
            REVERSE = self.buy_params[f'buy-node-reversed-{i}']
            WIGHT = self.buy_params[f'buy-node-wight-{i}']/self.decimals
            RESULT += DFINP*ENABLED*REVERSE*WIGHT

        conditions.append(RESULT > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        RESULT = 0
        for i in range(self.nodes):
            DFINP = dataframe[self.sell_params[f'sell-node-input-{i}']]
            ENABLED = self.sell_params[f'sell-node-enabled-{i}']
            REVERSE = self.sell_params[f'sell-node-reversed-{i}']
            WIGHT = self.sell_params[f'sell-node-wight-{i}']/self.decimals
            RESULT += DFINP*ENABLED*REVERSE*WIGHT

        conditions.append(RESULT > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell']=1

        return dataframe
