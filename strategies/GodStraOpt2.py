










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


class GodStraOpt2(IStrategy):








    buy_params = {
        "buy-indicator-0": "volatility_bbli",
        "buy-cross-0": "trend_stc",
        "buy-int-0": 44,
        "buy-real-0": -0.0345,
        "buy-oper-0": ">R",
    }

    sell_params = {
        "sell-indicator-0": "trend_adx",
        "sell-cross-0": "volume_vwap",
        "sell-int-0": 75,
        "sell-real-0": 0.65115,
        "sell-oper-0": "<R",
    }

    minimal_roi = {
        "0": 0.418,
        "5640": 0.276,
        "13649": 0.047,
        "18289": 0
    }

    stoploss = -0.229

    trailing_stop = True
    trailing_stop_positive = 0.023
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    timeframe = '12h'
    print('Add {\n\t"method": "AgeFilter",\n\t"min_days_listed": 30\n},\n to your pairlists in config (Under StaticPairList)')

    def dna_size(self, dct: dict):
        def int_from_str(st: str):
            str_int = ''.join([d for d in st if d.isdigit()])
            if str_int:
                return int(str_int)
            return -1  # in case if the parameter somehow doesn't have index
        return len({int_from_str(digit) for digit in dct.keys()})

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = dropna(dataframe)
        dataframe = add_all_ta_features(
            dataframe, open="open", high="high", low="low", close="close", volume="volume",
            fillna=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = list()

        for i in range(self.dna_size(self.buy_params)):

            OPR = self.buy_params[f'buy-oper-{i}']
            IND = self.buy_params[f'buy-indicator-{i}']
            CRS = self.buy_params[f'buy-cross-{i}']
            INT = self.buy_params[f'buy-int-{i}']
            REAL = self.buy_params[f'buy-real-{i}']
            DFIND = dataframe[IND]
            DFCRS = dataframe[CRS]

            if OPR == ">":
                conditions.append(DFIND > DFCRS)
            elif OPR == "=":
                conditions.append(np.isclose(DFIND, DFCRS))
            elif OPR == "<":
                conditions.append(DFIND < DFCRS)
            elif OPR == "CA":
                conditions.append(qtpylib.crossed_above(DFIND, DFCRS))
            elif OPR == "CB":
                conditions.append(qtpylib.crossed_below(DFIND, DFCRS))
            elif OPR == ">I":
                conditions.append(DFIND > INT)
            elif OPR == "=I":
                conditions.append(DFIND == INT)
            elif OPR == "<I":
                conditions.append(DFIND < INT)
            elif OPR == ">R":
                conditions.append(DFIND > REAL)
            elif OPR == "=R":
                conditions.append(np.isclose(DFIND, REAL))
            elif OPR == "<R":
                conditions.append(DFIND < REAL)

        print(conditions)
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = list()
        for i in range(self.dna_size(self.sell_params)):
            OPR = self.sell_params[f'sell-oper-{i}']
            IND = self.sell_params[f'sell-indicator-{i}']
            CRS = self.sell_params[f'sell-cross-{i}']
            INT = self.sell_params[f'sell-int-{i}']
            REAL = self.sell_params[f'sell-real-{i}']
            DFIND = dataframe[IND]
            DFCRS = dataframe[CRS]

            if OPR == ">":
                conditions.append(DFIND > DFCRS)
            elif OPR == "=":
                conditions.append(np.isclose(DFIND, DFCRS))
            elif OPR == "<":
                conditions.append(DFIND < DFCRS)
            elif OPR == "CA":
                conditions.append(qtpylib.crossed_above(DFIND, DFCRS))
            elif OPR == "CB":
                conditions.append(qtpylib.crossed_below(DFIND, DFCRS))
            elif OPR == ">I":
                conditions.append(DFIND > INT)
            elif OPR == "=I":
                conditions.append(DFIND == INT)
            elif OPR == "<I":
                conditions.append(DFIND < INT)
            elif OPR == ">R":
                conditions.append(DFIND > REAL)
            elif OPR == "=R":
                conditions.append(np.isclose(DFIND, REAL))
            elif OPR == "<R":
                conditions.append(DFIND < REAL)

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1

        return dataframe
