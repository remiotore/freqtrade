import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

from technical.indicators import PMAX, TKE

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class PMAXStrategy(IStrategy):

    PM_PERIOD = 10
    PM_MULTIPLIER = 3
    PM_LENGTH = 10
    PM_MAtype = 5

    TKE_LENGTH = 14
    TKE_EMA_PERIOD = 5

    PM_MA_COL = 'MA_' + str(PM_MAtype) + '_' + str(PM_LENGTH)
    PM_ATR_COL = 'ATR_' + str(PM_PERIOD)
    PM_COL = 'pm_' + str(PM_PERIOD) + '_' + str(PM_MULTIPLIER) + \
        '_' + str(PM_LENGTH) + '_' + str(PM_MAtype)
    PMX_COL = 'pmX_' + str(PM_PERIOD) + '_' + str(PM_MULTIPLIER) + \
        '_' + str(PM_LENGTH) + '_' + str(PM_MAtype)

    INTERFACE_VERSION = 2

    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }

    minimal_roi = {
        "0": 1000
    }

    stoploss = -0.3

    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.0  # Disabled / not configured

    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = PM_PERIOD

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Stoch
        stoch = ta.STOCH(dataframe, 14)
        dataframe['slowk'] = stoch['slowk']

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]
             ) / dataframe["bb_middleband"]
        )

        # PMax
        pmaxdf = PMAX(dataframe, period=self.PM_PERIOD, multiplier=self.PM_MULTIPLIER,
                      length=self.PM_LENGTH, MAtype=self.PM_MAtype)
        dataframe[self.PM_COL] = pmaxdf[self.PM_COL]
        dataframe[self.PMX_COL] = pmaxdf[self.PMX_COL]
        dataframe[self.PM_ATR_COL] = pmaxdf[self.PM_ATR_COL]
        dataframe[self.PM_MA_COL] = pmaxdf[self.PM_MA_COL]

        # TKE
        dataframe['TKE'], dataframe['TKEema'] = TKE(
            dataframe, length=self.TKE_LENGTH, emaperiod=self.TKE_EMA_PERIOD)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # qtpylib.crossed_above(
                #     dataframe[self.PM_MA_COL], dataframe[self.PM_COL]) &
                (dataframe[self.PM_MA_COL] > dataframe[self.PM_COL]) &
                (dataframe[self.PMX_COL] == "up") &
                (dataframe['slowk'] < 20) &
                (dataframe['bb_width'] > 0.08) &
                # (dataframe['TKE'] > 20) &
                qtpylib.crossed_above(dataframe['TKE'], 20) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                qtpylib.crossed_below(
                    dataframe[self.PM_MA_COL], dataframe[self.PM_COL]) &
                (dataframe[self.PMX_COL] == "down") &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
