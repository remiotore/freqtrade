import talib.abstract as ta
import numpy as np  # noqa
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter

__author__ = "Robert Roman"
__copyright__ = "Free For Use"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Robert Roman"
__email__ = "robertroman7@gmail.com"
__BTC_donation__ = "3FgFaG15yntZYSUzfEpxr5mDt1RArvcQrK"



class Bandtastic_2(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '15m'

    minimal_roi = {
        "0": 0.162,
        "69": 0.097,
        "229": 0.061,
        "566": 0
    }

    stoploss = -0.345

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.058
    trailing_only_offset_is_reached = False

    buy_fastema = IntParameter(low=1, high=236, default=211, space='buy', optimize=True, load=True)
    buy_slowema = IntParameter(low=1, high=126, default=364, space='buy', optimize=True, load=True)
    buy_rsi = IntParameter(low=15, high=70, default=52, space='buy', optimize=True, load=True)
    buy_mfi = IntParameter(low=15, high=70, default=30, space='buy', optimize=True, load=True)

    buy_rsi_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_mfi_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_ema_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_trigger = CategoricalParameter(["bb_lower1", "bb_lower2", "bb_lower3", "bb_lower4"], default="bb_lower1", space="buy")

    sell_fastema = IntParameter(low=1, high=365, default=7, space='sell', optimize=True, load=True)
    sell_slowema = IntParameter(low=1, high=365, default=6, space='sell', optimize=True, load=True)
    sell_rsi = IntParameter(low=30, high=100, default=57, space='sell', optimize=True, load=True)
    sell_mfi = IntParameter(low=30, high=100, default=46, space='sell', optimize=True, load=True)

    sell_rsi_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=False)
    sell_mfi_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=True)
    sell_ema_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=False)
    sell_trigger = CategoricalParameter(["sell-bb_upper1", "sell-bb_upper2", "sell-bb_upper3", "sell-bb_upper4"], default="sell-bb_upper2", space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)

        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['bb_upperband1'] = bollinger1['upper']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband4'] = bollinger4['lower']
        dataframe['bb_middleband4'] = bollinger4['mid']
        dataframe['bb_upperband4'] = bollinger4['upper']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.buy_rsi_enabled.value:
            conditions.append(dataframe['rsi'] < self.buy_rsi.value)
        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] < self.buy_mfi.value)
        if self.buy_ema_enabled.value:
            try:
                conditions.append(ta.EMA(dataframe, timeperiod=int(self.buy_fastema.value)) > ta.EMA(dataframe, timeperiod=int(self.buy_slowema.value)))
            except Exception:
                pass

        if self.buy_trigger.value == 'bb_lower1':
            conditions.append(dataframe["close"] < dataframe['bb_lowerband1'])
        if self.buy_trigger.value == 'bb_lower2':
            conditions.append(dataframe["close"] < dataframe['bb_lowerband2'])
        if self.buy_trigger.value == 'bb_lower3':
            conditions.append(dataframe["close"] < dataframe['bb_lowerband3'])
        if self.buy_trigger.value == 'bb_lower4':
            conditions.append(dataframe["close"] < dataframe['bb_lowerband4'])

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.sell_rsi_enabled.value:
            conditions.append(dataframe['rsi'] > self.sell_rsi.value)
        if self.sell_mfi_enabled.value:
            conditions.append(dataframe['mfi'] > self.sell_mfi.value)
        if self.sell_ema_enabled.value:
            try:
                conditions.append(ta.EMA(dataframe, timeperiod=int(self.sell_fastema.value)) < ta.EMA(dataframe, timeperiod=int(self.sell_slowema.value)))
            except Exception:
                pass

        if self.sell_trigger.value == 'sell-bb_upper1':
            conditions.append(dataframe["close"] > dataframe['bb_upperband1'])
        if self.sell_trigger.value == 'sell-bb_upper2':
            conditions.append(dataframe["close"] > dataframe['bb_upperband2'])
        if self.sell_trigger.value == 'sell-bb_upper3':
            conditions.append(dataframe["close"] > dataframe['bb_upperband3'])
        if self.sell_trigger.value == 'sell-bb_upper4':
            conditions.append(dataframe["close"] > dataframe['bb_upperband4'])

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe






