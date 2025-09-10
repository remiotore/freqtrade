





from freqtrade.strategy.hyper import IntParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce


class s13(IStrategy):

    buy_ma_count = IntParameter(2, 10, default=10, space='buy')
    buy_ma_gap = IntParameter(2, 10, default=2, space='buy')
    buy_ma_shift = IntParameter(0, 10, default=0, space='buy')


    sell_ma_count = IntParameter(2, 10, default=10, space='sell')
    sell_ma_gap = IntParameter(2, 10, default=2, space='sell')
    sell_ma_shift = IntParameter(0, 10, default=0, space='sell')


    minimal_roi = {
        "0": 0.30873,
        "569": 0.16689,
        "3211": 0.06473,
        "7617": 0
    }

    stoploss = -0.128

    timeframe = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:




        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for i in self.buy_ma_count.range:
            dataframe[f'buy-ma-{i+1}'] = ta.SMA(dataframe,
                                                timeperiod=int((i+1) * self.buy_ma_gap.value))

        conditions = []

        for i in self.buy_ma_count.range:
            if i > 1:
                shift = self.buy_ma_shift.value
                for shift in self.buy_ma_shift.range:
                    conditions.append(
                        dataframe[f'buy-ma-{i}'].shift(shift) >
                        dataframe[f'buy-ma-{i-1}'].shift(shift)
                    )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy']=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for i in self.sell_ma_count.range:
            dataframe[f'sell-ma-{i+1}'] = ta.SMA(dataframe,
                                                 timeperiod=int((i+1) * self.sell_ma_gap.value))

        conditions = []

        for i in self.sell_ma_count.range:
            if i > 1:
                shift = self.sell_ma_shift.value
                for shift in self.sell_ma_shift.range:
                    conditions.append(
                        dataframe[f'sell-ma-{i}'].shift(shift) <
                        dataframe[f'sell-ma-{i-1}'].shift(shift)
                    )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell']=1
        return dataframe
