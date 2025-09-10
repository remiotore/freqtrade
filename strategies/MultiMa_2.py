



from freqtrade.strategy.parameters import IntParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce


class MultiMa_2(IStrategy):


    buy_params = {
        "buy_ma_count": 4,
        "buy_ma_gap": 15,
    }

    sell_params = {
        "sell_ma_count": 12,
        "sell_ma_gap": 68,
    }

    minimal_roi = {
        "0": 0.523,
        "1553": 0.123,
        "2332": 0.076,
        "3169": 0
    }

    stoploss = -0.345

    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    timeframe = "4h"

    count_max = 20
    gap_max = 100

    buy_ma_count = IntParameter(1, count_max, default=7, space="buy")
    buy_ma_gap = IntParameter(1, gap_max, default=7, space="buy")

    sell_ma_count = IntParameter(1, count_max, default=7, space="sell")
    sell_ma_gap = IntParameter(1, gap_max, default=94, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for count in range(self.count_max):
            for gap in range(self.gap_max):
                if count*gap > 1 and count*gap not in dataframe.keys():
                    dataframe[count*gap] = ta.TEMA(
                        dataframe, timeperiod=int(count*gap)
                    )
        print(" ", metadata['pair'], end="\t\r")

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []



        for ma_count in range(self.buy_ma_count.value):
            key = ma_count*self.buy_ma_gap.value
            past_key = (ma_count-1)*self.buy_ma_gap.value
            if past_key > 1 and key in dataframe.keys() and past_key in dataframe.keys():
                conditions.append(dataframe[key] < dataframe[past_key])

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "buy"] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        for ma_count in range(self.sell_ma_count.value):
            key = ma_count*self.sell_ma_gap.value
            past_key = (ma_count-1)*self.sell_ma_gap.value
            if past_key > 1 and key in dataframe.keys() and past_key in dataframe.keys():
                conditions.append(dataframe[key] > dataframe[past_key])

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1
        return dataframe
