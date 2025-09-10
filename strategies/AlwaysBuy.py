from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class AlwaysBuy(IStrategy):


    minimal_roi = {
        "0": 1,
        "100": 2,
        "200": 3,
        "300": -1
        }


    stoploss = -0.2

    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    use_sell_signal = False


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, "buy"] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe
