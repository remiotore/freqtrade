from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
import numpy as np


class buysell(IStrategy):

    minimal_roi = {
        "0": 0.154,
        "18": 0.074,
        "50": 0.039,
        "165": 0.02
    }
    stoploss = -0.051
    timeframe = '5m'




    trailing_stop = True
    trailing_stop_positive = 0.009
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["buy"] = np.random.randint(0, 2, size=len(dataframe))
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sell"] = 0
        return dataframe







