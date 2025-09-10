

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

FTF, STF = 5, 10


class plasma_cutter(IStrategy):


    buy_params = {
        'buy-div-max': 0.96451, 'buy-div-min': 0.22313
    }

    sell_params = {
        'sell-div-max': 0.75476, 'sell-div-min': 0.16599
    }

    minimal_roi = {
        "0": 0.183,
        "486": 0.102,
        "1079": 0.036,
        "4570": 0
    }

    stoploss = -0.193

    trailing_stop = True
    trailing_stop_positive = 0.083
    trailing_stop_positive_offset = 0.173
    trailing_only_offset_is_reached = True






    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['buy-fastMA'] = ta.SMA(dataframe, timeperiod=FTF)
        dataframe['buy-slowMA'] = ta.SMA(dataframe, timeperiod=STF)
        dataframe['sell-fastMA'] = ta.SMA(dataframe, timeperiod=FTF)
        dataframe['sell-slowMA'] = ta.SMA(dataframe, timeperiod=STF)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['buy-fastMA'].div(dataframe['buy-slowMA'])
                    > self.buy_params['buy-div-min']) &
                (dataframe['buy-fastMA'].div(dataframe['buy-slowMA'])
                    < self.buy_params['buy-div-max'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['sell-slowMA'].div(dataframe['sell-fastMA'])
                    > self.sell_params['sell-div-min']) &
                (dataframe['sell-slowMA'].div(dataframe['sell-fastMA'])
                    < self.sell_params['sell-div-max'])
            ),
            'sell'] = 1
        return dataframe
