
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np



class Rsiqui(IStrategy):

    minimal_roi = {
        "0": 0.10,
    }

    stoploss = -0.25

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['rsi_gra'] = np.gradient(dataframe['rsi'], 60)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                    (dataframe['rsi'] < 30) &
                    qtpylib.crossed_above(dataframe['rsi_gra'], 0)

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                    (dataframe['rsi'] > 60) &
                    qtpylib.crossed_below(dataframe['rsi_gra'], 0)

            ),
            'sell'] = 1
        return dataframe
