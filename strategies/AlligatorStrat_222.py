
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class AlligatorStrat_222(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:
        buys and sells on crossovers - doesn't really perfom that well and its just a proof of concept
    """


    minimal_roi = {
        "0": 0.1



    }


    stoploss = -0.2

    ticker_interval = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:




        dataframe['SMAShort'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['SMAMedium'] = ta.SMA(dataframe, timeperiod=8)
        dataframe['SMALong'] = ta.SMA(dataframe, timeperiod=13)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']












        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (

                (
                qtpylib.crossed_above(dataframe['SMAShort'], dataframe['SMAMedium']) &
                ((dataframe['macd'] > -0.00001)) &
                (dataframe['macd'] > dataframe['macdsignal'])
                )
                |
                qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])






            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (

                ((dataframe['close'] < dataframe['SMAMedium']) &
                (dataframe['macd'] < dataframe['macdsignal'])
                )
                |
                qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])

            ),
            'sell'] = 1
        return dataframe
