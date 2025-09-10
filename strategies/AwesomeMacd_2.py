from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class AwesomeMacd_2(IStrategy):
    """
    author@: Gert Wohlgemuth
    converted from:
    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AwesomeMacd.cs
    """



    minimal_roi = {
        "0": 0.8
    }

    stoploss = -0.016

    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=5)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['macd'] > dataframe['macdsignal']) &
                    (dataframe['ao'] > dataframe['ao'].shift())

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['macd'] < dataframe['macdsignal']) &
                    (dataframe['ao'] < dataframe['ao'].shift())

            ),
            'sell'] = 1
        return dataframe

 