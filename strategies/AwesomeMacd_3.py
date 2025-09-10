
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class AwesomeMacd_3(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AwesomeMacd.cs

    """



    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.25

    ticker_interval = '1h'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['macd'] > 0) &
                    (dataframe['ao'] > 0) &
                    (dataframe['ao'].shift() < 0)

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['macd'] < 0) &
                    (dataframe['ao'] < 0) &
                    (dataframe['ao'].shift() > 0)

            ),
            'sell'] = 0
        return dataframe
