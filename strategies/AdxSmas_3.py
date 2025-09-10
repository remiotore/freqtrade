
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class AdxSmas_3(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AdxSmas.cs

    """



    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.25

    ticker_interval = '1h'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['long'] = ta.SMA(dataframe, timeperiod=6)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > 25) &
                    (qtpylib.crossed_above(dataframe['short'], dataframe['long']))

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] < 25) &
                    (qtpylib.crossed_above(dataframe['long'], dataframe['short']))

            ),
            'sell'] = 0
        return dataframe
