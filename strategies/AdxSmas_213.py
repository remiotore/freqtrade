
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class AdxSmas_213(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AdxSmas.cs

    """



    minimal_roi = {
        "0": 0.15
    }

    stoploss = -0.99

    timeframe = '1h'

    buy_adx = IntParameter(2, 70, default=25, space='buy')
    sell_adx = IntParameter(2, 70, default=25, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['long'] = ta.SMA(dataframe, timeperiod=6)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > self.buy_adx.value) &
                    (qtpylib.crossed_above(dataframe['short'], dataframe['long']))

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (



            ),
            'sell'] = 0
        return dataframe
