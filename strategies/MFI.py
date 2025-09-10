
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class MFI(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs

    """



    minimal_roi = {
        "1440": 0.01,
        "80": 0.02,
        "40": 0.03,
        "20": 0.04,
        "0":  0.05
    }


    stoploss = -0.10

    timeframe = '5m'

    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['MFI'] = ta.MFI(dataframe, timeperiod=14)




        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['MFI'].shift() <= 14)

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['MFI'] >= 75)

            ),
            'sell'] = 1
        return dataframe
