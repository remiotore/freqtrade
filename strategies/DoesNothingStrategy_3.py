
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame



class DoesNothingStrategy_3(IStrategy):
    """

    author@: Gert Wohlgemuth

    just a skeleton

    """



    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.25

    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 0
        return dataframe
