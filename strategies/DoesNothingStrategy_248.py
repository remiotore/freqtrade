
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame



class DoesNothingStrategy_248(IStrategy):
    """

    author@: Gert Wohlgemuth

    just a skeleton

    """



    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.25

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
