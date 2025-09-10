
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import logging
log = logging.getLogger(__name__)



class DoesNothing(IStrategy):
    """

    author@: Gert Wohlgemuth

    just a skeleton

    """



    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.25

    timeframe = '1h'

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
