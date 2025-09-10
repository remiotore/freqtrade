from pandas import DataFrame
from technical.indicators import cmf

from freqtrade.strategy.interface import IStrategy


class TechnicalExampleStrategy_2(IStrategy):
    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.05

    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['cmf'] = cmf(dataframe, 21)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe['cmf'] < 0)

                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:

        dataframe.loc[
            (
                    (dataframe['cmf'] > 0)
            ),
            'sell'] = 1
        return dataframe
