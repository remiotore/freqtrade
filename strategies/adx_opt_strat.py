
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta



class adx_opt_strat(IStrategy):
    """
    author@: Gert Wohlgemuth
    converted from:
        https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AdxMomentum.cs
    """



    minimal_roi = {
        "0": 0.0692,
        "7": 0.02682,
        "10": 0.00771,
        "32": 0
    }

    stoploss = -0.32766

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.32634
    trailing_stop_positive_offset = 0.34487  

    ticker_interval = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['mom'] < 0) &
                    (dataframe['minus_di'] > 48) &
                    (dataframe['plus_di'] < dataframe['minus_di'])

            ),
            'buy'] = 1
        
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['mom'] > 0) &
                    (dataframe['minus_di'] > 48) &
                    (dataframe['plus_di'] > dataframe['minus_di'])

            ),
            'sell'] = 1
        return dataframe