from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class ADX_Boll_Strategy(IStrategy):
    timeframe = '1h'
    
    def informative_pairs(self):
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_upperband'] = bollinger['upperband']

        dataframe['adx'] = ta.ADX(dataframe)
        
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['adx'] > 25) &
                (dataframe['close'] < dataframe['bb_lowerband'])
            ),
            'buy'] = 1
        return dataframe
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['adx'] > 25) &
                (dataframe['close'] > dataframe['bb_upperband'])
            ),
            'sell'] = 1
        return dataframe
