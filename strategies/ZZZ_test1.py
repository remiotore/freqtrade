from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class ZZZ_test1(IStrategy):
    
    timeframe = '5m'
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ema_fast = ta.EMA(dataframe, timeperiod=12)
        ema_slow = ta.EMA(dataframe, timeperiod=26)

        dataframe['ema_fast'] = ema_fast
        dataframe['ema_slow'] = ema_slow

        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[dataframe['ema_fast'] > dataframe['ema_slow'], 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[dataframe['ema_fast'] < dataframe['ema_slow'], 'sell'] = 1

        return dataframe