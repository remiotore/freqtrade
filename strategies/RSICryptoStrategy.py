from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class RSICryptoStrategy(IStrategy):
    minimal_roi = {"0": 0.10}
    stoploss = -0.1
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > self.RSI_long.value),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] < self.RSI_close.value),
            'exit_long'
        ] = 1

        if self.use_emergency.value:
            dataframe.loc[
                (dataframe['rsi'] < self.Emergency_close.value),
                'exit_long'
            ] = 1
        return dataframe
