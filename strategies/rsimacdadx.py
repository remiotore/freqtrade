
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class rsimacdadx(IStrategy):
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    stoploss = -0.10
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] < 28) &
                    (dataframe['macd'] < dataframe['macdsignal']) &
                    (dataframe['adx'] > 40)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 70) &
                    (dataframe['macd'] > dataframe['macdsignal']) &
                    (dataframe['adx'] > 40)

            ),
            'sell'] = 1
        return dataframe
