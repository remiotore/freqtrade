from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib

class MyStrategy(IStrategy):
    timeframe = '5m'
    minimal_roi = {
        "60": 0.10,
        "30": 0.05,
        "0": 0.02
    }
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = talib.EMA(dataframe['close'], timeperiod=12)
        dataframe['ema_slow'] = talib.EMA(dataframe['close'], timeperiod=26)
        macd, macdsignal, macdhist = talib.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd
        dataframe['macdsignal'] = macdsignal
        dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=14)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['rsi'] < 30), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['rsi'] > 70), 'sell'] = 1
        return dataframe

