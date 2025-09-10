
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open
from datetime import timedelta


class LowSeason(IStrategy):

    minimal_roi = {
        "0": 0.05,  # Take profit at 5%
        "30": 0.03,  # Take profit at 3% after 30 minutes
        "60": 0.01  # Take profit at 1% after 60 minutes
    }

    stoploss = -0.05  # Stoploss at 5%

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['upperband'] = bollinger['upperband']
        dataframe['middleband'] = bollinger['middleband']
        dataframe['lowerband'] = bollinger['lowerband']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &  # RSI less than 30
                (dataframe['macd'] > dataframe['macdsignal']) &  # MACD crosses above signal line
                (dataframe['close'] < dataframe['lowerband'])  # Price is below the lower Bollinger Band
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) |  # RSI greater than 70
                (dataframe['close'] > dataframe['upperband'])  # Price is above the upper Bollinger Band
            ),
            'sell'] = 1
        return dataframe
