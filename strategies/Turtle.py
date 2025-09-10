# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta

class Turtle(IStrategy):

    minimal_roi = {
        "0": 10
    }

    stoploss = -0.25

    startup_candle_count = 60

    ticker_interval = '1h'
    timeframe = '1h'

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_only_offset_is_reached = True
    trailing_stop_positive_offset = 0.1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rmax'] = dataframe['close'].rolling(240).max()
        dataframe['rmin'] = dataframe['close'].rolling(120).min()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['high'] >= dataframe['rmax'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['low'] <= dataframe['rmin'])
            ),
            'sell'] = 1
        return dataframe