from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.vendor.qtpylib.indicators import crossed_above
import talib.abstract as ta
from pandas import DataFrame

class advanced_strategy(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
       "0": 0.197,
       "14": 0.054,
       "74": 0.04,
       "85": 0
    }

    stoploss = -0.10

    timeframe = '5m'
    startup_candle_count: int = 50

    buy_rsi = IntParameter(20, 50, default=30)
    sell_rsi = IntParameter(50, 80, default=70)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value) &
                crossed_above(dataframe['rsi'], self.buy_rsi.value)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi.value) &
                crossed_above(dataframe['rsi'], self.sell_rsi.value)
            ),
            'sell'] = 1
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe
