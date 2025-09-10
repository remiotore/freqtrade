

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta

class Money_m5(IStrategy):

    buy_params = {
        "buy_rsi": 30,
        "buy_ema_short": 10,
        "buy_ema_long": 50,
    }

    sell_params = {
        "sell_rsi": 70,
        "sell_ema_short": 10,
        "sell_ema_long": 50,
    }

    stoploss = -0.4
    can_short = True

    trailing_stop = True
    trailing_stop_positive = 0.5
    trailing_stop_positive_offset = 0.6
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    buy_rsi = IntParameter(20, 40, default=30, space='buy')
    buy_ema_short = IntParameter(5, 15, default=10, space='buy')
    buy_ema_long = IntParameter(20, 60, default=50, space='buy')

    sell_rsi = IntParameter(60, 80, default=70, space='sell')
    sell_ema_short = IntParameter(5, 15, default=10, space='sell')
    sell_ema_long = IntParameter(20, 60, default=50, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.buy_ema_short.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.buy_ema_long.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value) &  # Signal: RSI less than buy_rsi
                (dataframe['ema_short'] > dataframe['ema_long'])  # Signal: Short EMA above long EMA
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi.value) &  # Signal: RSI greater than sell_rsi
                (dataframe['ema_short'] < dataframe['ema_long'])  # Signal: Short EMA below long EMA
            ),
            'enter_short'] = 1
        return dataframe
