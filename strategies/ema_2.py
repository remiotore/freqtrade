from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
from technical import qtpylib

class ema_2(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {'0': 0.01}

    stoploss = -0.1

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = '5m'
    can_short = True

    startup_candle_count = 150

    bb_period = IntParameter(10, 50, default=20, space='buy', optimize=True)
    bb_stddev = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)
    rsi_period = IntParameter(10, 50, default=14, space='buy', optimize=True)
    rsi_bb_period = IntParameter(10, 50, default=20, space='buy', optimize=True)
    rsi_bb_stddev = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)
    roi_profit = DecimalParameter(0.01, 0.1, default=0.01, space='sell', optimize=True)
    stoploss_value = DecimalParameter(-0.3, -0.01, default=-0.1, space='sell', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb = ta.BBANDS(dataframe['close'], timeperiod=self.bb_period.value, nbdevup=self.bb_stddev.value, nbdevdn=self.bb_stddev.value)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_lower'] = bb['lowerband']

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        rsi_bb = ta.BBANDS(dataframe['rsi'], timeperiod=self.rsi_bb_period.value, nbdevup=self.rsi_bb_stddev.value, nbdevdn=self.rsi_bb_stddev.value)
        dataframe['rsi_bb_upper'] = rsi_bb['upperband']
        dataframe['rsi_bb_lower'] = rsi_bb['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[qtpylib.crossed_above(dataframe['close'], dataframe['bb_lower']) & qtpylib.crossed_above(dataframe['rsi'], dataframe['rsi_bb_lower']), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # RSI is above the upper Bollinger Band
        dataframe.loc[qtpylib.crossed_below(dataframe['close'], dataframe['bb_upper']) & qtpylib.crossed_below(dataframe['rsi'], dataframe['rsi_bb_upper']), 'exit_long'] = 1
        return dataframe