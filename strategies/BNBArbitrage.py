from freqtrade.strategy import IStrategy, merge_informative_pair
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
import numpy as np


class BNBArbitrage(IStrategy):
    INTERFACE_VERSION = 3

    # 交易时间周期
    timeframe = '5m'

    # 使用一个较长周期来观察BNB价格
    informative_timeframe = '1h'

    # 最小回报
    minimal_roi = {
        "0": 0.03
    }

    stoploss = -0.05

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    leverage = 3
    position_adjustment_enable = True
    use_custom_stoploss = False
    use_custom_exit = False
    use_custom_entry = False
    can_short = True  # 支持做空

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 添加移动平均线
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)

        # RSI 判断趋势强弱
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger Bands
        boll = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lower'] = boll['lower']
        dataframe['bb_upper'] = boll['upper']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema_fast'] > dataframe['ema_slow']) &
                (dataframe['close'] < dataframe['bb_lower']) &
                (dataframe['rsi'] < 30)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['ema_fast'] < dataframe['ema_slow']) &
                (dataframe['close'] > dataframe['bb_upper']) &
                (dataframe['rsi'] > 70)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 60)
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] < 40)
            ),
            'exit_short'] = 1

        return dataframe

