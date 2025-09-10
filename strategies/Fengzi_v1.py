from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
from freqtrade.persistence import Trade
from datetime import datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta

class Fengzi_v1(IStrategy):
    stoploss = -0.99
    can_short = True

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.5  # Disabled / not configured

    timeframe = "5m"

    minimal_roi = {
        "0": 0.5,
        "30": 0.45,
        "60": 0.4,
        "120": 0.3,
        "240": 0.25,
        "480": 0.05
    }

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    startup_candle_count: int = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        length = 25
        dataframe['rsi_3'] = ta.RSI(dataframe, timeperiod=2)
        dataframe['rsi_6'] = ta.RSI(dataframe, timeperiod=4)

        dataframe['pivot_high'] = dataframe['high'][::-1].rolling(window=length, min_periods=1).max()[::-1]
        dataframe['pivot_high'] = dataframe['pivot_high'].where(dataframe['high'] == dataframe['pivot_high'], np.nan)

        dataframe['pivot_low'] = dataframe['low'][::-1].rolling(window=length, min_periods=1).min()[::-1]
        dataframe['pivot_low'] = dataframe['pivot_low'].where(dataframe['low'] == dataframe['pivot_low'], np.nan)

        dataframe['high_swing'] = dataframe['pivot_high'].shift(length // 2)
        dataframe['low_swing'] = dataframe['pivot_low'].shift(length // 2)




        dataframe['high_swing'] = dataframe['high_swing'].fillna(0)
        dataframe['low_swing'] = dataframe['low_swing'].fillna(0)


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        dataframe.loc[
            (qtpylib.crossed_above(dataframe['rsi_3'], dataframe['rsi_6'])) &
            (dataframe['low_swing'] != 0) & 
            (dataframe['close'] > dataframe['low_swing'].shift(1)) &
            (dataframe['enter_long'].shift().fillna(0) == 0) &  # Перевірка на відсутність попередніх сигналів
            (dataframe['enter_short'].shift().fillna(0) == 0),  # Перевірка на відсутність попередніх сигналів
            'enter_long'
        ] = 1

        dataframe.loc[
            (qtpylib.crossed_below(dataframe['rsi_3'], dataframe['rsi_6'])) &
            (dataframe['high_swing'] != 0) & 
            (dataframe['close'] < dataframe['high_swing'].shift(1)) &
            (dataframe['enter_long'].shift().fillna(0) == 0) &  # Перевірка на відсутність попередніх сигналів
            (dataframe['enter_short'].shift().fillna(0) == 0),  # Перевірка на відсутність попередніх сигналів
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        dataframe.loc[
            (qtpylib.crossed_above(dataframe['rsi_3'], dataframe['rsi_6'])) &
            (dataframe['low_swing'] != 0) & 
            (dataframe['close'] > dataframe['low_swing'].shift(1)) &
            (dataframe['enter_long'].shift().fillna(0) == 0) &  # Перевірка на відсутність попередніх сигналів
            (dataframe['enter_short'].shift().fillna(0) == 0),  # Перевірка на відсутність попередніх сигналів
            'exit_short'
        ] = 1

        dataframe.loc[
            (qtpylib.crossed_below(dataframe['rsi_3'], dataframe['rsi_6'])) &
            (dataframe['high_swing'] != 0) & 
            (dataframe['close'] < dataframe['high_swing'].shift(1)) &
            (dataframe['enter_long'].shift().fillna(0) == 0) &  # Перевірка на відсутність попередніх сигналів
            (dataframe['enter_short'].shift().fillna(0) == 0),  # Перевірка на відсутність попередніх сигналів
            'exit_long'
        ] = 1

        return dataframe
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return 5    
