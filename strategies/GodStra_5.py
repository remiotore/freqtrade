
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import CategoricalParameter, IntParameter, RealParameter
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
from functools import reduce

class GodStra_5(IStrategy):

    timeframe = '12h'
    startup_candle_count = 240


    buy_ema_short = IntParameter(10, 50, default=21, space="buy", optimize=True)
    buy_ema_long = IntParameter(50, 200, default=100, space="buy", optimize=True)
    buy_rsi_lower = IntParameter(20, 40, default=30, space="buy", optimize=True)  # RSI oversold threshold
    buy_rsi_upper = IntParameter(50, 70, default=60, space="buy", optimize=True)  # RSI upper limit for buying
    buy_volume_check = CategoricalParameter([True, False], default=True, space="buy", optimize=True)

    sell_ema_short = IntParameter(10, 50, default=21, space="sell", optimize=True)
    sell_ema_long = IntParameter(50, 200, default=100, space="sell", optimize=True)
    sell_rsi_lower = IntParameter(50, 70, default=60, space="sell", optimize=True)  # RSI lower limit for selling
    sell_rsi_upper = IntParameter(70, 90, default=80, space="sell", optimize=True)  # RSI overbought threshold
    sell_volume_check = CategoricalParameter([True, False], default=True, space="sell", optimize=True)

    minimal_roi = {
        "0": 0.02,
        "240": 0.01,
        "720": 0
    }

    stoploss = -0.06

    trailing_stop = True
    trailing_stop_positive = 0.004
    trailing_stop_positive_offset = 0.008
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_custom_stoploss = False

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe['ema_short_buy'] = ta.EMA(dataframe['close'], timeperiod=self.buy_ema_short.value)
        dataframe['ema_long_buy'] = ta.EMA(dataframe['close'], timeperiod=self.buy_ema_long.value)
        dataframe['ema_short_sell'] = ta.EMA(dataframe['close'], timeperiod=self.sell_ema_short.value)
        dataframe['ema_long_sell'] = ta.EMA(dataframe['close'], timeperiod=self.sell_ema_long.value)

        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']

        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'])
        dataframe['bb_upperband'], dataframe['bb_middleband'], dataframe['bb_lowerband'] = ta.BBANDS(dataframe['close'])

        dataframe['mom'] = ta.MOM(dataframe['close'], timeperiod=10)
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)

        dataframe.fillna(0, inplace=True)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        conditions = []

        conditions.append(dataframe['ema_short_buy'] > dataframe['ema_long_buy'])


        conditions.append(
            (dataframe['rsi'] > self.buy_rsi_lower.value) & 
            (dataframe['rsi'] < self.buy_rsi_upper.value)
        )

        if self.buy_volume_check.value:
            conditions.append(dataframe['volume_ratio'] > 1.0)

        conditions.append(dataframe['macd'] > dataframe['macdsignal'])

        conditions.append(dataframe['close'] > dataframe['bb_middleband'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        conditions = []

        conditions.append(dataframe['ema_short_sell'] < dataframe['ema_long_sell'])


        conditions.append(
            (dataframe['rsi'] > self.sell_rsi_lower.value) | 
            (dataframe['rsi'] > self.sell_rsi_upper.value)
        )

        if self.sell_volume_check.value:
            conditions.append(dataframe['volume_ratio'] < 0.8)

        conditions.append(dataframe['macd'] < dataframe['macdsignal'])

        conditions.append(dataframe['close'] < dataframe['bb_middleband'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'
            ] = 1

        return dataframe
