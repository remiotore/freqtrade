from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from typing import Dict, List
import talib.abstract as ta

class TrendFollowingStrategy_2(IStrategy):
    timeframe = '5m'

    buy_params = {
        "buy_trailing": 0.98,
        "buy_rsi": 53,
        "buy_ema_1": 38,
        "buy_ema_2": 68,
        "buy_williamsr": -98
    }

    sell_params = {
        "sell_trailing": 1.01,
        "sell_rsi": 43,
        "sell_ema_1": 60,
        "sell_ema_2": 28,
        "sell_williamsr": -34
    }

    minimal_roi = {
        "0": 0.05,
        "30": 0.02,
        "60": 0.01,
        "120": 0
    }

    stoploss = -0.1

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    optimal_timeframe = '5m'

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def populate_indicators(self, dataframe: dict, metadata: dict) -> dict:

        dataframe['ema_1'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['ema_2'] = ta.EMA(dataframe, timeperiod=28)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['williams_r'] = ta.WILLR(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: dict, metadata: dict) -> dict:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_1']) &
                (dataframe['ema_1'] > dataframe['ema_2']) &
                (dataframe['rsi'] > self.buy_rsi.value) &
                (dataframe['williams_r'] < self.buy_williamsr.value)
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: dict, metadata: dict) -> dict:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['ema_1']) &
                (dataframe['ema_1'] < dataframe['ema_2']) &
                (dataframe['rsi'] < self.sell_rsi.value) &
                (dataframe['williams_r'] > self.sell_williamsr.value)
            ),
            'sell'
        ] = 1
