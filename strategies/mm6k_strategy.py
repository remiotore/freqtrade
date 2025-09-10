







import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class mm6k_strategy(IStrategy):
    INTERFACE_VERSION = 2
    minimal_roi = {
        "60": 0.02,
        "30": 0.03,
        "0": 0.04
    }
    stoploss = -0.10
    trailing_stop = False
    timeframe = '1m'
    process_only_new_candles = False
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    startup_candle_count = 30
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
    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    rsi_buy_enabled = CategoricalParameter([True, False])
    rsi_sell_enabled = CategoricalParameter([True, False])
    rsi_buy = IntParameter(20, 45, default=30)
    rsi_sell = IntParameter(60, 85, default=70)

    def informative_pairs(self):
        return []

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict
    ) -> DataFrame:

        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
            / dataframe["bb_middleband"]
        )

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_buy_trend(
        self,
        dataframe: DataFrame,
        metadata: dict
    ) -> DataFrame:
        conditions = []

        if self.rsi_buy_enabled.value:
            conditions.append(
                qtpylib.crossed_above(
                    dataframe['rsi'],
                    self.rsi_buy.value
                )
            )

        conditions.append(dataframe['tema'] <= dataframe['bb_middleband'])
        conditions.append(dataframe['tema'] > dataframe['tema'].shift(1))
        conditions.append(dataframe['volume'] > 0)

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(
        self,
        dataframe: DataFrame,
        metadata: dict
    ) -> DataFrame:
        conditions = []

        if self.rsi_sell_enabled.value:
            conditions.append(
                qtpylib.crossed_below(
                    dataframe['rsi'],
                    self.rsi_sell.value
                )
            )
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 70))
                & (dataframe['tema'] > dataframe['bb_middleband'])
                & (dataframe['tema'] < dataframe['tema'].shift(1))
                & (dataframe['volume'] > 0)
            ),
            'sell'
        ] = 1
        return dataframe
