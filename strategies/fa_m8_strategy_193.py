import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (IntParameter, IStrategy, CategoricalParameter)
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class fa_m8_strategy_193(IStrategy):
    """
    This is FrostAura's mark 8 strategy which aims to make purchase decisions
    based on the RSI & overall performance of the asset from it's previous candlesticks.
    
    Last Optimization:
        Profit %        : 12.20%
        Optimized for   : Last 45 days, 4h
        Avg             : 4d 20h 48m
    """


    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.684,
        "864": 0.267,
        "3186": 0.057,
        "6992": 0
    }

    stoploss = -0.277

    trailing_stop = False

    timeframe = '4h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    order_types = {
        'buy': 'market',
        'sell': 'market',
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

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe

    buy_rsi = IntParameter([20, 80], default=71, space='buy')
    buy_rsi_direction = CategoricalParameter(['<', '>'], default='<', space='buy')

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        minimum_coin_price = 0.0000015

        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value if self.buy_rsi_direction.value == '<' else dataframe['rsi'] > self.buy_rsi.value) &
                (dataframe['close'] > minimum_coin_price)
            ),
            'buy'] = 1

        return dataframe

    sell_rsi = IntParameter([20, 80], default=43, space='sell')
    sell_rsi_direction = CategoricalParameter(['<', '>'], default='>', space='sell')
    sell_percentage = IntParameter([1, 50], default=12, space='sell')

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        previous_close = dataframe['close'].shift(1)
        current_close = dataframe['close']
        percentage_price_delta = ((previous_close - current_close) / previous_close) * -100
        
        dataframe.loc[
            (
                (dataframe['rsi'] < self.sell_rsi.value if self.sell_rsi_direction.value == '<' else dataframe['rsi'] > self.sell_rsi.value) |
                (percentage_price_delta > self.sell_percentage.value)
            ),
            'sell'] = 1
        
        return dataframe