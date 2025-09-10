import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class FrostAuraM21hStrategy(IStrategy):
    """
    This is FrostAura's mark 2 strategy which aims to make purchase decisions
    based on the Stochastic and RSI.
    
    Last Optimization:
        Sharpe Ratio    : 6.24747% (prev 6.41952)
        Profit %        : 1537.94% (1432.33%)
        Optimized for   : Last 109 days, 1h
        ATT             : 719.4m (prev 4321.0m)
    """


    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.32365,
        "359": 0.12673,
        "934": 0.08834,
        "2090": 0
    }

    stoploss = -0.44897

    trailing_stop = False

    timeframe = '15m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

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

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe)

        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        minimum_coin_price = 0.0000015
        
        dataframe.loc[
            (
                (dataframe['rsi'] > 48) &
                (dataframe["slowd"] > 79) &
                (dataframe["slowk"] > 77) &
                (dataframe["close"] > minimum_coin_price)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 48) &
                (dataframe["slowd"] < 79) &
                (dataframe["slowk"] < 77)
            ),
            'sell'] = 1
        
        return dataframe