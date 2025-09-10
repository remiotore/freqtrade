from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (IntParameter, IStrategy, CategoricalParameter)
from random import random
import numpy as np

class fa_random_strategy(IStrategy):
    """
    This is FrostAura's random strategy powered by nature.

    Last Optimization:
        Profit %        : 10-30%
        Optimized for   : Last 45 days, 1h
        Avg             : 2d - 5d
    """


    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.347,
        "450": 0.106,
        "1169": 0.032,
        "1624": 0
    }

    stoploss = -0.231

    trailing_stop = False

    timeframe = '1h'

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['random_number'] = np.random.randint(0, 100, dataframe.shape[0])

        return dataframe

    buy_prediction_delta_direction = CategoricalParameter(['<', '>'], default='>', space='buy')
    buy_probability = IntParameter([0, 100], default=76, space='buy')

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        random_number = dataframe['random_number']

        dataframe.loc[
            (
                (random_number < self.buy_probability.value if self.buy_prediction_delta_direction.value == '<' else random_number > self.buy_probability.value)
            ),
            'buy'] = 1

        return dataframe

    sell_prediction_delta_direction = CategoricalParameter(['<', '>'], default='<', space='sell')
    sell_probability = IntParameter([0, 100], default=0, space='sell')

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        random_number = dataframe['random_number']

        dataframe.loc[
            (
                (random_number < self.sell_probability.value if self.sell_prediction_delta_direction.value == '<' else random_number > self.sell_probability.value)
            ),
            'sell'] = 1

        return dataframe
