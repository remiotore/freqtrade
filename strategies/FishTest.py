import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, DatetimeIndex, merge, Series
from technical.indicators import hull_moving_average

"""
Autor: https://github.com/werkkrew/freqtrade-strategies
"""

class FishTest(IStrategy):

    # Buy hyperspace params:
    buy_params = {}

    # Sell hyperspace params:
    sell_params = {}
    
    max_open_trades = 2

    # ROI table:
    minimal_roi = {'0': 0.10}

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    timeframe = '15m'
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True
    
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration": 600
            },
            {
                "method": "StoplossGuard",
                "lookback_period": 1440,
                "trade_limit": 4,
                "stop_duration": 2880,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 1440,
                "trade_limit": 2,
                "stop_duration": 1440,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period": 1440,
                "trade_limit": 4,
                "stop_duration": 1440,
                "required_profit": 0.01
            }
        ]


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['hma'] = hull_moving_average(dataframe, 40, 'close')
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=40)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=40)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            ( 
            (dataframe['hma'] < dataframe['hma'].shift()) &
            (dataframe['cci'] <= -00.0) &
            (dataframe['fisher_rsi'] < -0.2) &
            (dataframe['volume'] > 0)
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        return dataframe
