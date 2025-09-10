# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge

rsi_trigger = 45

pre_buy_signal_count = 1
pre_buy_signal_count_window = 1

class RSIResampleV2(IStrategy):
    # ROI table:
    minimal_roi = {
        "0": 0.05,
        "1440": 0.01,
    }

    # Stoploss:
    stoploss = -0.25

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.5
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_sell_signal = True

    plot_config = {
        'main_plot': {
            'sma_5': {'color': 'orange'},
        },
        'subplots': {
            "Signals": {
                'rsi': {'color': 'blue'},
            },
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']
        dataframe['plus_di'] = ta.PLUS_DI(high, low, close, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(high, low, close, timeperiod=14)

        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe['rsi'] < 45) &
                        (dataframe['close'] < dataframe['sma_5']) &
                        (dataframe['volume'] > 0)
                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe['rsi'] > 65) &
                        (dataframe['volume'] > 0)
                )
            ),
            'sell'] = 1
        return dataframe
