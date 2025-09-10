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
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open


# PLEAS CHANGE THIS SETTINGS WITH BACKTEST
base_nb_candles = 1 #something higher than 1
low_offset = 1 # something lower than 1
high_offset = 1 # something higher than 1


class SMAOffset(IStrategy):
    # ROI table:
    minimal_roi = {
        "0": 1,
    }

    # Stoploss:
    stoploss = -0.5

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.5
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False

    process_only_new_candles = True
    startup_candle_count = base_nb_candles

    plot_config = {
        'main_plot': {
            'sma_30_offset': {'color': 'orange'},
            'sma_30_offset_pos': {'color': 'orange'},
        },
    }

    use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=1200) > trade.open_date_utc and current_profit < -0.05:
            return -0.001

        # return maximum stoploss value, keeping current stoploss price unchanged
        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['sma_30_offset'] = ta.SMA(dataframe, timeperiod=base_nb_candles) * low_offset
        dataframe['sma_30_offset_pos'] = ta.SMA(dataframe, timeperiod=base_nb_candles) * high_offset

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['sma_30_offset']) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['sma_30_offset_pos']) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
