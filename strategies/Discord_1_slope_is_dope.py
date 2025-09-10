# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from scipy.spatial.distance import cosine
import numpy as np


class slope_is_dope(IStrategy):
    # Minimal ROI designed for the strategy.

    INTERFACE_VERSION: int = 3

    minimal_roi = {
        "0": 0.6
        }

    stoploss = -0.9

    timeframe = '4h'

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.28

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)

        # Over all Moving average
        dataframe['marketMA'] = ta.SMA(dataframe, timeperiod=200)

        # Fast & Slow Moving average
        dataframe['fastMA'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['slowMA'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['entryMA'] = ta.SMA(dataframe, timeperiod=3)

        # Calculate slope of slowMA
        # See: https://www.wikihow.com/Find-the-Slope-of-a-Line
        dataframe['sy1'] = dataframe['slowMA'].shift(+11)
        dataframe['sy2'] = dataframe['slowMA'].shift(+1)
        sx1 = 1
        sx2 = 11
        dataframe['sy'] = dataframe['sy1'] - dataframe['sy1']
        dataframe['sx'] = sx2 - sx1
        dataframe['slow_slope'] = dataframe['sy'] / dataframe['sx']

        # Calculate slope of fastMA
        dataframe['fy1'] = dataframe['fastMA'].shift(+11)
        dataframe['fy2'] = dataframe['fastMA'].shift(+1)
        fx1 = 1
        fx2 = 11
        dataframe['fy'] = dataframe['fy2'] - dataframe['fy1']
        dataframe['fx'] = fx2 - fx1
        dataframe['fast_slope'] = dataframe['fy'] / dataframe['fx']
        # print(dataframe[['date','close', 'slow_slope','fast_slope']].tail(50))

        # ==== Trailing custom stoploss indicator ====
        dataframe['last_lowest'] = dataframe['low'].rolling(10).min().shift(1)

        return dataframe

    # required for graphing
    plot_config = {
        "main_plot": {
            # Configuration for main plot indicators.
            "fastMA": {"color": "red"},
            "slowMA": {"color": "blue"},
            },
            "subplots": {
                # Additional subplots
                "rsi": {"rsi": {"color": "blue"}},
                "fast_slope": {"fast_slope": {"color": "red"}, "slow_slope": {"color": "blue"}},
                },
        }

    # Indicating the buy trend
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Only enter when market is bullish (this is a choice)
                (dataframe['close'] > dataframe['marketMA']) &
                # Only trade when the fast slope is above 0
                (dataframe['fast_slope'] > 0) &
                # Only trade when the slow slope is above 0
                (dataframe['slow_slope'] > 0) &
                # Only buy when the close price is higher than the 3day average of ten periods ago
                (dataframe['close'] > dataframe['entryMA'].shift(+11)) &
                # Or only buy when the close price is higher than the close price of 3 days ago (this is a choice)
                # (dataframe['close'] > dataframe['close'].shift(+11)) &
                # Only enter trades when the RSI is higher than 55
                (dataframe['rsi'] > 55) &
                # Only trade when the fast MA is above the slow MA
                (dataframe['fastMA'] > dataframe['slowMA'])
                # Or trade when the fase MA crosses above the slow MA (This is a choice...)
                #(qtpylib.crossed_above(dataframe['fastMA'], dataframe['slowMA']))
            ),
            'enter_long',
        ] = 1

        return dataframe

    # Indicating the sell trend
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Close or do not trade when fastMA is below slowMA
                (dataframe['fastMA'] < dataframe['slowMA'])
                # Or close position when the close price gets below the last lowest candle price configured
                # (AKA candle based (Trailing) stoploss) 
                | (dataframe['close'] < dataframe['last_lowest'])
                # | (dataframe['close'] < dataframe['fastMA'])
            ),
            'exit_long',
        ] = 1

        return dataframe