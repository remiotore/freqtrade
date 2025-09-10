# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class SAR_and_BB(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 1
    }

    # Stoploss:
    stoploss = -0.14

    # Trailing stop:
    trailing_stop = True

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
 
        # EMA - Exponential Moving Average
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=144)

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        '''
        SAR and BB band
        '''
        dataframe.loc[
            (
                (dataframe['volume'] > 0) &
                (
                    (
                        (qtpylib.crossed_above(dataframe['close'], dataframe['sar'])) &
                        (dataframe['close'] > dataframe['ema'])
                    )
                    |
                    (
                        (dataframe['close'] < dataframe['ema']) &
                        (dataframe['close'] < 0.99 * dataframe['bb_lowerband']) &
                        (dataframe['volume'] < (dataframe['volume'].rolling(window=30).mean().shift(1) * 20))
                    )
                )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['volume'] > 0) &
                (
                    qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
                    |
                    (dataframe['close'] >= 1.05*dataframe['bb_middleband'])
                )
            ),
            'sell'] = 1
        return dataframe
