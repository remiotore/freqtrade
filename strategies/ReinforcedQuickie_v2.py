# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa

class ReinforcedQuickie_v2(IStrategy):
    """
    Modified version of ReinforcedQuickie strategy to avoid sudden drops.
    Author@: Gert Wohlgemuth (Modified by Assistant)
    """

    INTERFACE_VERSION: int = 3
    minimal_roi = {
        "0": 0.01
    }
    stoploss = -0.05
    timeframe = '5m'
    resample_factor = 12

    EMA_SHORT_TERM = 5
    EMA_MEDIUM_TERM = 12
    EMA_LONG_TERM = 21

    # 新增 ADX 指標
    ADX_PERIOD = 14
    ADX_THRESHOLD = 25  # 只有當 ADX 超過 25 時，認為趨勢強勁

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.resample(dataframe, self.timeframe, self.resample_factor)

        # 計算 EMA
        dataframe['ema_{}'.format(self.EMA_SHORT_TERM)] = ta.EMA(
            dataframe, timeperiod=self.EMA_SHORT_TERM
        )
        dataframe['ema_{}'.format(self.EMA_MEDIUM_TERM)] = ta.EMA(
            dataframe, timeperiod=self.EMA_MEDIUM_TERM
        )
        dataframe['ema_{}'.format(self.EMA_LONG_TERM)] = ta.EMA(
            dataframe, timeperiod=self.EMA_LONG_TERM
        )

        # 計算布林帶
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # 計算其他指標
        dataframe['min'] = ta.MIN(dataframe, timeperiod=self.EMA_MEDIUM_TERM)
        dataframe['max'] = ta.MAX(dataframe, timeperiod=self.EMA_MEDIUM_TERM)
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['average'] = (dataframe['close'] + dataframe['open'] + dataframe['high'] + dataframe['low']) / 4

        # 計算 MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # 計算 ADX
        adx = ta.ADX(dataframe, timeperiod=self.ADX_PERIOD)
        dataframe['adx'] = adx

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根據技術指標填充買入信號
        """
        dataframe.loc[
            (
                (
                    (
                        (dataframe['close'] < dataframe['ema_{}'.format(self.EMA_SHORT_TERM)]) &
                        (dataframe['close'] < dataframe['ema_{}'.format(self.EMA_MEDIUM_TERM)]) &
                        (dataframe['close'] == dataframe['min']) &
                        (dataframe['close'] <= dataframe['bb_lowerband'])
                    )
                    |
                    (
                        (dataframe['average'].shift(5) > dataframe['average'].shift(4)) &
                        (dataframe['average'].shift(4) > dataframe['average'].shift(3)) &
                        (dataframe['average'].shift(3) > dataframe['average'].shift(2)) &
                        (dataframe['average'].shift(2) > dataframe['average'].shift(1)) &
                        (dataframe['average'].shift(1) < dataframe['average'].shift(0)) &
                        (dataframe['low'].shift(1) < dataframe['bb_middleband']) &
                        (dataframe['cci'].shift(1) < -100) &
                        (dataframe['rsi'].shift(1) < 30) &
                        (dataframe['mfi'].shift(1) < 30)
                    )
                )
                &
                (
                    (dataframe['volume'] < (dataframe['volume'].rolling(window=30).mean().shift(1) * 20)) &
                    (dataframe['resample_sma'] < dataframe['close']) &
                    (dataframe['resample_sma'].shift(1) < dataframe['resample_sma']) &
                    (dataframe['adx'] > self.ADX_THRESHOLD)  # 加入 ADX 過濾
                )
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根據技術指標填充賣出信號
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_{}'.format(self.EMA_SHORT_TERM)]) &
                (dataframe['close'] > dataframe['ema_{}'.format(self.EMA_MEDIUM_TERM)]) &
                (dataframe['close'] >= dataframe['max']) &
                (dataframe['close'] >= dataframe['bb_upperband']) &
                (dataframe['mfi'] > 80)
            ) |

            (
                (dataframe['open'] < dataframe['close']) &
                (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) < dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) < dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) < dataframe['close'].shift(4)) &
                (dataframe['open'].shift(5) < dataframe['close'].shift(5)) &
                (dataframe['open'].shift(6) < dataframe['close'].shift(6)) &
                (dataframe['open'].shift(7) < dataframe['close'].shift(7)) &
                (dataframe['rsi'] > 70)
            ),
            'exit_long'
        ] = 1
        return dataframe

    def resample(self, dataframe, interval, factor):
        """
        進行重採樣以確認大趨勢
        """
        df = dataframe.copy()
        df = df.set_index(DatetimeIndex(df['date']))
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        df = df.resample(str(int(interval[:-1]) * factor) + 'min',
                         label="right").agg(ohlc_dict).dropna(how='any')
        df['resample_sma'] = ta.SMA(df, timeperiod=25, price='close')
        df = df.drop(columns=['open', 'high', 'low', 'close'])
        df = df.resample(interval[:-1] + 'min')
        df = df.interpolate(method='time')
        df['date'] = df.index
        df.index = range(len(df))
        dataframe = merge(dataframe, df, on='date', how='left')
        return dataframe