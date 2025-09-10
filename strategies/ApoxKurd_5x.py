import logging
import talib.abstract as ta
import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class Apoxkurd_1(IStrategy):
    timeframe = '15m'
    can_short = True

    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.09
    trailing_only_offset_is_reached = True

    stoploss = -0.021

    minimal_roi = {
        "0": 0.064,
        "118": 0.042,
        "247": 0.019,
        "403": 0,
    }

    use_exit_signal = True
    exit_profit_only = True
    position_adjustment_enable = True
    max_dca_multiplier = 4

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['adx'] = ta.ADX(df, timeperiod=14)
        df['pdi'] = ta.PLUS_DI(df, timeperiod=14)
        df['mdi'] = ta.MINUS_DI(df, timeperiod=14)
        df['rsi'] = ta.RSI(df, timeperiod=14)
        df['sma200'] = ta.SMA(df, timeperiod=200)

        # Fractal-based extrema (No lookahead bias)
        df['fractal_low'] = (
            (df['low'].shift(2) > df['low']) &
            (df['low'].shift(1) > df['low']) &
            (df['low'].shift(-1) > df['low']) &
            (df['low'].shift(-2) > df['low'])
        ).astype(int)

        df['fractal_high'] = (
            (df['high'].shift(2) < df['high']) &
            (df['high'].shift(1) < df['high']) &
            (df['high'].shift(-1) < df['high']) &
            (df['high'].shift(-2) < df['high'])
        ).astype(int)

        # VWAP Band
        vwap_low, vwap_mid, vwap_high = self.VWAPB(df, 20, 1)
        df['vwap_low'] = vwap_low
        df['vwap_mid'] = vwap_mid
        df['vwap_high'] = vwap_high

        # Chaikin Money Flow
        df['cmf'] = self.chaikin_mf(df, periods=20)

        # Heikin Ashi
        ha = qtpylib.heikinashi(df)
        df['ha_close'] = ha['close']
        df['ha_open'] = ha['open']

        # Murrey Math Levels
        murrey_levels = self.calculate_murrey_math_levels(df)
        for level, value in murrey_levels.items():
            df[level] = value

        return df

    def VWAPB(self, dataframe, window_size=20, num_of_std=1):
        df = dataframe.copy()
        df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
        rolling_std = df['vwap'].rolling(window=window_size).std()
        df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
        df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
        return df['vwap_low'], df['vwap'], df['vwap_high']

    def chaikin_mf(self, df, periods=20):
        close = df["close"]
        low = df["low"]
        high = df["high"]
        volume = df["volume"]
        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = mfv.fillna(0.0) * volume
        cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
        return pd.Series(cmf, name="cmf")

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        logger.info(f"{metadata['pair']} Checking entry conditions...")

        df['enter_long'] = 0
        df['enter_short'] = 0
        df['enter_tag'] = ''

        conditions_long = (
            ((df['pdi'] > df['mdi']) & (df['adx'] > 21)) &
            (df['close'] > df['sma200']) &
            (df['fractal_low'] == 1)
        )
        df.loc[conditions_long, ['enter_long', 'enter_tag']] = (1, 'Fractal Long Entry')

        conditions_short = (
            ((df['mdi'] > df['pdi']) & (df['adx'] > 21)) &
            (df['close'] < df['sma200']) &
            (df['fractal_high'] == 1)
        )
        df.loc[conditions_short, ['enter_short', 'enter_tag']] = (1, 'Fractal Short Entry')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = 0
        df['exit_short'] = 0
        df['exit_tag'] = ''

        conditions_long_exit = (
            (df['adx'] < 29) &
            (df['rsi'] > 53)
        )
        df.loc[conditions_long_exit, ['exit_long', 'exit_tag']] = (1, 'Exit Long Enhanced')

        conditions_short_exit = (
            (df['adx'] < 29) &
            (df['rsi'] < 45)
        )
        df.loc[conditions_short_exit, ['exit_short', 'exit_tag']] = (1, 'Exit Short Enhanced')

        return df

    @staticmethod
    def calculate_murrey_math_levels(df, window_size=64):
        rolling_max_H = df["high"].rolling(window=window_size).max()
        rolling_min_L = df["low"].rolling(window=window_size).min()

        max_H = rolling_max_H
        min_L = rolling_min_L
        range_HL = max_H - min_L

        murrey_math_levels = {}

        for i in range(len(df)):
            mn = min_L.iloc[i]
            mx = max_H.iloc[i]
            dmml = (mx - mn) / 8
            levels = {f"[{int(i - 4)}/8]P": mn + j * dmml for j in range(0, 9)}
            for k, v in levels.items():
                murrey_math_levels.setdefault(k, []).append(v)

        return {k: pd.Series(v) for k, v in murrey_math_levels.items()}

    def leverage(self, pair: str, current_time: datetime, **kwargs) -> float:
        return 5.0