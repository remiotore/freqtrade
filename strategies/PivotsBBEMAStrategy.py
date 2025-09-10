from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class PivotsBBEMAStrategy(IStrategy):
    """
    Strategie mit Pivot Points, Bollinger Bands und EMAs, einschließlich Erweiterungen.
    """
    # ROI und Stoploss
    minimal_roi = {"0": 0.10}
    stoploss = -0.10
    timeframe = '1h'

    # Parameter für Indikatoren
    pivot_lookback = 15  # Anzahl der Pivot-Punkte
    bb_length = 20  # Länge der Bollinger-Bänder
    bb_stddev = 2.0  # Standardabweichung für BB
    ema_lengths = [20, 50, 100, 200]  # EMA-Zeiträume
    ema_fast = 20  # Schnelle EMA für Crossovers
    ema_slow = 50  # Langsame EMA für Crossovers

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Berechnet Pivot Points, Bollinger Bands, EMAs und Fibonacci-Pivot-Levels.
        """
        # Pivot Points (lokale Hochs und Tiefs)
        dataframe['pivot_high'] = dataframe['high'].rolling(window=self.pivot_lookback).max()
        dataframe['pivot_low'] = dataframe['low'].rolling(window=self.pivot_lookback).min()

        # Fibonacci Pivot Points
        dataframe['fib_r1'] = dataframe['pivot_high'] + (dataframe['pivot_high'] - dataframe['pivot_low']) * 0.382
        dataframe['fib_s1'] = dataframe['pivot_low'] - (dataframe['pivot_high'] - dataframe['pivot_low']) * 0.382

        # Bollinger Bands
        dataframe['bb_middle'] = ta.SMA(dataframe['close'], timeperiod=self.bb_length)
        dataframe['bb_upper'] = dataframe['bb_middle'] + (ta.STDDEV(dataframe['close'], timeperiod=self.bb_length) * self.bb_stddev)
        dataframe['bb_lower'] = dataframe['bb_middle'] - (ta.STDDEV(dataframe['close'], timeperiod=self.bb_length) * self.bb_stddev)

        # EMAs
        for length in self.ema_lengths:
            dataframe[f'ema_{length}'] = ta.EMA(dataframe['close'], timeperiod=length)

        # EMA Crossovers
        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=self.ema_fast)
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=self.ema_slow)
        dataframe['ema_crossover'] = np.where(dataframe['ema_fast'] > dataframe['ema_slow'], 1, -1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generiert Kauf-Signale basierend auf erweiterten Indikatoren.
        """
        dataframe.loc[
            (dataframe['close'] > dataframe['pivot_high']) &  # Breakout über Pivot High
            (dataframe['close'] > dataframe['bb_upper']) &  # Bollinger Band Breakout
            (dataframe['ema_crossover'] > 0) &  # EMA-Trend bullisch (Crossover)
            (dataframe['close'] > dataframe['ema_20']) &  # Kurs oberhalb EMA 20
            (dataframe['close'] > dataframe['fib_r1']),  # Kurs über Fibonacci R1
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generiert Verkaufs-Signale basierend auf erweiterten Indikatoren.
        """
        dataframe.loc[
            (dataframe['close'] < dataframe['pivot_low']) &  # Breakout unter Pivot Low
            (dataframe['close'] < dataframe['bb_lower']) &  # Bollinger Band Breakout
            (dataframe['ema_crossover'] < 0) &  # EMA-Trend bärisch (Crossover)
            (dataframe['close'] < dataframe['ema_20']) &  # Kurs unterhalb EMA 20
            (dataframe['close'] < dataframe['fib_s1']),  # Kurs unter Fibonacci S1
            'exit_long'
        ] = 1

        return dataframe
