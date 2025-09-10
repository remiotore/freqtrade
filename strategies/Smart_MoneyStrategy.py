from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class SmartMoneyStrategy(IStrategy):
    # ROI und Stoploss
    minimal_roi = {"0": 0.10}
    stoploss = -0.10

    # Zeitrahmen
    timeframe = '1h'

    # Parameter für Indikatoren
    orderblock_atr_multiplier = 2  # Multiplikator für ATR zur Orderblock-Erkennung
    swing_length = 10  # Länge des Fensters für Equal Highs/Lows
    fvg_threshold = 0.5  # Schwelle für Fair Value Gaps
    eqhl_threshold = 0.1  # Schwelle für Equal High/Low-Identifikation

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Berechnet Indikatoren für die Strategie.
        """
        # ATR-Berechnung
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)

        # Gleichwertige Hochs (Equal Highs) und Tiefs (Equal Lows)
        dataframe['eq_high'] = (
            (dataframe['high'] == dataframe['high'].rolling(self.swing_length).max())
        ).astype(int)
        dataframe['eq_low'] = (
            (dataframe['low'] == dataframe['low'].rolling(self.swing_length).min())
        ).astype(int)

        # Fair Value Gaps (FVG)
        dataframe['fvg_up'] = (
            (dataframe['low'].shift(1) > dataframe['high'].shift(2)) &
            (dataframe['close'] > dataframe['high'].shift(2))
        ).astype(int)
        dataframe['fvg_down'] = (
            (dataframe['high'].shift(1) < dataframe['low'].shift(2)) &
            (dataframe['close'] < dataframe['low'].shift(2))
        ).astype(int)

        # Orderblocks
        dataframe = self.calculate_orderblocks(dataframe)

        return dataframe

    def calculate_orderblocks(self, dataframe: DataFrame) -> DataFrame:
        """
        Berechnet bullische und bärische Orderblocks basierend auf ATR und Preisbewegungen.
        """
        dataframe['bullish_ob'] = (
            (dataframe['low'] < dataframe['low'].rolling(5).min()) &
            (dataframe['close'] > dataframe['open']) &
            ((dataframe['high'] - dataframe['low']) < dataframe['atr'] * self.orderblock_atr_multiplier)
        ).astype(int)

        dataframe['bearish_ob'] = (
            (dataframe['high'] > dataframe['high'].rolling(5).max()) &
            (dataframe['close'] < dataframe['open']) &
            ((dataframe['high'] - dataframe['low']) < dataframe['atr'] * self.orderblock_atr_multiplier)
        ).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Kauf-Signale basierend auf Fair Value Gaps, Orderblocks und Break of Structure.
        """
        dataframe.loc[
            (dataframe['eq_low'] > 0) &  # Equal Low vorhanden
            (dataframe['fvg_up'] > 0) &  # Bullischer FVG
            (dataframe['bullish_ob'] > 0) &  # Bullischer Orderblock
            (dataframe['close'] > dataframe['high'].shift(1)),  # Breakout
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Verkaufs-Signale basierend auf Fair Value Gaps, Orderblocks und Break of Structure.
        """
        dataframe.loc[
            (dataframe['eq_high'] > 0) &  # Equal High vorhanden
            (dataframe['fvg_down'] > 0) &  # Bärischer FVG
            (dataframe['bearish_ob'] > 0) &  # Bärischer Orderblock
            (dataframe['close'] < dataframe['low'].shift(1)),  # Breakout nach unten
            'exit_long'
        ] = 1
        return dataframe
