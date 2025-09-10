from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class SwingTradingDmiStrategy(IStrategy):
    """
    Swing Trading Strategy mit Parabolic SAR, ATR, MACD, SMA und DMI.
    """
    # ROI und Stoploss
    minimal_roi = {"0": 0.03}  # Ziel: 3% Gewinn
    stoploss = -0.1  # Maximaler Verlust: 10%
    timeframe = '1h'

    # Parameter für Indikatoren
    sar_start = 0.02  # SAR Startwert
    sar_increment = 0.02  # SAR Steigerungsrate
    sar_max = 0.2  # SAR Maximalwert
    atr_length = 10  # ATR Periode
    macd_short = 12  # MACD Kurzfristige Periode
    macd_long = 26  # MACD Langfristige Periode
    macd_signal = 9  # MACD Signalperiode
    sma_length = 50  # SMA Periode
    dmi_length = 14  # DMI Periode
    adx_smoothing = 14  # ADX Glättungsperiode
    target_profit_percentage = 3.0  # Zielgewinn in Prozent

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Berechnet Indikatoren für die Strategie.
        """
        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe['high'], dataframe['low'], acceleration=self.sar_increment, maximum=self.sar_max)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=self.atr_length)

        # MACD
        macd = ta.MACD(dataframe['close'], fastperiod=self.macd_short, slowperiod=self.macd_long, signalperiod=self.macd_signal)
        dataframe['macd_line'] = macd['macd']
        dataframe['macd_signal_line'] = macd['macdsignal']

        # SMA
        dataframe['sma'] = ta.SMA(dataframe['close'], timeperiod=self.sma_length)

        # DMI und ADX
        dmi = ta.DMI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=self.dmi_length)
        dataframe['plus_di'] = dmi['plus_di']
        dataframe['minus_di'] = dmi['minus_di']
        dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=self.adx_smoothing)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generiert Kauf-Signale basierend auf den berechneten Indikatoren.
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['sar']) &  # Kurs oberhalb des SAR
                (dataframe['close'] > dataframe['sma']) &  # Kurs oberhalb der SMA
                (dataframe['plus_di'] > dataframe['minus_di']) &  # DMI bullisch
                (dataframe['adx'] > 25)  # ADX zeigt starken Trend
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generiert Verkaufs-Signale basierend auf den Indikatoren.
        """
        dataframe.loc[
            (
                (dataframe['macd_line'] < dataframe['macd_signal_line']) |  # MACD-Sell-Signal
                (dataframe['close'] < dataframe['sar'])  # Kurs unterhalb des SAR
            ),
            'exit_long'
        ] = 1

        return dataframe
