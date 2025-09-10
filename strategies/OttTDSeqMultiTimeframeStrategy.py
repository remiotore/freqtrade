import talib
import pandas as pd

class OttTDSeqMultiTimeframeStrategy(IStrategy):
    INTERFACE_VERSION = 2
    
    # Parametreler
    timeframe_1 = "5m"  # 5 dakika
    timeframe_2 = "4h"  # 4 saat
    rsi_period = 14
    macd_short_period = 12
    macd_long_period = 26
    macd_signal_period = 9
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # RSI hesaplama (14 periyot)
        dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=self.rsi_period)
        
        # MACD hesaplama
        dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = talib.MACD(
            dataframe['close'],
            fastperiod=self.macd_short_period,
            slowperiod=self.macd_long_period,
            signalperiod=self.macd_signal_period
        )
        
        return dataframe
    
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 5 dakikalık zaman diliminde RSI ve MACD sinyalleri
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &  # Aşırı satım
                (dataframe['macd'] > dataframe['macd_signal'])  # MACD, sinyal çizgisinin üzerinde
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 5 dakikalık zaman diliminde RSI ve MACD sinyalleri
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) &  # Aşırı alım
                (dataframe['macd'] < dataframe['macd_signal'])  # MACD, sinyal çizgisinin altında
            ),
            'sell'] = 1
        return dataframe

    def populate_indicators_4h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 4 saatlik zaman diliminde RSI ve MACD hesaplama
        dataframe['rsi_4h'] = talib.RSI(dataframe['close'], timeperiod=self.rsi_period)
        dataframe['macd_4h'], dataframe['macd_signal_4h'], dataframe['macd_hist_4h'] = talib.MACD(
            dataframe['close'],
            fastperiod=self.macd_short_period,
            slowperiod=self.macd_long_period,
            signalperiod=self.macd_signal_period
        )
        return dataframe

    def populate_buy_trend_4h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 4 saatlik zaman diliminde RSI ve MACD sinyalleri
        dataframe.loc[
            (
                (dataframe['rsi_4h'] < 30) &  # Aşırı satım
                (dataframe['macd_4h'] > dataframe['macd_signal_4h'])  # MACD, sinyal çizgisinin üzerinde
            ),
            'buy_4h'] = 1
        return dataframe

    def populate_sell_trend_4h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 4 saatlik zaman diliminde RSI ve MACD sinyalleri
        dataframe.loc[
            (
                (dataframe['rsi_4h'] > 70) &  # Aşırı alım
                (dataframe['macd_4h'] < dataframe['macd_signal_4h'])  # MACD, sinyal çizgisinin altında
            ),
            'sell_4h'] = 1
        return dataframe
