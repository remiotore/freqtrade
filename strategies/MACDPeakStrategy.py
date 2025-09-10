# --- Dołącz bibliotek ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class MACDPeakStrategy(IStrategy):
    # Parametry strategii
    minimal_roi = {
        "0": 0.1,  # Minimalny zysk 10%
    }
    stoploss = -0.1  # Maksymalna strata 10%
    timeframe = '5m'  # Interwał czasowy świec

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Oblicza wskaźniki wykorzystywane w strategii i dodaje je do dataframe.
        """
        # Oblicz RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # Oblicz MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Określa warunki kupna.
        """
        dataframe.loc[
            (dataframe['rsi'] < 35) &  # RSI poniżej 35
            (dataframe['macd'] > dataframe['macdsignal']) &  # MACD powyżej linii sygnałowej
            (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)),  # Przecięcie MACD
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Określa warunki sprzedaży (na szczycie MACD i RSI > 65).
        """
        dataframe.loc[
            (dataframe['macd'] > dataframe['macd'].shift(1)) &  # MACD rośnie
            (dataframe['macd'].shift(-1) < dataframe['macd']) &  # MACD spada w kolejnej świecy
            (dataframe['rsi'] > 65),  # RSI powyżej 65
            'sell'] = 1
        return dataframe

