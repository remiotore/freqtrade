
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import pandas_ta as pta


class ImpulseMACDVolumeStrategy(IStrategy):
    # Minimalne i maksymalne parametry par handlowych
    minimal_roi = {"0": 0.1}  # Minimalny zwrot inwestycji
    stoploss = -0.05  # Maksymalna strata
    timeframe = "5m"  # Interwał czasowy

    # Włącz lub wyłącz użycie wskaźników dodatkowych
    use_custom_stoploss = False
    process_only_new_candles = True

    # Wczytanie danych historycznych
    startup_candle_count = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Wskaźnik MACD (Impulse MACD)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Volume-based Support & Resistance (symulacja przez średnią objętość)
        dataframe["volume_avg"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_supp_res"] = (dataframe["volume"] > dataframe["volume_avg"]).astype(int)

        # Momentum (używając RSI jako pomocniczego wskaźnika dla momentum)
        dataframe["rsi"] = ta.RSI(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sygnał zakupu, gdy MACD przecina sygnał od dołu i objętość jest większa niż średnia
        dataframe.loc[
            (
                (dataframe["macd"] > dataframe["macdsignal"])  # MACD sygnał wzrostu
                & (dataframe["volume"] > dataframe["volume_avg"])  # Zwiększona objętość
                & (dataframe["rsi"] > 50)  # RSI wskazuje momentum
            ),
            "buy",
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sygnał sprzedaży, gdy MACD przecina sygnał od góry lub objętość spada poniżej średniej
        dataframe.loc[
            (
                (dataframe["macd"] < dataframe["macdsignal"])  # MACD sygnał spadku
                | (dataframe["volume"] < dataframe["volume_avg"])  # Zmniejszona objętość
            ),
            "sell",
        ] = 1
        return dataframe
