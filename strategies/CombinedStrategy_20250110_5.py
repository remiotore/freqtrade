from freqtrade.strategy import IStrategy
from pandas import DataFrame
from datetime import datetime
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class CombinedStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Configurações gerais
    can_short = True
    timeframe = '5m'  # Timeframe principal
    informative_timeframe = '5m'  # Timeframe informativo adicional
    stoploss = -0.25

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    minimal_roi = {
        "0": 0.05,
        "20": 0.04,
        "30": 0.03,
        "60": 0.01
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Popula os indicadores necessários para a estratégia.
        """
        # Certifique-se de que o dataframe não está vazio
        if dataframe.empty:
            return dataframe

        # RSI (Índice de Força Relativa)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # MACD (Moving Average Convergence Divergence)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # ADX (Average Directional Index)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # Médias Móveis Exponenciais (EMA)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)

        # Gradiente do RSI (Gradiente da EMA do RSI)
        rsi_ema = ta.EMA(dataframe['rsi'], timeperiod=14)
        dataframe['rsi_gra'] = np.gradient(rsi_ema)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de entrada para long e short.
        """
        # Certifique-se de que o dataframe não está vazio
        if dataframe.empty:
            return dataframe

        # Entrada long (compra)
        dataframe.loc[
            (
                (dataframe['rsi'] > 50) &  # RSI acima de 50
                (dataframe['macd'] > dataframe['macdsignal']) &  # MACD cruzando para cima
                (dataframe['adx'] > 25) &  # ADX indicando tendência forte
                (dataframe['rsi_gra'] > 0)  # Gradiente positivo do RSI
            ),
            'enter_long'] = 1

        # Entrada short (venda)
        dataframe.loc[
            (
                (dataframe['rsi'] < 50) &  # RSI abaixo de 50
                (dataframe['macd'] < dataframe['macdsignal']) &  # MACD cruzando para baixo
                (dataframe['adx'] > 25) &  # ADX indicando tendência forte
                (dataframe['rsi_gra'] < 0)  # Gradiente negativo do RSI
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de saída para long e short.
        """
        # Certifique-se de que o dataframe não está vazio
        if dataframe.empty:
            return dataframe

        # Saída long
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) |  # RSI em sobrecompra
                qtpylib.crossed_below(dataframe['ema20'], dataframe['ema50'])  # EMA20 cruzando abaixo da EMA50
            ),
            'exit_long'] = 1

        # Saída short
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) |  # RSI em sobrevenda
                qtpylib.crossed_above(dataframe['ema20'], dataframe['ema50'])  # EMA20 cruzando acima da EMA50
            ),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stop-loss dinâmico baseado no lucro atual.
        """
        if current_profit > 0.3:
            return 0.01
        elif current_profit > 0.1:
            return 0.02
        elif current_profit > 0.05:
            return 0.05

        return 0.1
