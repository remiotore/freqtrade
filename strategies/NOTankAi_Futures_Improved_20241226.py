# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class NOTankAi_Futures_Improved(IStrategy):
    """
    Estratégia ajustada para maior assertividade e lucratividade:
    - Simplificação das condições de entrada e saída.
    - Redução do stop-loss para limitar perdas.
    - Uso de trailing stop para proteger lucros.
    - Ajustes de ROI para ciclos de trades mais curtos.
    """

    # Timeframe principal
    timeframe = "15m"

    # Configurações de Stop-Loss
    stoploss = -0.2

    # Configuração de ROI
    minimal_roi = {
        "0": 0.05,  # 5% de retorno imediato
        "30": 0.04,  # 4% após 30 minutos
        "60": 0.03,  # 3% após 60 minutos
        "120": 0,    # Sem ROI mínimo após 2 horas
    }

    # Configuração de trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Número de candles necessários para iniciar a estratégia
    startup_candle_count = 96

    # Condição para processar apenas novos candles
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adiciona os indicadores necessários ao DataFrame.
        """
        # Indicador RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Média do volume (usada para filtro de liquidez)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        # Média móvel simples (para saída adicional)
        dataframe['sma'] = ta.SMA(dataframe['close'], timeperiod=50)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições para entrada em posições (long e short).
        """
        # Entrada em posição Long
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &  # RSI indicando sobrevenda
                (dataframe['volume'] > dataframe['volume_mean'] * 1.5)  # Volume acima de 1,5x a média
            ),
            ['enter_long', 'enter_tag']
        ] = (1, "RSI Oversold + High Volume")

        # Entrada em posição Short
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) &  # RSI indicando sobrecompra
                (dataframe['volume'] > dataframe['volume_mean'] * 1.5)  # Volume acima de 1,5x a média
            ),
            ['enter_short', 'enter_tag']
        ] = (1, "RSI Overbought + High Volume")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições para saída de posições (long e short).
        """
        # Saída de Long
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) |  # RSI indicando sobrecompra
                (qtpylib.crossed_below(dataframe['close'], dataframe['sma']))  # Cruzamento abaixo da SMA
            ),
            ['exit_long', 'exit_tag']
        ] = (1, "RSI Overbought or SMA Cross")

        # Saída de Short
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) |  # RSI indicando sobrevenda
                (qtpylib.crossed_above(dataframe['close'], dataframe['sma']))  # Cruzamento acima da SMA
            ),
            ['exit_short', 'exit_tag']
        ] = (1, "RSI Oversold or SMA Cross")

        return dataframe
