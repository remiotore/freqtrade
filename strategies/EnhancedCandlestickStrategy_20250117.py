import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

class EnhancedCandlestickStrategy(IStrategy):
    """
    Estratégia de trading com padrões de candlestick e indicadores técnicos ajustados
    para melhorar a assertividade e os lucros.
    """

    ########################################################################
    #                         PARÂMETROS HYPEROPT
    ########################################################################
    can_short = True
    timeframe = CategoricalParameter(
        ["5m", "15m", "1h"], default="5m", space="buy", optimize=True
    )

    leverage_level = IntParameter(1, 20, default=5, space="buy", optimize=True)
    bullish_threshold = IntParameter(5, 20, default=10, space="buy", optimize=True)
    bearish_threshold = IntParameter(-20, -5, default=-10, space="sell", optimize=True)

    stoploss_param = DecimalParameter(
        -0.25, -0.02, default=-0.05, decimals=2, space="sell", optimize=True
    )

    trailing_stop_param = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=True
    )
    trailing_stop_positive_param = DecimalParameter(
        0.001, 0.05, default=0.02, decimals=3, space="sell", optimize=True
    )
    trailing_stop_positive_offset_param = DecimalParameter(
        0.001, 0.10, default=0.03, decimals=3, space="sell", optimize=True
    )
    trailing_only_offset_is_reached_param = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=True
    )

    ema_short_period = IntParameter(10, 100, default=50, space="buy", optimize=True)
    ema_long_period = IntParameter(100, 300, default=200, space="buy", optimize=True)
    rsi_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    adx_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    atr_period = IntParameter(10, 30, default=14, space="buy", optimize=True)

    ########################################################################
    #                         CONSTRUTOR
    ########################################################################
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.stoploss = float(self.stoploss_param.value)
        self.trailing_stop = bool(self.trailing_stop_param.value)
        self.trailing_stop_positive = float(self.trailing_stop_positive_param.value)
        self.trailing_stop_positive_offset = float(
            self.trailing_stop_positive_offset_param.value
        )
        self.trailing_only_offset_is_reached = bool(
            self.trailing_only_offset_is_reached_param.value
        )

    ########################################################################
    #                         INDICADORES
    ########################################################################
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adiciona indicadores técnicos ao dataframe.
        """
        dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=self.ema_short_period.value)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self.ema_long_period.value)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Padrões de candlestick mais relevantes
        dataframe["engulfing"] = ta.CDLENGULFING(dataframe)
        dataframe["hammer"] = ta.CDLHAMMER(dataframe)
        dataframe["shooting_star"] = ta.CDLSHOOTINGSTAR(dataframe)

        return dataframe

    ########################################################################
    #                         ENTRADA
    ########################################################################
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de entrada com base nos indicadores e padrões.
        """
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # Condições para entradas long
        long_condition = (
            (dataframe["ema_short"] > dataframe["ema_long"]) &
            (dataframe["rsi"] < 30) &  # RSI em sobrevenda
            (dataframe["adx"] > 20) &  # Tendência moderada ou forte
            (
                (dataframe["engulfing"] > 0) |  # Padrão engolfo de alta
                (dataframe["hammer"] > 0)  # Padrão martelo
            )
        )

        # Condições para entradas short
        short_condition = (
            (dataframe["ema_short"] < dataframe["ema_long"]) &
            (dataframe["rsi"] > 70) &  # RSI em sobrecompra
            (dataframe["adx"] > 20) &  # Tendência moderada ou forte
            (
                (dataframe["engulfing"] < 0) |  # Padrão engolfo de baixa
                (dataframe["shooting_star"] < 0)  # Padrão estrela cadente
            )
        )

        dataframe.loc[long_condition, "enter_long"] = 1
        dataframe.loc[short_condition, "enter_short"] = 1

        return dataframe

    ########################################################################
    #                         SAÍDA
    ########################################################################
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de saída com base em indicadores e trailing stop.
        """
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Condições para saída de long
        exit_long_condition = (
            (dataframe["rsi"] > 70) |  # RSI indicando sobrecompra
            (dataframe["close"] > dataframe["ema_long"])  # Preço muito acima da média
        )

        # Condições para saída de short
        exit_short_condition = (
            (dataframe["rsi"] < 30) |  # RSI indicando sobrevenda
            (dataframe["close"] < dataframe["ema_long"])  # Preço muito abaixo da média
        )

        dataframe.loc[exit_long_condition, "exit_long"] = 1
        dataframe.loc[exit_short_condition, "exit_short"] = 1

        return dataframe

    ########################################################################
    #                         ALAVANCAGEM
    ########################################################################
    def leverage(
        self, pair: str, current_time, current_rate, proposed_leverage, max_leverage, side, **kwargs
    ) -> float:
        """
        Ajusta a alavancagem com base na volatilidade (ATR).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe["atr"].iloc[-1]

        if atr > 0.01:  # Exemplo: ajuste conforme a volatilidade
            return max(1, max_leverage / 2)
        return min(max_leverage, proposed_leverage)
