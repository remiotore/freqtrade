from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class ConsecutiveDipFuturesStrategyFutures(IStrategy):
    """
    Estratégia para operar no mercado de futures, comprando e vendendo pares
    com base em velas consecutivas negativas e outros indicadores técnicos.
    """

    # Configurações gerais
    timeframe = "15m"  # Timeframe de 1 hora
    startup_candle_count = 50  # Número de candles necessários antes de começar
    can_short = True  # Suporte para operações de venda (short)

    # ROI (Return on Investment)
    minimal_roi = {
        "0": 0.05,  # 5% de retorno mínimo
        "60": 0.03,  # 3% após 1 hora
        "120": 0.02,  # 2% após 2 horas
        "180": 0.01,  # 1% após 3 horas
        "240": 0  # Sem limite após 4 horas
    }

    # Stoploss fixo
    stoploss = -0.10  # 10% de stop-loss (mais conservador para futures)

    # Trailing stop para proteger lucros
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    # Parâmetros otimizáveis
    rsi_threshold = IntParameter(10, 50, default=30, space="buy", optimize=True)  # RSI de entrada
    consecutive_candles = IntParameter(3, 6, default=4, space="buy", optimize=True)  # Número de velas negativas consecutivas
    atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)  # Multiplicador do ATR para stop-loss dinâmico

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adiciona indicadores ao DataFrame.
        """
        # RSI para detectar sobrevenda
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # ATR para medir volatilidade
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # ADX para confirmar força da tendência
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # Média do volume para identificar picos
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de entrada para compra e venda.
        """
        consecutive = self.consecutive_candles.value

        # Condições de compra (long)
        dataframe["consecutive_dip"] = True
        for i in range(1, consecutive + 1):
            dataframe["consecutive_dip"] &= dataframe["close"].shift(i) < dataframe["close"].shift(i - 1)

        dataframe.loc[
            (
                dataframe["consecutive_dip"] &
                (dataframe["rsi"] < self.rsi_threshold.value) &
                (dataframe["volume"] > dataframe["volume_mean"]) &
                (dataframe["adx"] > 25)  # Tendência forte
            ),
            "enter_long"
        ] = 1

        # Condições de venda (short)
        dataframe["consecutive_rise"] = True
        for i in range(1, consecutive + 1):
            dataframe["consecutive_rise"] &= dataframe["close"].shift(i) > dataframe["close"].shift(i - 1)

        dataframe.loc[
            (
                dataframe["consecutive_rise"] &
                (dataframe["rsi"] > 100 - self.rsi_threshold.value) &
                (dataframe["volume"] > dataframe["volume_mean"]) &
                (dataframe["adx"] > 25)  # Tendência forte
            ),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de saída.
        """
        # Saída de posições long
        dataframe.loc[
            (
                (dataframe["rsi"] > 70) |  # RSI indicando sobrecompra
                (dataframe["close"] > (dataframe["close"] + dataframe["atr"] * self.atr_multiplier.value))  # Preço acima do ATR ajustado
            ),
            "exit_long"
        ] = 1

        # Saída de posições short
        dataframe.loc[
            (
                (dataframe["rsi"] < 30) |  # RSI indicando sobrevenda
                (dataframe["close"] < (dataframe["close"] - dataframe["atr"] * self.atr_multiplier.value))  # Preço abaixo do ATR ajustado
            ),
            "exit_short"
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Ajusta a alavancagem dinamicamente com base no ATR (volatilidade).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe["atr"].iloc[-1]

        # Maior volatilidade = menor alavancagem
        dynamic_leverage = max(1.0, min(max_leverage, 2.0 / atr))
        return dynamic_leverage
