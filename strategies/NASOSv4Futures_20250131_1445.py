from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import stoploss_from_open
from functools import reduce
import logging

logger = logging.getLogger(__name__)

class NASOSv4Futures(IStrategy):
    INTERFACE_VERSION = 3

    # ✅ Ativar suporte para Futures (Long e Short)
    can_short = True  

    # ✅ ROI otimizado para Futures (lucro rápido e seguro)
    minimal_roi = {
        "0": 0.04,   # 4% imediato
        "30": 0.03,  # 3% após 30 min
        "120": 0.02, # 2% após 2h
        "240": 0     # Sem limite após 4h
    }

    # ✅ Stoploss reduzido para evitar liquidações
    stoploss = -0.025  # Apenas 2.5% de perda

    # ✅ Trailing Stop ativado para capturar lucros
    trailing_stop = True
    trailing_stop_positive = 0.015  # 1.5%
    trailing_stop_positive_offset = 0.02  # 2%
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    startup_candle_count = 50  

    # ✅ Definição dos tipos de ordem para evitar slippage
    order_types = {
        "buy": "market",
        "sell": "market",
        "stoploss": "market",
        "forcebuy": "market",
        "stoploss_on_exchange": False
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calcula indicadores técnicos essenciais para Futures."""

        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=20).mean()

        # ✅ Proteção contra volatilidade extrema
        dataframe["atr_percent"] = dataframe["atr"] / dataframe["close"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define regras para abrir posições LONG e SHORT."""

        # ✅ Condições para entrada em LONG (compra)
        long_conditions = [
            (dataframe["rsi"] < 50),  
            (dataframe["volume"] > dataframe["volume_mean"] * 0.8),  
            (dataframe["atr_percent"] < dataframe["atr_percent"].rolling(window=50).mean() * 2),  
            (dataframe["close"] > dataframe["ema_50"]) | (dataframe["close"] < dataframe["ema_50"] * 1.02)
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, long_conditions), "enter_long"
        ] = 1

        # ✅ Condições para entrada em SHORT (venda primeiro)
        short_conditions = [
            (dataframe["rsi"] > 60),  
            (dataframe["volume"] > dataframe["volume_mean"] * 0.8),  
            (dataframe["atr_percent"] < dataframe["atr_percent"].rolling(window=50).mean() * 2),  
            (dataframe["close"] < dataframe["ema_50"]) | (dataframe["close"] > dataframe["ema_50"] * 0.98)
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, short_conditions), "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define regras para fechar posições LONG e SHORT."""

        # ✅ Condições para saída de LONG (venda)
        long_exit_conditions = [
            (dataframe["rsi"] > 70),
            (dataframe["close"] < dataframe["ema_50"]),
            (dataframe["atr_percent"] > dataframe["atr_percent"].rolling(window=50).mean() * 1.5)
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, long_exit_conditions), "exit_long"
        ] = 1

        # ✅ Condições para saída de SHORT (recompra)
        short_exit_conditions = [
            (dataframe["rsi"] < 40),
            (dataframe["close"] > dataframe["ema_50"]),
            (dataframe["atr_percent"] > dataframe["atr_percent"].rolling(window=50).mean() * 1.5)
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, short_exit_conditions), "exit_short"
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """Ajusta a alavancagem dinamicamente com base no ATR."""

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe["atr"].iloc[-1]

        # ✅ Maior volatilidade = menor alavancagem
        dynamic_leverage = max(1.0, min(max_leverage, 2.0 / atr))
        return dynamic_leverage
