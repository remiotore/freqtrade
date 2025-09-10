# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import DecimalParameter
from technical.indicators import ichimoku
from freqtrade.persistence import Trade
import talib.abstract as ta
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

LOG_FILENAME = datetime.now().strftime('logfile_%d_%m_%Y.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s :: %(message)s')


class brawbot(IStrategy):
    """
    Estratégia otimizada para entradas cautelosas, evitando moedas com alta volatilidade
    e priorizando transações rápidas e seguras, com stoploss dinâmico.
    """
    # Configuração do tipo de ordem
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Retorno mínimo sobre o investimento (ROI)
    minimal_roi = {
        "60": 0.0002,
        "40": 0.0005,
        "20": 0.001,
        "10": 0.002,  # 0.2% após 10 minutos
        "5": 0.005,   # 0.5% após 5 minutos
        "0": 0.01     # 1% no final
    }

    # Configuração do timeframe
    timeframe = '5m'

    # Stoploss padrão
    stoploss = -0.10  # Início com -10%

    # Número máximo de trades abertos
    max_open_trades = 3  # Limitado a 3 trades para maior controle

    # Interface do Freqtrade (necessário usar versão 2)
    INTERFACE_VERSION = 2

    # Indicadores customizáveis via Hyperopt
    ema_short = DecimalParameter(5, 15, default=8, space='buy')
    ema_long = DecimalParameter(20, 50, default=21, space='buy')
    rsi_buy = DecimalParameter(30, 70, default=50, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Popula os indicadores técnicos usados pela estratégia.
        """
        # Indicadores Ichimoku
        ichi = ichimoku(dataframe)
        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['senkou_a'] = ichi['senkou_span_a']
        dataframe['senkou_b'] = ichi['senkou_span_b']

        # Médias móveis exponenciais (curta e longa)
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=int(self.ema_short.value))
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=int(self.ema_long.value))

        # Índice de Força Relativa (RSI)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ATR (Average True Range)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Variação máxima nas últimas 48h
        dataframe['price_change_48h'] = (dataframe['high'].rolling(288).max() - dataframe['low'].rolling(288).min()) / dataframe['low'].rolling(288).min()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de compra.
        """
        dataframe.loc[
            (
                # Filtro de volatilidade: ignora moedas com variação > 8% nas últimas 48h
                (dataframe['price_change_48h'] < 0.08) &
                # Condição 1: Cruzamento de EMA curta acima da EMA longa
                (dataframe['ema_short'] > dataframe['ema_long']) &
                (dataframe['close'] > dataframe['ema_short']) &  # Preço acima da EMA curta
                # Condição 2: RSI indicando força (compras em tendência)
                (dataframe['rsi'] > self.rsi_buy.value) &
                # Condição 3: Volume médio alto
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.2)  # Volume 20% maior que a média
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
  #      """
  #      Define as condições de venda.
  #      """
  #      dataframe.loc[
  #          (
  #              # Condição 1: RSI indicando sobrecompra
  #              (dataframe['rsi'] > 70) |
  #              # Condição 2: Cruzamento de EMA curta abaixo da EMA longa
  #              (dataframe['ema_short'] < dataframe['ema_long']) |
  #              # Condição 3: Preço abaixo da EMA curta (reversão de tendência)
  #              (dataframe['close'] < dataframe['ema_short'])
  #          ),
  #          'sell'] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Ajusta o stoploss dinamicamente com base no tempo.
        A cada dia, o stoploss aumenta de -10% para -8%, -6%, ..., até -1%.
        """
        max_stoploss = -0.10  # -10% (inicial)
        min_stoploss = -0.01  # Valor mínimo de stoploss (-1%)
        trade_duration = (current_time - trade.open_date_utc).total_seconds()

        # Calcular o número de dias de operação
        days_open = trade_duration / (24 * 60 * 60)

        # Ajuste progressivo do stoploss com base no tempo
        # Cada dia a mais, aumenta 2% (menos negativo)
        dynamic_stoploss = max_stoploss + days_open * 0.02

        # Garantir que o stoploss não ultrapasse o mínimo permitido
        dynamic_stoploss = min(dynamic_stoploss, min_stoploss)

        return dynamic_stoploss
