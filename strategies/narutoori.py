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
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

LOG_FILENAME = datetime.now().strftime('logfile_%d_%m_%Y.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s :: %(message)s')


class naruto(IStrategy):
    # Configuração do tipo de ordem
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    # Retorno mínimo sobre o investimento (ROI)
    minimal_roi = {
        "120": 0.001,
        "90": 0.0025,
        "60": 0.005,
        "30": 0.015,
        "15": 0.025,
        "0": 0.035,
    }

    # Configuração do timeframe
    timeframe = '5m'

    # Stoploss padrão
    stoploss = -0.05

    # Número máximo de trades abertos
    max_open_trades = 5

    # Interface do Freqtrade (necessário usar versão 2)
    INTERFACE_VERSION = 2

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
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']
        dataframe['chikou'] = ichi['chikou_span']

        # Indicadores adicionais
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # Índice de Força Relativa
        dataframe['ema8'] = ta.EMA(dataframe, timeperiod=8)  # Média móvel exponencial curta
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)  # Média móvel exponencial longa
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # Índice de força direcional
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)  # Volatilidade
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe)  # MACD

        return dataframe

    def is_bullish_engulfing(self, dataframe: DataFrame) -> pd.Series:
        """
        Identifica padrão de engolfo de alta no dataframe.
        """
        return (
            (dataframe['close'] > dataframe['open']) &  # Candle atual é de alta
            (dataframe['close'].shift(1) < dataframe['open'].shift(1)) &  # Candle anterior foi de baixa
            (dataframe['close'] > dataframe['open'].shift(1)) &  # Candle atual "engolfa" o anterior
            (dataframe['open'] < dataframe['close'].shift(1))
        )

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de compra aprimoradas.
        """
        dataframe.loc[
            (
                # Condição 1: Rompimento da Senkou B com confirmações
                (dataframe['open'].shift(1) < dataframe['senkou_b'].shift(1)) &  # Rompeu nuvem no candle anterior
                (dataframe['close'].shift(1) > dataframe['senkou_b'].shift(1)) &
                (dataframe['open'] > dataframe['senkou_b']) &                   # Confirmado no candle atual
                (dataframe['close'] > dataframe['senkou_b']) &
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean()) &  # Volume forte
                (dataframe['adx'] > 25) &                                       # ADX indicando força da tendência
                (dataframe['macd'] > dataframe['macdsignal'])                   # MACD indicando tendência de alta
                |
                # Condição 2: Cruzamento de EMA8 acima de EMA21 com RSI
                (dataframe['ema8'] > dataframe['ema21']) &                       # EMA8 cruzou acima da EMA21
                (dataframe['close'] > dataframe['ema8']) &                       # Preço acima da EMA8
                (dataframe['rsi'] > 50) &                                        # RSI indicando força
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean()) & # Volume forte
                (dataframe['adx'] > 25)                                          # ADX indicando tendência forte
                |
                # Condição 3: RSI em sobrevendido com reversão (filtro adicional)
                (dataframe['rsi'] < 30) &                                        # RSI sobrevendido
                (dataframe['close'] > dataframe['ema8']) &                       # Preço recupera acima da EMA8
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean()) & # Volume forte
                (dataframe['macd'] > dataframe['macdsignal']) &                  # MACD indicando reversão
                (dataframe['adx'] > 20)                                          # ADX indicando início de força
                |
                # Condição 4: Padrões de velas de reversão
                (self.is_bullish_engulfing(dataframe)) &                         # Padrão de engolfo de alta
                (dataframe['rsi'] > 40) &                                        # RSI confirma reversão
                (dataframe['adx'] > 20)                                          # ADX indicando início de força
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de venda.
        """
        dataframe.loc[
            (
                # Condição 1: Cruzamento para baixo da Senkou B com volume
                ((dataframe['open'] > dataframe['senkou_b']) & 
                 (dataframe['close'] < dataframe['senkou_b']) &
                 (dataframe['volume'] > dataframe['volume'].rolling(10).mean()))  # Volume alto
                |
                # Condição 2: RSI > 70 com fraqueza no preço
                ((dataframe['rsi'] > 70) & 
                 (dataframe['close'] < dataframe['ema8']))  # Preço abaixo da EMA8
                |
                # Condição 3: Cruzamento de EMA8 para baixo da EMA21 (com tolerância)
                ((dataframe['ema8'] < dataframe['ema21'] * 0.999) &
                 (dataframe['close'] < dataframe['ema21']))
            ),
            'sell'] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Ajusta o stoploss com base no número de trades abertos e na duração do trade.
        """
        max_stoploss = -0.05
        min_stoploss = -0.01
        reduction_per_day = 0.01

        open_trades = Trade.get_open_trades()
        if len(open_trades) >= self.max_open_trades:
            trade_duration = current_time - trade.open_date_utc
            trade_age_days = trade_duration.total_seconds() / (24 * 60 * 60)
            adjusted_stoploss = max_stoploss + (trade_age_days * reduction_per_day)
            adjusted_stoploss = max(adjusted_stoploss, min_stoploss)
            logger.info(f"[{pair}] Stoploss ajustado para {adjusted_stoploss:.4f} após {trade_age_days:.2f} dias.")
            return adjusted_stoploss

        return max_stoploss
