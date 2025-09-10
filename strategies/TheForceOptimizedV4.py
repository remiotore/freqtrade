import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class TheForceOptimizedV4(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.02,
        "15": 0.01,
        "30": 0.005
    }

    stoploss = -0.10  # Stoploss fixo de 10%
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015

    timeframe = '15m'

    # Definir um número suficiente de candles para calcular indicadores corretamente
    startup_candle_count: int = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula os indicadores técnicos usados na estratégia.
        """
        # Garantir que o dataframe tenha colunas essenciais
        if dataframe.empty or len(dataframe) < self.startup_candle_count:
            return dataframe

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe, fastk_period=5, slowk_period=3, slowd_period=3)
        dataframe['fastk'] = stoch_fast['slowk']  # slowk representa %K
        dataframe['fastd'] = stoch_fast['slowd']  # slowd representa %D

        # MACD
        macd = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # EMA
        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=50)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Volume Médio
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de compra.
        """
        dataframe.loc[
            (
                (dataframe['fastk'] > 20) & (dataframe['fastk'] < 80) &  # Stochastic em zona neutra
                (dataframe['fastd'] > 20) & (dataframe['fastd'] < 80) &
                (dataframe['macd'] > dataframe['macdsignal']) &  # MACD em alta
                (dataframe['ema5c'] > dataframe['ema50']) &  # Confirmação de tendência
                (dataframe['rsi'] > 40) & (dataframe['rsi'] < 70) &  # RSI em zona neutra
                (dataframe['volume'] > dataframe['volume_mean'])  # Volume acima da média
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de venda.
        """
        dataframe.loc[
            (
                (dataframe['fastk'] < 80) &
                (dataframe['fastd'] < 80) &
                (dataframe['macd'] < dataframe['macdsignal']) &  # MACD em baixa
                (dataframe['ema5c'] < dataframe['ema50']) &  # Reversão de tendência
                (dataframe['rsi'] < 40)  # RSI indicando perda de força
            ),
            'sell'] = 1
        return dataframe
