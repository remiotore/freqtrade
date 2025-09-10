
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta



class ADXMomentumOtimizeHyperOpt(IStrategy):


    minimal_roi = {
        "0": 0.279,
        "28": 0.081,
        "48": 0.031,
        "150": 0
    }

    stoploss = -0.219

    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    timeframe = '5m'

    startup_candle_count: int = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14) #(Índice Direcional Médio)indicador de tendencia se mercado está em alta ou baixa
        dataframe['di +'] = ta.PLUS_DI(dataframe, timeperiod=25) #Positive Directional Indicator(Quando a linha de crescimento do DI+ fica acima da linha do DI-, a indicação é de alta, podendo ser considerado como um sinal para comprar ações)
        dataframe['di -'] = ta.MINUS_DI(dataframe, timeperiod=25) #Negative Directional Indicator(Quando a linha de crescimento do DI- fica acima da linha do DI+, a indicação é de baixa, podendo ser considerado como um sinal para vender ações)
        dataframe['sar'] = ta.SAR(dataframe)#Stop and Reverse "Parada e Reversão", ou seja, o indicador pode determinar a tendência e, além disso, sinalizar a hora de fechar a operação em cima dessa tendência e olhar na direção oposta)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14) #mede o quanto o preço de uma ação mudou durante um certo período de tempo. Momento = Preço de Fechamento hoje - Preço de Fechamento n dias atrás.indicador da velocidade do mercado.

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > 25) &
                    (dataframe['mom'] > 0) &
                    (dataframe['di +'] > 25) &
                    (dataframe['di +'] > dataframe['di -'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > 25) &
                    (dataframe['mom'] < 0) &
                    (dataframe['di -'] > 25) &
                    (dataframe['di +'] < dataframe['di -'])

            ),
            'sell'] = 1
        return dataframe