from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import pandas as pd

class Futures10PercentStrategy(IStrategy):
    # Timeframe usado para análise
    timeframe = '1h'

    # Estratégia suporta short
    can_short = True

    # Stoploss fixo
    stoploss = -0.05  # 5% de stoploss

    # Retorno sobre o investimento
    minimal_roi = {
        "0": 0.1  # ROI mínimo de 10%
    }

    # Parâmetros configuráveis
    drop_threshold = DecimalParameter(0.05, 0.20, default=0.10, space="buy", decimals=2)  # Threshold de queda para compra
    rise_threshold = DecimalParameter(0.05, 0.20, default=0.10, space="sell", decimals=2)  # Threshold de alta para venda (short)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calcula indicadores personalizados para a estratégia.
        """
        # Variação percentual em relação ao fechamento das últimas 24 horas (24 candles de 1h)
        dataframe['price_change_pct'] = (dataframe['close'] - dataframe['close'].shift(24)) / dataframe['close'].shift(24) * 100
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define as condições de entrada para compra (long) e venda (short).
        """
        # Condição para compra (long)
        dataframe.loc[
            (dataframe['price_change_pct'] <= -self.drop_threshold.value),  # Queda maior que o threshold
            'buy'] = 1

        # Condição para venda (short)
        dataframe.loc[
            (dataframe['price_change_pct'] >= self.rise_threshold.value),  # Alta maior que o threshold
            'sell'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define as condições de saída para posições long e short.
        """
        # Sair de posições long quando atingir ROI ou stoploss
        dataframe.loc[
            dataframe['close'] > dataframe['close'].shift(1),  # Preço começou a subir
            'exit_long'] = 1

        # Sair de posições short quando atingir ROI ou stoploss
        dataframe.loc[
            dataframe['close'] < dataframe['close'].shift(1),  # Preço começou a cair
            'exit_short'] = 1

        return dataframe
