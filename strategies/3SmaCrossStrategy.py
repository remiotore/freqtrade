from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class ThreeSmaCrossStrategy(IStrategy):
    INTERFACE_VERSION = 2
    can_short = True
    timeframe = '1m'

    # Parámetros de riesgo y recompensa
    stoploss = -0.99  # Stop loss del 1%
    minimal_roi = {"0": 0.02}  # Take profit del 3%

    # Parámetros de las SMAs
    sma_fast_period = 2
    sma_normal_period = 5
    sma_slow_period = 10
    lookback_period = 20  # Período para contar velas verdes y rojas en la ventana reciente
    prior_period = 20  # Período para contar velas verdes y rojas en la ventana anterior
    large_candle_multiplier = 1  # Factor para definir una vela grande
    cooldown_candle_count = 3  # Número de velas de espera entre entradas consecutivas

    # Exit signal
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Cálculo de SMAs
        dataframe['sma_fast'] = ta.SMA(dataframe['close'], timeperiod=self.sma_fast_period)
        dataframe['sma_slow'] = ta.SMA(dataframe['close'], timeperiod=self.sma_slow_period)
        dataframe['sma_normal'] = ta.SMA(dataframe['close'], timeperiod=self.sma_normal_period)

        # Determinar si cada vela es verde (cierre > apertura) o roja (cierre < apertura)
        dataframe['is_green'] = dataframe['close'] > dataframe['open']
        dataframe['is_red'] = dataframe['close'] < dataframe['open']

        # Contar el número de velas verdes y rojas en el período de lookback y prior
        dataframe['green_count_lookback'] = dataframe['is_green'].rolling(window=self.lookback_period).sum()
        dataframe['red_count_lookback'] = dataframe['is_red'].rolling(window=self.lookback_period).sum()

        # Ventana anterior al período de lookback
        dataframe['green_count_prior'] = dataframe['is_green'].shift(self.lookback_period).rolling(window=self.prior_period).sum()
        dataframe['red_count_prior'] = dataframe['is_red'].shift(self.lookback_period).rolling(window=self.prior_period).sum()

        # Identificar velas grandes
        dataframe['candle_range'] = dataframe['high'] - dataframe['low']
        dataframe['avg_candle_range'] = dataframe['candle_range'].rolling(window=self.lookback_period).mean()
        dataframe['is_large_candle'] = dataframe['candle_range'] > (dataframe['avg_candle_range'] * self.large_candle_multiplier)

        # Contar el número de velas grandes en el período de lookback
        dataframe['large_candle_count'] = dataframe['is_large_candle'].rolling(window=self.lookback_period).sum()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Inicializar columnas para evitar el KeyError
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['candle_count_since_last_entry'] = 0

        # Contador de velas desde la última entrada
        for i in range(1, len(dataframe)):
            # Si la última entrada fue long o short, reinicia el contador
            if dataframe.at[i - 1, 'enter_long'] == 1 or dataframe.at[i - 1, 'enter_short'] == 1:
                dataframe.at[i, 'candle_count_since_last_entry'] = 1
            else:
                # Caso contrario, incrementa el contador
                dataframe.at[i, 'candle_count_since_last_entry'] = dataframe.at[i - 1, 'candle_count_since_last_entry'] + 1

        # Condición para permitir entrada si se ha cumplido el cooldown
        can_enter = dataframe['candle_count_since_last_entry'] >= self.cooldown_candle_count

        # Señal de compra (largo): SMA rápida cruza por encima de la SMA lenta, más velas verdes que rojas en el período anterior, velas grandes y respeto al cooldown
        dataframe.loc[
            (dataframe['sma_fast'] > dataframe['sma_slow']) &
            (dataframe['sma_fast'].shift(1) <= dataframe['sma_slow'].shift(1)) &  # Cruce alcista
            (dataframe['sma_normal'] > dataframe['sma_slow']) &
            (dataframe['sma_normal'].shift(1) <= dataframe['sma_slow'].shift(1)) &
            (dataframe['green_count_prior'] > dataframe['red_count_prior']) &  # Más velas verdes que rojas en el período anterior
            (dataframe['large_candle_count'] > 0) &  # Al menos una vela grande en el período de lookback
            can_enter,
            'enter_long'
        ] = 1

        # Señal de venta en corto: SMA rápida cruza por debajo de la SMA lenta, más velas rojas que verdes en el período anterior, velas grandes y respeto al cooldown
        dataframe.loc[
            (dataframe['sma_fast'] < dataframe['sma_slow']) &
            (dataframe['sma_fast'].shift(1) >= dataframe['sma_slow'].shift(1)) &  # Cruce bajista
            (dataframe['sma_normal'] < dataframe['sma_slow']) &
            (dataframe['sma_normal'].shift(1) >= dataframe['sma_slow'].shift(1)) &
            (dataframe['red_count_prior'] > dataframe['green_count_prior']) &  # Más velas rojas que verdes en el período anterior
            (dataframe['large_candle_count'] > 0) &  # Al menos una vela grande en el período de lookback
            can_enter,
            'enter_short'
        ] = 1

        # Reiniciar el contador de velas después de una entrada
        dataframe.loc[dataframe['enter_long'] == 1, 'candle_count_since_last_entry'] = 0
        dataframe.loc[dataframe['enter_short'] == 1, 'candle_count_since_last_entry'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Cierre de posición en largo: SMA rápida cruza por debajo de la SMA lenta y más velas rojas que verdes
        dataframe.loc[
            (dataframe['sma_fast'] < dataframe['sma_slow']) &
            (dataframe['sma_fast'].shift(1) >= dataframe['sma_slow'].shift(1)) &  # Cruce bajista
            (dataframe['sma_normal'] < dataframe['sma_slow']) &
            (dataframe['sma_normal'].shift(1) >= dataframe['sma_slow'].shift(1)) &
            (dataframe['red_count_lookback'] > dataframe['green_count_lookback']),  # Más velas rojas que verdes en lookback
            'exit_long'
        ] = 1

        # Cierre de posición en corto: SMA rápida cruza por encima de la SMA lenta y más velas verdes que rojas
        dataframe.loc[
            (dataframe['sma_fast'] > dataframe['sma_slow']) &
            (dataframe['sma_fast'].shift(1) <= dataframe['sma_slow'].shift(1)) &  # Cruce alcista
            (dataframe['sma_normal'] > dataframe['sma_slow']) &
            (dataframe['sma_normal'].shift(1) <= dataframe['sma_slow'].shift(1)) &
            (dataframe['green_count_lookback'] > dataframe['red_count_lookback']),  # Más velas verdes que rojas en lookback
            'exit_short'
        ] = 1

        return dataframe
