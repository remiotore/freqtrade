import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class TheForceOptimized(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.02,
        "15": 0.01,
        "30": 0.005
    }

    # Stoploss aumentado para 25%
    stoploss = -0.25
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015

    timeframe = '15m'

    startup_candle_count: int = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Adiciona indicadores técnicos ao dataframe."""
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 3)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['macd'], dataframe['macdsignal'], _ = ta.MACD(dataframe['close'], 12, 26, 9)
        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define as condições de compra."""
        dataframe.loc[
            (
                (dataframe['fastk'] > 20) & (dataframe['fastk'] < 80) &
                (dataframe['fastd'] > 20) & (dataframe['fastd'] < 80) &
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['ema5c'] > dataframe['ema50']) &
                (dataframe['rsi'] > 40) & (dataframe['rsi'] < 70) &
                (dataframe['volume'] > dataframe['volume_mean'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define as condições de venda."""
        dataframe.loc[
            (
                (dataframe['fastk'] < 80) &
                (dataframe['fastd'] < 80) &
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['ema5c'] < dataframe['ema50']) &
                (dataframe['rsi'] < 40)
            ),
            'sell'] = 1
        return dataframe
