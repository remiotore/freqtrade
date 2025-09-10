from freqtrade.strategy import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
import pandas as pd
import talib.abstract as ta

class AdaptiveStrategy(IStrategy):
    timeframe = '15m'
    minimal_roi = {"0": 0.03, "30": 0.015}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    rsi_low = IntParameter(25, 40, default=35, space="buy")
    rsi_high = IntParameter(60, 80, default=70, space="sell")
    atr_multiplier = DecimalParameter(0.8, 2.0, default=1.2, space="buy")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)  # Грошовий потік
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']

        dataframe['atr_mean'] = dataframe['atr'].rolling(14).mean()
        dataframe['atr_mean'].fillna(method='bfill', inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe = dataframe.copy()

        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_low.value) &
                (dataframe['close'] > dataframe['ema50']) &
                (dataframe['ema50'] > dataframe['ema200']) &  # Додано підтвердження тренду
                (dataframe['macd'] > dataframe['macd_signal']) &  # MACD підтверджує зростання
                (dataframe['mfi'] > 20) &  # Відсікаємо ринок без ліквідності
                (dataframe['atr'] > dataframe['atr_mean'] * self.atr_multiplier.value)
            ),
            'buy'] = 1

        dataframe['buy'].fillna(0, inplace=True)
        dataframe['buy'] = dataframe['buy'].astype(int)

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe = dataframe.copy()

        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_high.value) &
                (dataframe['close'] < dataframe['ema50']) &
                (dataframe['macd'] < dataframe['macd_signal'])  # MACD показує спад
            ),
            'sell'] = 1

        dataframe['sell'].fillna(0, inplace=True)
        dataframe['sell'] = dataframe['sell'].astype(int)

        return dataframe