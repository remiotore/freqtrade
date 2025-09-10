from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import numpy as np

class Zone1(IStrategy):
    INTERFACE_VERSION = 3

    stoploss = -0.5
    can_short = True
    trailing_stop = True
    trailing_stop_positive = 0.5
    trailing_stop_positive_offset = 0.6
    timeframe = '5m'

    mult = DecimalParameter(0.1, 5.0, default=1.0, space='buy', decimals=1)
    left = IntParameter(1, 50, default=10, space='buy')
    right = IntParameter(1, 50, default=1, space='buy')
    min_bars = IntParameter(0, 50, default=0, space='buy')
    max_bars = IntParameter(50, 1000, default=500, space='buy')
    set_back = True
    display = 'Bullish AND Bearish'

    def pivot_point(self, series: DataFrame, left: int, right: int, is_high: bool) -> DataFrame:
        """
        Визначення Pivot точок.
        :param series: Серія з цінами (високими або низькими)
        :param left: Кількість лівих барів для перевірки
        :param right: Кількість правих барів для перевірки
        :param is_high: Якщо True, шукаємо високі точки (Pivot High), інакше низькі (Pivot Low)
        :return: Серія з Pivot точками
        """
        if is_high:
            pivots = series == series.rolling(window=left + right + 1, center=True).max()
        else:
            pivots = series == series.rolling(window=left + right + 1, center=True).min()
        
        return series.where(pivots)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        atr_period = 200
        dataframe['atr'] = dataframe['close'].rolling(atr_period).apply(
            lambda x: np.mean(np.abs(np.diff(x))), raw=False
        ) * self.mult.value

        dataframe['pivot_high'] = self.pivot_point(dataframe['high'], self.left.value, self.right.value, True)
        dataframe['pivot_low'] = self.pivot_point(dataframe['low'], self.left.value, self.right.value, False)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        bull_state = 0
        box_bottom = box_top = 0

        for i in range(1, len(dataframe)):
            if bull_state == 0 and not np.isnan(dataframe['pivot_low'].iloc[i]):
                bull_state = 1
                box_bottom = dataframe['pivot_low'].iloc[i]
                box_top = dataframe['pivot_low'].iloc[i] + dataframe['atr'].iloc[i]
            elif bull_state == 1:
                if dataframe['close'].iloc[i] > box_top and dataframe['open'].iloc[i] < box_top:
                    dataframe.loc[i, 'enter_long'] = 1
                    bull_state = 0
                elif dataframe['close'].iloc[i] < box_bottom:
                    box_bottom = dataframe['pivot_low'].iloc[i]
                    box_top = dataframe['pivot_low'].iloc[i] + dataframe['atr'].iloc[i]
                elif i - np.argmax(dataframe['enter_long'].values[:i][::-1]) > self.max_bars.value:
                    bull_state = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_short'] = 0
        bear_state = 0
        box_top = box_bottom = 0

        for i in range(1, len(dataframe)):
            if bear_state == 0 and not np.isnan(dataframe['pivot_high'].iloc[i]):
                bear_state = 1
                box_top = dataframe['pivot_high'].iloc[i]
                box_bottom = dataframe['pivot_high'].iloc[i] - dataframe['atr'].iloc[i]
            elif bear_state == 1:
                if dataframe['close'].iloc[i] < box_bottom and dataframe['open'].iloc[i] > box_bottom:
                    dataframe.loc[i, 'enter_short'] = 1
                    bear_state = 0
                elif dataframe['close'].iloc[i] > box_top:
                    box_top = dataframe['pivot_high'].iloc[i]
                    box_bottom = dataframe['pivot_high'].iloc[i] - dataframe['atr'].iloc[i]
                elif i - np.argmax(dataframe['enter_short'].values[:i][::-1]) > self.max_bars.value:
                    bear_state = 0

        return dataframe
