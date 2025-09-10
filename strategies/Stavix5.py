import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, merge_informative_pair
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Stavix5(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {'0': 100}

    stoploss = -0.31

    trailing_stop = True
    trailing_stop_positive = 0.28622
    trailing_stop_positive_offset = 0.3654
    trailing_only_offset_is_reached = True

    timeframe = '15m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 124

    order_types = {
            'buy': 'market',
            'sell': 'market',
            'stoploss': 'market',
            'stoploss_on_exchange': True
            }

    order_time_in_force = {
            'buy': 'gtc',
            'sell': 'gtc'
            }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def ema(self, values, window):
        weights = np.exp(np.linspace(-1.,0.,window))
        weights /= weights.sum()
        a = np.convolve(values, weights) [:len(values)]
        a[:window] = a[window]
        return a

    def wwma(self,values, n):
        """
        J. Welles Wilder's EMA 
        """
        return values.ewm(alpha=1/n, adjust=False).mean()

    def atr(self, df, n):
        data = df.copy()
        high = data['high']
        low = data['low']
        close = data['close']
        data['tr0'] = abs(high - low)
        data['tr1'] = abs(high - close.shift())
        data['tr2'] = abs(low - close.shift())
        tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)

        atr = self.wwma(tr, n)
        return atr

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['atr'] = self.atr(dataframe, 192)
        multiplier = 27
        dataframe['avg_price'] = (dataframe['high']  + dataframe['low']) / 2
        dataframe['upper_band'] = dataframe['avg_price'] + (multiplier * dataframe['atr'])
        dataframe['lower_band'] = dataframe['avg_price'] - (multiplier * dataframe['atr'])
        dataframe['final_upper'] = 0

        for i, row in enumerate(dataframe.itertuples(), 0):
            if i == 0:
                continue
            if row.upper_band < dataframe['final_upper'].iloc[i-1] or dataframe['avg_price'].iloc[i-1] > dataframe['final_upper'].iloc[i-1]:
                dataframe['final_upper'].iloc[i] = row.upper_band
            else:
                dataframe['final_upper'].iloc[i] = dataframe['final_upper'].iloc[i-1]

        dataframe['final_lower'] = 0

        for i, row in enumerate(dataframe.itertuples(), 0):
            if i == 0:
                continue
            if row.lower_band > dataframe['final_lower'].iloc[i-1] or dataframe['avg_price'].iloc[i-1] < dataframe['final_lower'].iloc[i-1]:
                dataframe['final_lower'].iloc[i] = row.lower_band
            else:
                dataframe['final_lower'].iloc[i] = dataframe['final_lower'].iloc[i-1]

        dataframe['super_trend'] = 0

        for i, row in enumerate(dataframe.itertuples(), 0):
            if i == 0:
                continue
            if dataframe['super_trend'].iloc[i-1] == dataframe['final_upper'].iloc[i-1] and row.avg_price <= row.final_upper:
                dataframe['super_trend'].iloc[i] = row.final_upper
            elif dataframe['super_trend'].iloc[i-1] == dataframe['final_upper'].iloc[i-1] and row.avg_price > row.final_upper:
                dataframe['super_trend'].iloc[i] = row.final_lower
            elif dataframe['super_trend'].iloc[i-1] == dataframe['final_lower'].iloc[i-1] and row.avg_price >= row.final_lower:
                dataframe['super_trend'].iloc[i] = row.final_lower
            elif dataframe['super_trend'].iloc[i-1] == dataframe['final_lower'].iloc[i-1] and row.avg_price < row.final_lower:
                dataframe['super_trend'].iloc[i] = row.final_upper

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
                (   
                    (qtpylib.crossed_above(dataframe['avg_price'],  dataframe['super_trend'])) &
                    (dataframe['volume'] > 0)
                ),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
                (   
                    (qtpylib.crossed_above(dataframe['super_trend'],  dataframe['avg_price'])) &
                    (dataframe['volume'] > 0)
                ),
                'sell'] = 1

        return dataframe
