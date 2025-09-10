import numpy as np
import pandas as pd
import pandas_ta as ta
import talib.abstract as talib
from technical import qtpylib
from freqtrade.strategy import IStrategy, stoploss_from_open, stoploss_from_absolute
from freqtrade.persistence import Trade
from datetime import datetime


class VWAPStrategy_05(IStrategy):

    INTERFACE_VERSION = 2

    timeframe = '5m'

    minimal_roi = {
        "0": 1
    }

    custom_info = {}

    stoploss = -0.1

    use_custom_stoploss = True

    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = False


    def custom_stoploss(self, pair: str, Trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if Trade.id not in self.custom_info or not isinstance(self.custom_info[Trade.id], dict):
            self.custom_info[Trade.id] = {
                'stoploss': last_candle['ATR_stoploss'], 
                'updated_1_percent': False,
                'updated_2_percent': False
            }

        divided_current_profit = current_profit / 2

        if current_profit >= 0.02 and divided_current_profit > self.custom_info[Trade.id]['stoploss']:
            if not self.custom_info[Trade.id]['updated_2_percent']:
                self.custom_info[Trade.id]['stoploss'] = divided_current_profit
                self.custom_info[Trade.id]['updated_2_percent'] = True
                return stoploss_from_open(0.5, current_profit)
        
        elif current_profit >= 0.01 and divided_current_profit > self.custom_info[Trade.id]['stoploss']:
            if not self.custom_info[Trade.id]['updated_1_percent']:
                self.custom_info[Trade.id]['stoploss'] = divided_current_profit
                self.custom_info[Trade.id]['updated_1_percent'] = True
                return stoploss_from_open(0.5, current_profit)
        
        return self.custom_info[Trade.id]['stoploss']


    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['ATR'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=150)

        dataframe['ATR_stoploss'] = dataframe['close'] - dataframe['ATR'] * 2.5

        for length in [50, 200]:
            dataframe[f'ema{length}'] = talib.EMA(dataframe, length)

        dataframe['rsi'] = ta.rsi(dataframe['close'], length=16)

        dataframe['date_copy'] = pd.to_datetime(dataframe['date'])

        dataframe['date_copy'] = dataframe['date_copy'].dt.tz_localize(None)

        dataframe.set_index('date_copy', inplace=True)

        dataframe['VWAP'] = ta.vwap(dataframe['high'], dataframe['low'],
                                    dataframe['close'], dataframe['volume'], anchor='D', offset=None)

        sma = dataframe['close'].rolling(window=14).mean()

        std_dev = dataframe['close'].rolling(window=14).std()

        dataframe['upper_band'] = sma + (std_dev * 2.0)

        dataframe['lower_band'] = sma - (std_dev * 2.0)

        VWAP_signal = [0] * len(dataframe)
        backcandles = 15

        for row in range(backcandles, len(dataframe)):

            up_trend = 1
            down_trend = 1

            for i in range(row - backcandles, row + 1):

                if max(dataframe['open'][i], dataframe['close'][i]) >= dataframe['VWAP'][i]:
                    down_trend = 0  # Set down_trend to 0

                if min(dataframe['open'][i], dataframe['close'][i]) <= dataframe['VWAP'][i]:
                    up_trend = 0  # Set up_trend to 0

            if up_trend == 1 and down_trend == 1:
                VWAP_signal[row] = 3  # Neutral signal
            elif up_trend == 1:

                VWAP_signal[row] = 2
            elif down_trend == 1:

                VWAP_signal[row] = 1

        dataframe['VWAP_signal'] = VWAP_signal

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (
                (dataframe['volume'] > 0) &  # Buy when volume > 0
                (dataframe['VWAP_signal'] == 2) & # Buy when VWAP_signal is 2 (15 candles above VWAP line - indicating bullish trend)
                (dataframe['rsi'] < 45) & # Buy when rsi < 45
                (dataframe['close'] <= dataframe['lower_band']) & # Buy when the current closing price is less than or equal to the current lower bband
                (dataframe['close'].shift(1) <= dataframe['lower_band'].shift(1)) & # Buy when the previous closing price was less than or equal to the lower bband
                (dataframe['lower_band'] != dataframe['upper_band']) # Make sure lower and upper bband is not the same (no momentum)
            ),

            'buy'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (
                (dataframe['volume'] > 0) &  # Sell when volume > 0
                (dataframe['close'] >= dataframe['upper_band']) & # Sell when closing price is the same or above the upper bband
                (dataframe['rsi'] > 55) & # Sell when rsi > 55
                (dataframe['lower_band'] != dataframe['upper_band']) # Make sure lower and upper bband is not the same (no momentum)
            ),

            'sell'
        ] = 1

        return dataframe
