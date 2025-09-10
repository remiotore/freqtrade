import numpy as np
import pandas as pd
import pandas_ta as ta
from technical import qtpylib
from freqtrade.strategy import IStrategy, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime





class VWAPStrategy_1(IStrategy):
        
    custom_stoploss_per_trade = {}
    recent_trades = set()

    INTERFACE_VERSION = 2

    timeframe = '5m'

    minimal_roi = {
    "0": 1
    }

    stoploss = -0.2


    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = False
        

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:

        if trade.id in self.custom_stoploss_per_trade:

            if current_profit >= 0.02:
                updated_stoploss = current_profit / 2

                if updated_stoploss < self.custom_stoploss_per_trade[trade.id]:
                    self.custom_stoploss_per_trade[trade.id] = updated_stoploss

                    return updated_stoploss

            elif current_profit >= 0.01:
                updated_stoploss = current_profit / 2

                if updated_stoploss < self.custom_stoploss_per_trade[trade.id]:
                    self.custom_stoploss_per_trade[trade.id] = updated_stoploss

                    return updated_stoploss

            return self.custom_stoploss_per_trade[trade.id]

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if pd.notna(last_candle['buy_atr']):

            ratio = last_candle['buy_atr'] / trade.open_rate
            stoploss_from_open = trade.open_rate * (1 - ratio)
            self.custom_stoploss_per_trade[trade.id] = stoploss_from_open

            return stoploss_from_open


        return self.stoploss


    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.set_index(pd.DatetimeIndex(dataframe["date"]), inplace=True)

        dataframe['ATR'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=150)

        dataframe['ATR-close'] = dataframe['close'] - (dataframe['ATR'] * 3)

        dataframe['rsi'] = ta.rsi(dataframe['close'], length=16)

        dataframe['VWAP'] = ta.vwap(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], anchor='D', offset=None)

        b_bands = ta.bbands(dataframe['close'], length=14, std=2.0)
        dataframe = dataframe.join(b_bands)

        VWAP_signal = [0] * len(dataframe)
        backcandles = 15

        for row in range(backcandles, len(dataframe)):

            up_trend = 1
            down_trend = 1

            for i in range(row - backcandles, row + 1):

                if max(dataframe['open'][i], dataframe['close'][i]) >= dataframe['VWAP'][i]:
                    down_trend = 0 # Set down_trend to 0

                if min(dataframe['open'][i], dataframe['close'][i]) <= dataframe['VWAP'][i]:
                    up_trend = 0 # Set up_trend to 0

            if up_trend == 1 and down_trend == 1:
                VWAP_signal[row] = 3 # Neutral signal
            elif up_trend == 1:
                VWAP_signal[row] = 2 # Upward trend signal - 15 candles above the VWAP line
            elif down_trend == 1:
                VWAP_signal[row] = 1 # Downward trend signal - 15 candles below the VWAP line

        dataframe['VWAP_signal'] = VWAP_signal

        return dataframe

    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        condition = (
            (dataframe['volume'] > 0) & # Buy when volume > 0
            (dataframe['VWAP_signal'] == 2) & # Buy when VWAP_signal is 2 (15 candles above VWAP line - indicating bullish trend)
            (dataframe['rsi'] < 45) & # Buy when rsi < 45
            (dataframe['close'] <= dataframe['BBL_14_2.0']) & # Buy when the current closing price is less than or equal to the current lower bband
            (dataframe['close'].shift(1) <= dataframe['BBL_14_2.0'].shift(1)) & # Buy when the previous closing price was less than or equal to the previous lower bband
            (dataframe['BBL_14_2.0'] != dataframe['BBU_14_2.0']) # Make sure lower and upper bband is not the same (no momentum)
        )


        dataframe.loc[condition, 'buy'] = 1
        dataframe.loc[condition, 'buy_atr'] = dataframe.loc[condition, 'ATR-close']

        return dataframe


    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (
                (dataframe['volume'] > 0) & # Sell when volume > 0
                (dataframe['VWAP_signal'] == 2) & # Sell when VWAP_signal is 2 (15 candles above VWAP line - indicating bullish trend)
                (dataframe['close'] >= dataframe['BBU_14_2.0']) & # Sell when closing price is the same or more than upper bband'
                (dataframe['rsi'] > 55) & # Sell when rsi > 55
                (dataframe['rsi'] <= 90) & # Sell when rsi <= 90
                (dataframe['BBL_14_2.0'] != dataframe['BBU_14_2.0']) # Make sure lower and upper bband is not the same (no momentum)
            ),

            'sell'
        ] = 1

        return dataframe
