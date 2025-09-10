import numpy as np
import pandas as pd
import pandas_ta as ta
import talib.abstract as talib
from technical import qtpylib
from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime


class trailingTrendStrategy(IStrategy):

    INTERFACE_VERSION = 2

    timeframe = '15m'

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.2

    use_custom_stoploss = True

    exit_profit_only = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_profit: float, current_rate: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        if trade.stop_loss == trade.initial_stop_loss:


            atr_stoploss = stoploss_from_absolute(last_candle['ATR_stoploss'], current_rate)
            if np.isnan(atr_stoploss):
                return None
            else:
                return atr_stoploss
            
        if current_profit > 0.01:


            divided_profit = current_profit / 2
            return stoploss_from_open(divided_profit, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        return trade.stop_loss

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['date_copy'] = pd.to_datetime(dataframe['date'])

        dataframe['date_copy'] = dataframe['date_copy'].dt.tz_localize(None)

        dataframe.set_index('date_copy', inplace=True)

        dataframe['VWAP'] = ta.vwap(dataframe['high'], dataframe['low'],
                                    dataframe['close'], dataframe['volume'], anchor='D', offset=None)

        dataframe['ATR'] = ta.atr(
            dataframe['high'], dataframe['low'], dataframe['close'], length=150)

        dataframe['ATR_stoploss'] = dataframe['close'] - dataframe['ATR'] * 6.5

        dataframe['rsi'] = ta.rsi(dataframe['close'], length=16)

        sma = dataframe['close'].rolling(14).mean()

        std_dev = dataframe['close'].rolling(14).std()

        dataframe['upper_band'] = sma + (std_dev * 2.0)

        dataframe['lower_band'] = sma - (std_dev * 2.0)

        VWAP_signal = [0] * len(dataframe)
        backcandles = 10

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

                (dataframe['volume'] > 0) &

                (dataframe['VWAP_signal'] == 2) &

                (dataframe['rsi'] < 45) &

                (dataframe['close'] <= dataframe['lower_band']) &

                (dataframe['close'].shift(1) <= dataframe['lower_band'].shift(1)) &

                (dataframe['lower_band'] != dataframe['upper_band'])
            ),

            'buy'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (

                (dataframe['volume'] > 0) &

                (dataframe['close'] >= dataframe['upper_band']) &

                (dataframe['rsi'] > 60) &

                (dataframe['lower_band'] != dataframe['upper_band'])
            ),

            'sell'
        ] = 1

        return dataframe
